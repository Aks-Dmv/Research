# '''
# Code from spinningup repo.
# Refer[Original Code]: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
# '''

from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import time
import sys
import sac.sac_agent

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, device=torch.device('cpu'), size=int(1e6)):
        self.state = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.next_state = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.action = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.reward = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device
        # print(device)

    def store_batch(self, obs, act, rew, next_obs, done):
        num = len(obs)
        full =  self.ptr + num > self.max_size
        if not full:
            self.state[self.ptr: self.ptr + num] = obs
            self.next_state[self.ptr: self.ptr + num] = next_obs
            self.action[self.ptr: self.ptr + num] = act
            self.reward[self.ptr: self.ptr + num] = rew
            self.done[self.ptr: self.ptr + num] = done
            self.ptr = self.ptr + num
        else:
            idx = np.arange(self.ptr,self.ptr+num)%self.max_size
            self.state[idx] = obs
            self.next_state[idx]=next_obs
            self.action[idx]=act
            self.reward[idx]=rew
            self.done[idx]=done
            self.ptr= (self.ptr+num)%self.max_size            
            

        self.size = min(self.size + num, self.max_size)

    def store(self, obs, act, rew, next_obs, done):
        self.state[self.ptr] = obs
        self.next_state[self.ptr] = next_obs
        self.action[self.ptr] = act
        self.reward[self.ptr] = rew
        self.done[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.state[idxs],
                     obs2=self.next_state[idxs],
                     act=self.action[idxs],
                     rew=self.reward[idxs],
                     done=self.done[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in batch.items()}


class SAC:

    def __init__(self, state_dims, encoder,num_atoms, k=1, actor_critic=sac.sac_agent.MLPActorCritic, ac_kwargs=dict(), seed=0, 
            steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, add_time=False,
            polyak=0.995, lr=0.00005, alpha=0.2, batch_size=100, start_steps=10000, update_num=20,
            update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
            log_step_interval=None, reward_state_indices=None,
            save_freq=1, device=torch.device("cpu"), automatic_alpha_tuning=True, reinitialize=True,
            **kwargs):
        """
        Soft Actor-Critic (SAC)
        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.
            actor_critic: The constructor method for a PyTorch Module with an ``act`` 
                method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
                The ``act`` method and ``pi`` module should accept batches of 
                observations as inputs, and ``q1`` and ``q2`` should accept a batch 
                of observations and a batch of actions as inputs. When called, 
                ``act``, ``q1``, and ``q2`` should return:
                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                            | observation.
                ``q1``       (batch,)          | Tensor containing one current estimate
                                            | of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ``q2``       (batch,)          | Tensor containing the other current 
                                            | estimate of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ===========  ================  ======================================
                Calling ``pi`` should return:
                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                            | given observations.
                ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                            | actions in ``a``. Importantly: gradients
                                            | should be able to flow back into ``a``.
                ===========  ================  ======================================
            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
                you provided to SAC.
            seed (int): Seed for random number generators.
            steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
                for the agent and the environment in each epoch.
            epochs (int): Number of epochs to run and train agent.
            replay_size (int): Maximum length of replay buffer.
            gamma (float): Discount factor. (Always between 0 and 1.)
            polyak (float): Interpolation factor in polyak averaging for target 
                networks. Target networks are updated towards main networks 
                according to:
                .. math:: \\theta_{\\text{targ}} \\leftarrow 
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
                where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
                close to 1.)
            lr (float): Learning rate (used for both policy and value learning).
            alpha (float): Entropy regularization coefficient. (Equivalent to 
                inverse of reward scale in the original SAC paper.)
            batch_size (int): Minibatch size for SGD.
            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.
            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.
            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long 
                you wait between updates, the ratio of env steps to gradient steps 
                is locked to 1.
            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.
            max_ep_len (int): Maximum length of trajectory / episode / rollout.
            logger_kwargs (dict): Keyword args for EpochLogger.
            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.
        """

        self.obs_dim = state_dims
        self.act_dim = state_dims
        self.max_ep_len=max_ep_len
        self.start_steps=start_steps
        self.batch_size=batch_size
        self.gamma=gamma
        self.encoder_params = encoder.parameters()
        self.encoder = encoder
        
        self.polyak=polyak
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = 1.0
        self.steps_per_epoch = steps_per_epoch
        self.update_after = update_after
        self.update_every = update_every
        self.udpate_num = update_num
        self.num_test_episodes = num_test_episodes
        self.epochs = epochs
        # Create actor-critic module and target networks
        self.ac = actor_critic(self.obs_dim, self.act_dim, num_atoms, self.act_limit, k, add_time=add_time, device=device, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters(), self.encoder_params)
        self.pi_params = itertools.chain(self.ac.pi.parameters(), self.encoder_params)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        self.var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.pi_params, lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        self.device = device

        self.automatic_alpha_tuning = automatic_alpha_tuning
        if self.automatic_alpha_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(self.obs_dim).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha

        self.true_state_dim = self.obs_dim

        if log_step_interval is None:
            log_step_interval = steps_per_epoch
        self.log_step_interval = log_step_interval
        self.reinitialize = reinitialize

        self.reward_function = None
        self.reward_state_indices = reward_state_indices

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data, edges, rel_rec, rel_send, prediction_steps, next_st, r, deterministic=False):
        q1 = self.ac.q1(data, edges, rel_rec, rel_send, prediction_steps, next_st)
        q2 = self.ac.q2(data, edges, rel_rec, rel_send, prediction_steps, next_st)

        d = 0.

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(data, edges, rel_rec, rel_send, prediction_steps, deterministic, True)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(data, edges, rel_rec, rel_send, prediction_steps, next_st)
            q2_pi_targ = self.ac_targ.q2(data, edges, rel_rec, rel_send, prediction_steps, next_st)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data, edges, rel_rec, rel_send, prediction_steps, deterministic=False):
        pi, logp_pi = self.ac.pi(data, edges, rel_rec, rel_send, prediction_steps, deterministic, True)
        q1_pi = self.ac.q1(data, edges, rel_rec, rel_send, prediction_steps, pi)
        q2_pi = self.ac.q2(data, edges, rel_rec, rel_send, prediction_steps, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        return loss_pi, logp_pi


    # Set up model saving

    def update(self, data, edges, rel_rec, rel_send, prediction_steps, obs2, r):
        # called by adv IRL
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data, edges, rel_rec, rel_send, prediction_steps, obs2, r)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, log_pi = self.compute_loss_pi(data, edges, rel_rec, rel_send, prediction_steps)
        loss_pi.backward()
        self.pi_optimizer.step()

        if self.automatic_alpha_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        return np.array([loss_q.item(), loss_pi.item(), log_pi.detach().cpu().mean().item()])

    def get_action(self, data, edges, rel_rec, rel_send, prediction_steps, deterministic=False, get_logprob=False):
        if len(data.shape) < 2:
            data = data[None, :]
        return self.ac.act(data, edges, rel_rec, rel_send, prediction_steps, deterministic, get_logprob=get_logprob)
        # return self.ac.act(torch.as_tensor(o[:, :self.true_state_dim], dtype=torch.float32).to(self.device), 
        #             deterministic, get_logprob=get_logprob)

    def get_action_batch(self, data, edges, rel_rec, rel_send, prediction_steps, deterministic=False, get_logprob=False):
        if len(data.shape) < 2:
            data = data[None, :]
        return self.ac.act_batch(data, edges, rel_rec, rel_send, prediction_steps, deterministic, get_logprob=get_logprob)
        # return self.ac.act_batch(torch.as_tensor(o[:, :self.true_state_dim], dtype=torch.float32).to(self.device), deterministic)

    def init_cuda(self):
        self.ac.q1.cuda()
        self.ac.q2.cuda()
        self.ac.pi.cuda()
        self.ac_targ.q1.cuda()
        self.ac_targ.q2.cuda()
        self.ac_targ.pi.cuda()
        
    def init_multiple_cuda(self):
        self.ac.q1 = torch.nn.DataParallel(self.ac.q1)
        self.ac.q2 = torch.nn.DataParallel(self.ac.q2)
        self.ac.pi = torch.nn.DataParallel(self.ac.pi)
        self.ac_targ.q1 = torch.nn.DataParallel(self.ac_targ.q1)
        self.ac_targ.q2 = torch.nn.DataParallel(self.ac_targ.q2)
        self.ac_targ.pi = torch.nn.DataParallel(self.ac_targ.pi)


    def init_eval(self):
        self.ac.q1.eval()
        self.ac.q2.eval()
        self.ac.pi.eval()
        self.ac_targ.q1.eval()
        self.ac_targ.q2.eval()
        self.ac_targ.pi.eval()

    def reset(self):
        pass


    @property
    def networks(self):
        return [self.ac.pi, self.ac.q1, self.ac.q2, self.encoder]
