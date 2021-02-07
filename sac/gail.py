# '''
# Code built on top of https://github.com/KamyarGh/rl_swiss 
# Refer[Original Code]: https://github.com/KamyarGh/rl_swiss
# '''
# rl_swiss/rlkit/torch/state_marginal_matching/adv_smm.py
# rl_swiss/rlkit/core/base_algorithm.py

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch import autograd
import torch.nn.functional as F
from sac.sac import ReplayBuffer
from matplotlib import pyplot as plt
import os, copy
import os.path as osp

class AdvSMM:
    '''
        Features removed from v1.0:
            - gradient clipping
            - target disc (exponential moving average disc)
            - target policy (exponential moving average policy)
            - disc input noise
    '''
    def __init__(
        self,
        num_atoms,
        inp_dims, 
        discriminator,
        agent, # e.g. SAC
        state_indices, # omit timestamp
        target_state_buffer, # from sampling method (e.g. rejection sampling)
        replay_buffer_size,
        policy_optim_batch_size,
        num_edges,
        
        num_epochs=400,
        num_steps_per_epoch=60000,
        min_steps_before_training=1200, 
        num_disc_updates_per_loop_iter=5,
        num_policy_updates_per_loop_iter=2,
        num_initial_disc_iters=100,

        disc_optim_batch_size=128,
        disc_lr=0.0005,
        disc_momentum=0.0,
        disc_optimizer_class=optim.Adam,
        use_grad_pen=True,
        grad_pen_weight=1.0, # NOTE

        reward_scale=0.25,  # NOTE this is 1/alpha
        rew_clip_min=None,
        rew_clip_max=None,

        device=torch.device("cpu")
    ):  

        self.num_atoms=num_atoms
        self.inp_dims=inp_dims
        self.replay_obs_dim = self.inp_dims*self.num_atoms
        self.num_edges = num_edges
        self.replay_edge_dim = self.num_atoms*(self.num_atoms-1)*self.num_edges

        self.device = device

        self.state_indices = torch.LongTensor(state_indices).to(device) # for Disciminator observation space
        self.target_state_buffer = target_state_buffer
        self.target_edges_buffer = None
        self.agent = agent

        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(
                                self.replay_obs_dim, 
                                self.replay_edge_dim,
                                device=device,
                                size=replay_buffer_size)

        # discriminator
        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer_class(
            self.discriminator.parameters(),
            lr=disc_lr,
            betas=(disc_momentum, 0.999)
        )
        self.bce = nn.BCEWithLogitsLoss().to(device)      

        self.bce_targets = torch.cat(
            [
                torch.ones(disc_optim_batch_size, 1),
                torch.zeros(disc_optim_batch_size, 1)
            ],
            dim=0
        ).to(device)

        self.disc_optim_batch_size = disc_optim_batch_size
        self.policy_optim_batch_size = policy_optim_batch_size

        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight

        self.rew_clip_min = rew_clip_min
        self.rew_clip_max = rew_clip_max
        self.reward_scale = reward_scale

        self._n_train_steps_total = 0
        self.not_done_initial_disc_iters = True
        # remove initial training of discriminator, we do not have expert samples anyway.
        self.max_real_return_det, self.max_real_return_sto = -np.inf, -np.inf

        self.num_epochs = num_epochs
        self.num_steps_per_epoch = num_steps_per_epoch
        self.max_path_length = self.agent.max_ep_len
        self.min_steps_before_training = min_steps_before_training
        self.num_disc_updates_per_loop_iter = num_disc_updates_per_loop_iter
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter
        self.num_initial_disc_iters = num_initial_disc_iters


    def train(self, data, edges, rel_rec, rel_send, prediction_steps, epoch):
        # based on common/sac.py learn()

        output = self.agent.get_action_batch(data, edges, rel_rec, rel_send, prediction_steps)

        output = np.transpose(output, (0, 2, 1, 3))
        numb_repeats = output.shape[1] - 1
        ob1 = output[:,:-1,:,:].reshape((-1, self.replay_obs_dim))
        ob2 = output[:,1:,:,:].reshape((-1, self.replay_obs_dim))
        
        edges_np = edges.cpu().data.numpy().reshape((edges.shape[0],-1))
        edges_np = np.repeat(edges_np, numb_repeats, 0)
        
        self.replay_buffer.store_batch( ob1, edges_np, np.ones(ob1.shape[0]), ob2, np.zeros(ob1.shape[0]) )

        self._try_to_train(rel_rec, rel_send, prediction_steps, epoch)


    def _try_to_train(self, rel_rec, rel_send, prediction_steps, epoch):
        if self.replay_buffer.size >= self.min_steps_before_training:
            self.training_mode(True)
            self._do_training(rel_rec, rel_send, 1, epoch)
            self._n_train_steps_total += 1
            self.training_mode(False)

    def _do_training(self, rel_rec, rel_send, prediction_steps, epoch):
        if self.not_done_initial_disc_iters:
            for _ in range(self.num_initial_disc_iters):
                self._do_reward_training(rel_rec, rel_send, prediction_steps, epoch)
            self.not_done_initial_disc_iters = False

        disc_stat = np.zeros((3,))
        policy_stat = np.zeros((7,))
        for _ in range(self.num_disc_updates_per_loop_iter):
            disc_stat += self._do_reward_training(rel_rec, rel_send, prediction_steps, epoch)
        for _ in range(self.num_policy_updates_per_loop_iter):
            policy_stat += self._do_policy_training(rel_rec, rel_send, prediction_steps, epoch)
        print("disc/policy stat ", disc_stat[1], policy_stat[1])

    def update_target_batch(self, new_target_batch):
        s1 = new_target_batch.transpose(1, 2).contiguous()
        s2 = torch.clone(s1)

        s1 = torch.reshape(s1[:,:-1,:,:], (-1, self.replay_obs_dim))
        s2 = torch.reshape(s2[:,1:,:,:], (-1, self.replay_obs_dim))
        self.target_state_buffer = torch.cat([s1, s2], dim=-1)

    def update_target_edges(self, edges, numb_repeats):
        self.target_edges_buffer = torch.clone(edges)
        self.target_edges_buffer = torch.repeat_interleave(self.target_edges_buffer, numb_repeats, dim=0)

    def get_target_batch_and_edges(self, batch_size):
        selected_indices = np.random.choice(self.target_state_buffer.shape[0], size=batch_size)
        batch = self.target_state_buffer[selected_indices]
        edges_for_batch = self.target_edges_buffer[selected_indices]
        return batch.to(self.device), edges_for_batch.to(self.device)



    def _do_reward_training(self, rel_rec, rel_send, prediction_steps, epoch):
        '''
            Train the discriminator
        '''
        self.disc_optimizer.zero_grad()

        sampled_batch = self.replay_buffer.sample_batch(self.disc_optim_batch_size)

        policy_disc_input = torch.cat([
            sampled_batch['obs'].reshape((-1, 1, self.num_atoms, self.inp_dims)), 
            sampled_batch['obs2'].reshape((-1, 1, self.num_atoms, self.inp_dims))], dim=-1)

        orig_edges = sampled_batch['act'].reshape((-1, self.num_atoms*(self.num_atoms-1), self.num_edges))
        orig_edges = orig_edges.to(self.device)
        
        expert_disc_input, expert_disc_edges = self.get_target_batch_and_edges(self.disc_optim_batch_size)
        expert_disc_input = expert_disc_input.reshape((-1, 1, self.num_atoms, 2*self.inp_dims)) # access to expert samples

        edges = torch.cat([orig_edges, expert_disc_edges], dim=0)
        

        policy_disc_input = policy_disc_input.transpose(1, 2)
        expert_disc_input = expert_disc_input.transpose(1, 2)

        disc_input = torch.cat([expert_disc_input, policy_disc_input], dim=0) # (2*B, 2*S)

        disc_logits, _ = self.disc_forward(disc_input, edges, rel_rec, rel_send, prediction_steps)
        disc_ce_loss = self.bce(disc_logits, self.bce_targets)

        if self.use_grad_pen: # gradient penalty
            eps = torch.rand((self.disc_optim_batch_size, 1)).to(self.device)
            
            interp_obs = eps*expert_disc_input + (1-eps)*policy_disc_input # interpolate
            interp_obs = interp_obs.detach()
            interp_obs.requires_grad_(True)
            gradients = autograd.grad(
                outputs=self.disc_forward(interp_obs, orig_edges, rel_rec, rel_send, prediction_steps)[0].sum(),
                inputs=[interp_obs],
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]       # gradients w.r.t. inputs (instead of parameters)
            
            # GP from Gulrajani et al. https://arxiv.org/pdf/1704.00028.pdf (WGAN-GP)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight

            # # GP from Mescheder et al. https://arxiv.org/pdf/1801.04406.pdf
            # gradient_penalty = (gradients.norm(2, dim=1) ** 2).mean()
            # disc_grad_pen_loss = gradient_penalty * 0.5 * self.grad_pen_weight
        else:
            disc_grad_pen_loss = 0.0

        disc_total_loss = disc_ce_loss + disc_grad_pen_loss
        disc_total_loss.backward(retain_graph=True)
        self.disc_optimizer.step()

        # return disc stat
        return np.array([disc_total_loss.item(), disc_ce_loss.item(), disc_total_loss.item() - disc_ce_loss.item()])


    def _do_policy_training(self, rel_rec, rel_send, prediction_steps, epoch):
        policy_batch = self.replay_buffer.sample_batch(self.policy_optim_batch_size)
        obs = policy_batch['obs'].reshape((-1, 1, self.num_atoms, self.inp_dims))
        obs2= policy_batch['obs2'].reshape((-1, 1, self.num_atoms, self.inp_dims))
        edges = policy_batch['act'].reshape((-1, self.num_atoms*(self.num_atoms-1), self.num_edges))

        obs = obs.transpose(1, 2)
        obs2 = obs2.transpose(1, 2)

        policy_batch['rew'] = self.get_reward(obs, obs2, edges, rel_rec, rel_send, prediction_steps)

        # policy optimization step
        agent_stat = self.agent.update(obs, edges, rel_rec, rel_send, prediction_steps, obs2, policy_batch['rew']) # loss-q, loss-pi, log-pi. 
        # in original f-MAX, it adds regularization on policy mean & logstd

        # used for logging
        reward_stat = np.array([policy_batch['rew'].mean().item(), policy_batch['rew'].std().item(), 
                                policy_batch['rew'].max().item(), policy_batch['rew'].min().item()])
        return np.concatenate((agent_stat, reward_stat)) # 7-dim


    def disc_forward(self, policy_disc_input, edges, rel_rec, rel_send, prediction_steps):
        # policy_disc_input: [B, S], where S >= len(self.state_indices)
        # NOTE: sampled from replay buffer is mixture of old policies, but empirically works well. called off-policy training as DAC
        # if policy_disc_input.shape[1] > len(self.state_indices):
        #     disc_input = torch.index_select(policy_disc_input, 1, self.state_indices)
        # else:
        #     disc_input = policy_disc_input
        disc_logits = self.discriminator(policy_disc_input, edges, rel_rec, rel_send, prediction_steps) # (B, 1)
        disc_preds = (disc_logits > 0).type(disc_logits.data.type())

        return disc_logits, disc_preds

    def get_reward(self, obs, obs2, edges, rel_rec, rel_send, prediction_steps):
        cat_obs = torch.cat([obs,obs2], dim=-1)
        self.discriminator.eval()
        with torch.no_grad():
            disc_logits, _ = self.disc_forward(cat_obs, edges, rel_rec, rel_send, prediction_steps) # D' = log(D) - log(1-D)
            disc_logits = disc_logits.view(-1) # must squeeze
        self.discriminator.train()

        # NOTE: important: compute the reward using the algorithm
        rewards = F.softplus(disc_logits, beta=-1) # -log(1-D) = -log(1+e^-D') ignore constant log2

        if self.rew_clip_max is not None:
            rewards = torch.clamp(rewards, max=self.rew_clip_max)
        if self.rew_clip_min is not None:
            rewards = torch.clamp(rewards, min=self.rew_clip_min)

        rewards *= self.reward_scale
        return rewards

    @property
    def networks(self):
        return [self.discriminator] + self.agent.networks

    def training_mode(self, mode): # mainly for batch1D in Discrim
        for net in self.networks:
            net.train(mode)

    
