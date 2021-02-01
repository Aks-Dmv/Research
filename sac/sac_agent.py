# '''
# Code from spinningup repo.
# Refer[Original Code]: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
# '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from sac.modules import MLPDecoder

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def gnn(sizes):
    decoder = MLPDecoder(n_in_node=sizes[0],
                         edge_types=2, # hard coded
                         msg_hid=sizes[-1],
                         msg_out=sizes[-1],
                         n_hid=sizes[-1],
                         do_prob=0.0, # hard coded
                         skip_first=False) # hard coded
    return decoder


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = gnn( [obs_dim] + list(hidden_sizes) )
        self.mu_layer = nn.Linear(obs_dim, act_dim)
        self.log_std_layer = nn.Linear(obs_dim, act_dim)
        self.act_limit = act_limit

    def forward(self, data, edges, rel_rec, rel_send, prediction_steps, deterministic=False, with_logprob=True):
        net_out = self.net(data, edges, rel_rec, rel_send, prediction_steps)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(dim=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(dim=-1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi
        
    def log_prob_unclipped(self,data, edges, rel_rec, rel_send, prediction_steps,action):
        net_out = self.net(data, edges, rel_rec, rel_send, prediction_steps)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        
        pi_action = action
        logp_pi = pi_distribution.log_prob(pi_action).sum(dim=-1)
        return logp_pi

    def log_prob(self, data, edges, rel_rec, rel_send, prediction_steps, act):
        net_out = self.net(data, edges, rel_rec, rel_send, prediction_steps)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)

        # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
        # NOTE: The correction formula is a little bit magic. To get an understanding 
        # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
        # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
        # Try deriving it yourself as a (very difficult) exercise. :)
        logp_pi = pi_distribution.log_prob(act).sum(dim=-1)
        logp_pi -= (2*(np.log(2) - act - F.softplus(-2*act))).sum(dim=-1)

        return logp_pi

class SquashedGmmMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, k):
        super().__init__()
        print("gmm")
        self.net = gnn( [obs_dim] + list(hidden_sizes) )
        self.mu_layer = nn.Linear(obs_dim, k*act_dim)
        self.log_std_layer = nn.Linear(obs_dim, k*act_dim)
        self.act_limit = act_limit
        self.k = k 
        

    def forward(self, data, edges, rel_rec, rel_send, prediction_steps, deterministic=False, with_logprob=True):
        net_out = self.net(data, edges, rel_rec, rel_send, prediction_steps)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # n = batch size
        n, _ = mu.shape
        mixture_components = torch.from_numpy(np.random.randint(0, self.k, (n))) # NOTE: fixed equal weight

        # change shape to k x batch_size x act_dim
        mu = mu.view(n, self.k, -1).permute(1, 0, 2)
        std = std.view(n, self.k, -1).permute(1, 0, 2)

        mu_sampled = mu[mixture_components, torch.arange(0,n).long(), :]
        std_sampled = std[mixture_components, torch.arange(0,n).long(), :]

        if deterministic:
            pi_action = mu_sampled
        else:
            pi_action = Normal(mu_sampled, std_sampled).rsample() # (n, act_dim)

        if with_logprob:
            # logp_pi[i,j] contains probability of ith action under jth mixture component
            logp_pi = torch.zeros((n, self.k)).to(pi_action)

            for j in range(self.k):
                pi_distribution = Normal(mu[j,:,:], std[j,:,:]) # (n, act_dim)

                logp_pi_mixture = pi_distribution.log_prob(pi_action).sum(dim=-1)
                logp_pi_mixture -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(dim=-1)
                logp_pi[:,j] = logp_pi_mixture

            # logp_pi = (sum of p_pi over mixture components)/k
            logp_pi = torch.logsumexp(logp_pi, dim=-1) - torch.FloatTensor([np.log(self.k)]).to(logp_pi) # numerical stable
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes,  num_atoms):
        super().__init__()
        self.q1 = gnn( [obs_dim + act_dim] + list(hidden_sizes) )
        self.q2 = nn.Linear(obs_dim + act_dim, 1)
        self.q3 = nn.Linear(num_atoms, 1)

    def forward(self, data, edges, rel_rec, rel_send, prediction_steps, act):
        q = self.q1(torch.cat([data, act], dim=-1), edges, rel_rec, rel_send, prediction_steps)
        q = self.q2(q)
        q = torch.squeeze(q)
        q = self.q3(q)
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, num_atoms, action_limit, k, hidden_sizes=(256,256), add_time=False,
                 activation=nn.ReLU, device=torch.device("cpu")):
        super().__init__()

        obs_dim = observation_space
        act_dim = action_space
        act_limit = action_limit
        self.device = device
        # print("MLP actor critic device: ", device)

        # build policy and value functions
        # if add_time: # policy ignores the time index. only Q function uses the time index
        #     self.pi = SquashedGaussianMLPActor(obs_dim - 1, act_dim, hidden_sizes, activation, act_limit).to(self.device)
        # else:

        # old code: gaussian
        #self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(self.device)

        if k == 1:
            self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(self.device)
        else:
            self.pi = SquashedGmmMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, k).to(self.device)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, num_atoms).to(self.device)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, num_atoms).to(self.device)

    def act(self, data, edges, rel_rec, rel_send, prediction_steps, deterministic=False, get_logprob = False):
        with torch.no_grad():
            a, logpi = self.pi(data, edges, rel_rec, rel_send, prediction_steps, deterministic, True)
            if get_logprob:
                return a.cpu().data.numpy().flatten(), logpi.cpu().data.numpy()
            else:
                return a.cpu().data.numpy().flatten()

    def act_batch(self, data, edges, rel_rec, rel_send, prediction_steps, deterministic=False, get_logprob = False):
        with torch.no_grad():
            a, logpi = self.pi(data, edges, rel_rec, rel_send, prediction_steps, deterministic, True)
            if get_logprob:
                return a.cpu().data.numpy(), logpi.cpu().data.numpy()
            else:
                return a.cpu().data.numpy()

    def log_prob(self, data, edges, rel_rec, rel_send, prediction_steps, act):
        return self.pi.log_prob(data, edges, rel_rec, rel_send, prediction_steps, act)


class MLPDisc(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, num_atoms, use_bn=True, clamp_magnitude=10.0):
        super().__init__()
        self.clamp_magnitude = clamp_magnitude

        self.q1 = gnn( [ 2*obs_dim ] + list(hidden_sizes) )
        self.q2 = nn.Linear(2*obs_dim, 1)
        self.q3 = nn.Linear(num_atoms, 1)

    def forward(self, data, edges, rel_rec, rel_send, prediction_steps):
        q = self.q1(data, edges, rel_rec, rel_send, prediction_steps)
        q = self.q2(q)
        q = torch.squeeze(q)
        q = self.q3(q)
        q = torch.clamp(q, -1.0*self.clamp_magnitude, self.clamp_magnitude)
        return q