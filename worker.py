import torch
import numpy as np
import gym
from gym.spaces import Discrete
from policy import MlpPolicy, DiagonalGaussianMlpPolicy
from utils import get_inner_model, save_frames_as_gif
from utils import env_wrapper
import random

class Worker:

    def __init__(self,
                 id,
                 is_Byzantine,
                 env_name,
                 hidden_units,
                 gamma,
                 activation = 'Tanh',
                 output_activation = 'Identity',
                 attack_type = None,
                 max_epi_len = 0,
                 opts = None
                 ):
        super(Worker, self).__init__()
        
        # setup
        self.id = id
        self.is_Byzantine = is_Byzantine
        self.gamma = gamma
        # make environment, check spaces, get obs / act dims
        self.env_name = env_name
        #加render_mode="human"的位置
        self.env = gym.make(env_name)
        self.attack_type = attack_type
        self.max_epi_len = max_epi_len
        
        assert opts is not None
        
        # get observation dim
        obs_dim = self.env.observation_space.shape[0]
        '''
        isinstance 是 Python 中的内置函数，用于检查一个对象是否是指定类型或类型元组中的一种类型。
        例如下面检查动作是否连续
        '''
        if isinstance(self.env.action_space, Discrete):
            n_acts = self.env.action_space.n
        else:
            n_acts = self.env.action_space.shape[0]
        
        hidden_sizes = list(eval(hidden_units))
        self.sizes = [obs_dim]+hidden_sizes+[n_acts] # make core of policy network
        
        # get policy net
        if isinstance(self.env.action_space, Discrete):
            self.logits_net = MlpPolicy(self.sizes, activation, output_activation)
        else:
            self.logits_net = DiagonalGaussianMlpPolicy(self.sizes, activation,)
        
        if self.id == 1:
            print(self.logits_net)

    
    def load_param_from_master(self, param):
        model_actor = get_inner_model(self.logits_net)
        model_actor.load_state_dict({**model_actor.state_dict(), **param})

    def rollout(self, device, max_steps = 1000, render = True, env = None, obs = None, sample = True, mode = 'human', save_dir = './', filename = '.'):
        
        if env is None and obs is None:
            env = gym.make(self.env_name,render_mode='human')
            obs = env.reset()[0]
            
        done = False  
        ep_rew = []
        frames = []
        step = 0
        while not done and step < max_steps:
            step += 1
            # if render:
            #     if mode == 'rgb':
            #         frames.append(env.render(mode="rgb_array"))
            #     else:
            #         env.render()
                
            obs = env_wrapper(env.unwrapped.spec.id, obs)
            action = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = sample)[0]
            obs, rew, done, _ ,_= env.step(action)
            ep_rew.append(rew)

        if mode == 'rgb': save_frames_as_gif(frames, save_dir, filename)
        return np.sum(ep_rew), len(ep_rew), ep_rew
    '''
    B为batch-size，B为采样轨迹的个数，batch_log_prob为所有轨迹的动作对数概率
    collect_experience_for_training负责采样B条轨迹
    '''
    def collect_experience_for_training(self, B, device, record = False, sample = True, attack_type = None):
        # make some empty lists for logging.
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths
        batch_log_prob = []     # for gradient computing

        # reset episode-specific variables
        obs = self.env.reset()[0]  # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep
        
        # make two lists for recording the trajectory
        if record:
            batch_states = []
            batch_actions = []

        t = 1
        # collect experience by acting in the environment with current policy
        while True:
            # save trajectory
            if record:
                batch_states.append(obs)
            # act in the environment  
            obs = env_wrapper(self.env_name, obs)
            
            # simulate random-action attacker if needed
            if self.is_Byzantine and attack_type is not None and self.attack_type == 'random-action':
                act_rnd = self.env.action_space.sample()
                if isinstance(act_rnd, int): # discrete action space
                    act_rnd = 0
                else: # continuous
                    act_rnd = np.zeros(len(self.env.action_space.sample()), dtype=np.float32)
                    '''
                    自动调用forward
                    输出动作和对数概率值
                    '''
                act, log_prob= self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = sample, fixed_action = act_rnd)
            else:
                act, log_prob= self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = sample)
           
            obs, rew, done, info, _ = self.env.step(act)
            
            # simulate reward-flipping attacker if needed
            if self.is_Byzantine and attack_type is not None and self.attack_type == 'reward-flipping': 
                rew = - rew
                
            # timestep
            t = t + 1
            
            # save action_log_prob, reward
            batch_log_prob.append(log_prob)
            
            ep_rews.append(rew)
            
            # save trajectory
            if record:
                batch_actions.append(act)

            if done or len(ep_rews) >= self.max_epi_len:
                
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                
                # the weight for each logprob(a_t|s_T) is sum_t^T (gamma^(t'-t) * r_t')
                '''
                returns反映了一个轨迹中每个状态的价值函数
                '''
                returns = []
                R = 0
                # simulate random-reware attacker if needed
                '''
                拜占庭智能体的攻击方式之一：随意修改轨迹的回报值
                '''
                if self.is_Byzantine and attack_type is not None and self.attack_type == 'random-reward': 
                    random.shuffle(ep_rews)
                    for r in ep_rews:
                        R = r + self.gamma * R
                        returns.insert(0, R)
                else:
                    for r in ep_rews[::-1]:
                        R = r + self.gamma * R
                        '''
                        returns.insert(0, R)：将计算得到的回报值 R 插入到 returns 列表的开头。
                        由于 insert(0, R) 操作，新的回报值将成为列表的第一个元素，而之前的元素依次后移。'''
                        returns.insert(0, R)            
                returns = torch.tensor(returns, dtype=torch.float32)
                
                # return whitening
                advantage = (returns - returns.mean()) / (returns.std() + 1e-20)
                batch_weights += advantage

                # end experience loop if we have enough of it
                if len(batch_lens) >= B:
                    break
                
                # reset episode-specific variables
                obs, done, ep_rews, t = self.env.reset()[0], False, [], 1


        # make torch tensor and restrict to batch_size
        weights = torch.as_tensor(batch_weights, dtype = torch.float32).to(device)
        logp = torch.stack(batch_log_prob)

        if record:
            return weights, logp, batch_rets, batch_lens, batch_states, batch_actions
        else:
            return weights, logp, batch_rets, batch_lens
    
    '''
    B为batch-size，为采样轨迹的个数，sample为true
    '''
    def train_one_epoch(self, B, device, sample):
        
        # collect experience by acting in the environment with current policy
        weights, logp, batch_rets, batch_lens = self.collect_experience_for_training(B, device, sample = sample, attack_type = self.attack_type)
        
        # calculate policy gradient loss
        batch_loss = -(logp * weights).mean()
    
        # take a single policy gradient update step
        self.logits_net.zero_grad()
        batch_loss.backward()
        
        # determine if the agent is byzantine
        if self.is_Byzantine and self.attack_type is not None:
            # 若是拜占庭节点则返回带有噪声的错误的梯度信息
            grad = []
            for item in self.parameters():
                if self.attack_type == 'zero-gradient':
                    grad.append(item.grad * 0)
                
                elif self.attack_type == 'random-noise':
                    rnd = (torch.rand(item.grad.shape, device = item.device) * 2 - 1) * (item.grad.max().data - item.grad.min().data) * 3
                    grad.append(item.grad + rnd)
                
                elif self.attack_type == 'sign-flipping':
                    grad.append(-2.5 * item.grad)
                    
                elif self.attack_type == 'reward-flipping':
                    grad.append(item.grad)
                    # refer to collect_experience_for_training() to see attack

                elif self.attack_type == 'random-action':
                    grad.append(item.grad)
                    # refer to collect_experience_for_training() to see attack
                
                elif self.attack_type == 'random-reward':
                    grad.append(item.grad)
                    # refer to collect_experience_for_training() to see attack
                
                elif self.attack_type == 'FedScsPG-attack':
                    grad.append(item.grad)
                    # refer to agent.py to see attack
                    
                else: raise NotImplementedError()

    
        else:
            # 如果是有效节点则返回真实梯度
            grad = [item.grad for item in self.parameters()]
        
        # report the results to the agent for training purpose
        # ep_ret = sum(ep_rews)一个轨迹的回报、batch_rets是一个列表记录了所有轨迹的回报，batch_rets.append(ep_ret)
        # 一个节点梯度的结构如下
        # [tensor([[4.1594e-05, 1.9726e-06, 3.7926e-06, 1.7533e-04],
        #          [-2.0125e-05, -6.5574e-04, 8.5985e-05, 1.1184e-03],
        #          [5.1331e-07, 5.3330e-04, -2.7224e-05, -8.6897e-04],
        #          [-2.2477e-05, -2.2906e-04, 5.7204e-06, 3.8916e-04],
        #          [1.5800e-05, 1.3683e-04, 7.6842e-06, -2.1993e-04],
        #          [-6.4858e-05, -2.3191e-04, -2.2259e-05, 2.1133e-04],
        #          [-1.1813e-04, -1.2153e-05, -3.0623e-05, -2.8716e-04],
        #          [-1.0669e-05, -1.0380e-05, 1.0554e-05, -3.2861e-05],
        #          [-2.5769e-06, 5.7941e-05, -7.4148e-06, -9.2961e-05],
        #          [-3.4233e-05, -2.9185e-04, -1.3769e-05, 4.6688e-04],
        #          [-1.3888e-05, -1.9152e-04, 2.0695e-05, 2.9869e-04],
        #          [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        #          [7.9208e-06, 8.6482e-05, -3.6289e-05, -1.4376e-04],
        #          [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        #          [1.6663e-05, 7.1851e-05, 8.4122e-06, -8.7895e-05],
        #          [-2.7073e-05, -3.1136e-04, 6.8445e-05, 5.5951e-04]]),
        #  tensor([-3.6762e-04, -5.9204e-05, 8.4646e-04, -1.0421e-03, 7.0395e-04,
        #          -1.1332e-03, -1.5209e-03, 4.3484e-04, 1.5258e-04, -1.5444e-03,
        #          1.6648e-04, 0.0000e+00, -1.6070e-04, 0.0000e+00, 4.2604e-04,
        #          -6.0727e-05]), tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00],
        #                                 [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00],
        #                                 [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00],
        #                                 [-4.7553e-04, 2.4279e-04, 1.2758e-04, 9.3654e-05, 6.1213e-05,
        #                                  -4.8027e-04, -2.8400e-04, -1.4653e-04, 2.5398e-05, 2.0239e-04,
        #                                  -2.6724e-04, 0.0000e+00, 1.8259e-04, 0.0000e+00, -6.4824e-04,
        #                                  4.0893e-04],
        #                                 [4.9815e-04, -5.0882e-04, -2.8068e-04, -2.0293e-04, -1.2714e-04,
        #                                  5.9299e-04, -4.5709e-05, -7.1899e-05, -5.2578e-05, -4.2162e-04,
        #                                  -8.5598e-04, 0.0000e+00, -3.8179e-04, 0.0000e+00, 3.9845e-04,
        #                                  -8.5528e-04],
        #                                 [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00],
        #                                 [1.4933e-04, -1.5839e-04, -9.0611e-05, -6.4790e-05, -3.9297e-05,
        #                                  1.7851e-04, -1.5056e-05, -2.4274e-05, -1.6209e-05, -1.3063e-04,
        #                                  -2.6411e-04, 0.0000e+00, -1.1863e-04, 0.0000e+00, 1.2071e-04,
        #                                  -2.6582e-04],
        #                                 [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00],
        #                                 [-8.7699e-04, -8.6706e-04, -1.4376e-03, -8.2140e-04, -1.5629e-04,
        #                                  -9.0750e-04, -1.5581e-05, -4.7165e-04, -6.4693e-05, -5.1787e-04,
        #                                  0.0000e+00, 0.0000e+00, -5.7160e-04, 0.0000e+00, -6.1170e-05,
        #                                  -1.3078e-03],
        #                                 [-5.5908e-04, 5.6187e-04, 3.0487e-04, 2.2155e-04, 1.4083e-04,
        #                                  -6.6436e-04, 4.9178e-05, 7.6427e-05, 5.8307e-05, 4.6655e-04,
        #                                  9.4891e-04, 0.0000e+00, 4.2193e-04, 0.0000e+00, -4.4520e-04,
        #                                  9.4511e-04],
        #                                 [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00],
        #                                 [1.1833e-04, -1.0822e-04, -5.2708e-05, -3.9664e-05, -2.7643e-05,
        #                                  1.3926e-04, -7.9363e-06, -1.1204e-05, -1.1523e-05, -9.1002e-05,
        #                                  -1.8713e-04, 0.0000e+00, -8.1661e-05, 0.0000e+00, 9.1914e-05,
        #                                  -1.8281e-04],
        #                                 [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00],
        #                                 [-3.1653e-04, 3.1063e-04, 1.6435e-04, 1.2039e-04, 7.8221e-05,
        #                                  -3.7519e-04, 2.6116e-05, 3.9797e-05, 3.2440e-05, 2.5873e-04,
        #                                  5.2766e-04, 0.0000e+00, 2.3354e-04, 0.0000e+00, -2.5044e-04,
        #                                  5.2305e-04],
        #                                 [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00],
        #                                 [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                  0.0000e+00]]),
        #  tensor([0.0000, 0.0000, 0.0000, -0.0016, 0.0011, 0.0000, 0.0003, 0.0000,
        #          -0.0044, -0.0012, 0.0000, 0.0003, 0.0000, -0.0007, 0.0000, 0.0000]),
        #  tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00, 7.9615e-04, 1.1739e-03,
        #           0.0000e+00, 6.2351e-04, 0.0000e+00, -5.7802e-04, 1.8905e-03,
        #           0.0000e+00, -1.4598e-04, 0.0000e+00, 1.2379e-04, 0.0000e+00,
        #           0.0000e+00],
        #          [0.0000e+00, 0.0000e+00, 0.0000e+00, -6.4889e-04, -1.0092e-03,
        #           0.0000e+00, -5.3214e-04, 0.0000e+00, 4.5968e-04, -1.5826e-03,
        #           0.0000e+00, 9.4370e-05, 0.0000e+00, -1.6548e-04, 0.0000e+00,
        #           0.0000e+00]]), tensor([0.0037, -0.0032])]
        return grad, batch_loss.item(), np.mean(batch_rets), np.mean(batch_lens)


    def to(self, device):
        self.logits_net.to(device)
        return self
    
    def eval(self):
        self.logits_net.eval()
        return self
        
    def train(self):
        self.logits_net.train()
        return self
    
    def parameters(self):
        return self.logits_net.parameters()
