import os
import numpy as np
import torch
import torch.optim as optim
from torch.multiprocessing import Pool
from tqdm import tqdm
from sklearn import metrics
from matplotlib import pyplot as plt
from itertools import repeat
from scipy.interpolate import Rbf
import scipy.stats as st

from worker import Worker
from utils import torch_load_cpu, get_inner_model, env_wrapper

#Memory用于绘图
class Memory:
    def __init__(self):
        self.steps = {}
        self.eval_values = {}
        self.training_values = {}


def euclidean_dist(x, y):  # 欧式距离
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]

    x 是一个 PyTorch 变量，其形状为 [m, d]，表示包含 m 个点，每个点有 d 个维度。
y 是一个 PyTorch 变量，其形状为 [n, d]，表示包含 n 个点，每个点同样有 d 个维度。

dist[i, j] 就表示 x 中第 i 个点到 y 中第 j 个点之间的欧式距离。
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)  # 计算x每行元素的平方和得到[m,1],再在第二维度复制得到[m,n]
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).T  # 计算y每行元素的平方和得到[n,1],再在第二维复制得到[n,m]，再转置得到[m,n]
    dist = xx + yy
    dist.addmm_(1, -2, x, y.T)
    dist[dist < 0] = 0
    dist = dist.sqrt()
    return dist


def worker_run(worker, param, opts, Batch_size, seed):
    '''

1分发参数：函数的第一个步骤是将当前的参数 param 从主进程（master）
传递给工作进程（worker）。这意味着主进程中可能有一组模型参数，
而这些参数需要被传递给各个工作进程，以便它们可以使用这些参数来执行训练。

2设置随机种子：函数调用了 worker.env.seed(seed)，
这是为了设置工作进程中的环境（worker.env）的随机数种子。
通过设置相同的种子，可以确保每个工作进程的随机性保持一致，从而提高实验的可重复性。

3训练一个周期：函数调用了 worker.train_one_epoch，
这是用于在工作进程中训练模型的函数。根据函数的命名，它可能会执行一个周期的训练。
这个函数会返回梯度和其他信息。

4返回结果：最后，工作进程将训练的结果（梯度和其他信息）通过 return out 返回给主进程。
这些结果可能包括训练得到的梯度，以及其他在训练过程中收集的信息。

在多进程/多智能体的并行训练中，主进程通常会分发模型参数给各个工作进程，然后每个工作进程使用这些参数在环境中运行一段时间，收集经验并计算梯度，最后将梯度传递给主进程，主进程用这些梯度来更新模型参数。这种并行训练的方式可以提高训练速度。

    '''
    # distribute current parameters
    worker.load_param_from_master(param)
    worker.env.reset(seed=seed)
    '''
    do_sample_for_training默认为true
    '''
    # get returned gradients and info from all agents
    '''
    一个工作节点根据n条采样的轨迹训练一次
    '''
    # do_sample_for_training决定采取动作时是采取贪婪策略还是试探策略
    '''
        if fixed_action is not None:
            action = torch.tensor(fixed_action, device = obs.device)
        elif do_sample_for_training:
            try:
                action = policy.sample()
            except:
                print(logits,obs)
        else:
            action = policy.probs.argmax()
        return action.item(), policy.log_prob(action)
    '''
    out = worker.train_one_epoch(Batch_size, opts.device, opts.do_sample_for_training)
    print('10个节点分别的平均回报为：', out[2])
    # store all values
    # out里包含grad, batch_loss.item(), np.mean(batch_rets), np.mean(batch_lens)

    return out


class Agent:

    def __init__(self, opts):

        # figure out the options
        self.opts = opts
        # setup arrays for distrubuted RL
        self.world_size = opts.num_worker

        # figure out the master
        self.master = Worker(
            id=0,
            is_Byzantine=False,
            env_name=opts.env_name,
            gamma=opts.gamma,
            hidden_units=opts.hidden_units,
            activation=opts.activation,
            output_activation=opts.output_activation,
            max_epi_len=opts.max_epi_len,
            opts=opts
        ).to(opts.device)
        '''
        opts.hidden_units = '16,16'
        opts.activation = 'ReLU'
        opts.output_activation = 'Tanh'
        '''
        # old master用于重要性采样
        # figure out a copy of the master node for importance sampling purpose
        self.old_master = Worker(
            id=-1,
            is_Byzantine=False,
            env_name=opts.env_name,
            gamma=opts.gamma,
            hidden_units=opts.hidden_units,
            activation=opts.activation,
            output_activation=opts.output_activation,
            max_epi_len=opts.max_epi_len,
            opts=opts
        ).to(opts.device)

        # figure out all the actors
        self.workers = []
        self.true_Byzantine = []
        '''
        world_size为结点总个数
        '''
        for i in range(self.world_size):
            # true_Byzantine中有num_Byzantine个true和world_size-num_Byzantine个false
            # 前面是拜占庭节点，后面是正常节点
            self.true_Byzantine.append(True if i < opts.num_Byzantine else False)
            self.workers.append(Worker(
                id=i + 1,
                is_Byzantine=True if i < opts.num_Byzantine else False,
                env_name=opts.env_name,
                gamma=opts.gamma,
                hidden_units=opts.hidden_units,
                activation=opts.activation,
                output_activation=opts.output_activation,
                attack_type=opts.attack_type,
                max_epi_len=opts.max_epi_len,
                opts=opts
            ).to(opts.device))
        print(
            f'{opts.num_worker} workers initilized with {opts.num_Byzantine if opts.num_Byzantine > 0 else "None"} of them are Byzantine.')

        if not opts.eval_only:
            # figure out the optimizer
            # 只有master有优化器
            self.optimizer = optim.Adam(self.master.logits_net.parameters(), lr=opts.lr_model)
        '''
        Pool 类是 Python 的 multiprocessing 模块提供的一个工具，用于实现进程池（process pool）。
        进程池是一种并行计算的方式，它通过将任务分配给多个进程来加速程序的执行。
        每个进程都运行在独立的 Python 解释器中，
        可以并行地执行任务。在深度学习中，进程池常用于并行地处理大规模数据集、执行多个模型的训练或评估等任务。
        '''
        self.pool = Pool(self.world_size)
        self.memory = Memory()

    '''
    加载训练好的参数，默认为none
    '''

    def load(self, load_path):
        assert load_path is not None
        load_data = torch_load_cpu(load_path)
        # load data for actor
        model_actor = get_inner_model(self.master.logits_net)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('master', {})})

        if not self.opts.eval_only:
            # load data for optimizer
            self.optimizer.load_state_dict(load_data['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.opts.device)
            # load data for torch and cuda
            torch.set_rng_state(load_data['rng_state'])
            if self.opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])

        print(' [*] Loading data from {}'.format(load_path))

    '''保存模型参数'''

    def save(self, epoch, run_id):
        print('Saving model and state...')
        torch.save(
            {
                'master': get_inner_model(self.master.logits_net).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            # 默认是不保存的，可以到options里修改parser.add_argument('--no_saving', action='store_true', help='Disable saving checkpoints')
            os.path.join(self.opts.save_dir, 'r{}-epoch-{}.pt'.format(run_id, epoch))
        )

    def eval(self):
        # turn model to eval mode
        self.master.eval()

    def train(self):
        # turn model to trainig mode
        self.master.train()

    '''
    tb_logger默认不需要日志
    
    '''

    def start_training(self, tb_logger=None, run_id=None):
        # 这个opts有好多参、
        # parameters of running
        opts = self.opts

        # for storing number of trajectories sampled

        step = 0
        epoch = 0
        ratios_step = 0

        # Start the training loop
        # 每一次while所有节点根据采样到的batch——size个轨迹训练一次
        # epoch为迭代次数，一次迭代指的是10个节点一起训练，训练完成之后将梯度上传至中心节点，中心节点更新参数
        # max_trajectories也是迭代次数，默认为100
        while step <= opts.max_trajectories:
            # epoch for storing checkpoints of model
            epoch += 1

            # Turn model into training mode
            print('\n\n')
            print("|", format(f" Training step {step} run_id {run_id} in {opts.seeds}", "*^60"), "|")
            self.train()

            # setup lr_scheduler
            print("Training with lr={:.3e} for run {}".format(self.optimizer.param_groups[0]['lr'], opts.run_name),
                  flush=True)

            # some emplty list for training and logging purpose
            gradient = []
            batch_loss = []
            batch_rets = []
            batch_lens = []

            '''state_dict()是一个字典存储了模型的各项参数
            param是master的参数
            '''
            # distribute current params and Batch_Size to all workers

            param = get_inner_model(self.master.logits_net).state_dict()

            #随机的batch_size
            if opts.FedPG_BR:
                Batch_size = np.random.randint(opts.Bmin, opts.Bmax + 1)
            else:
                Batch_size = opts.B
            '''
            world_size为节点总个数
            np.random.randint(1, 100000, self.world_size) 生成了一个包含 self.world_size 个元素的数组，
            这些元素都是从1到100000之间的随机整数。
            .tolist() 将生成的 NumPy 数组转换为 Python 列表。
            因此，seeds 列表包含了用于初始化多个随机数生成器的种子值。在并行化或分布式环境中，
            为了确保各个任务之间的独立性，常常会使用不同的随机种子。
            '''
            seeds = np.random.randint(1, 100000, self.world_size).tolist()
            '''self.workers是一个数组，里面有拜占庭和正常worker'''
            args = zip(self.workers,
                       repeat(param),
                       repeat(opts),
                       repeat(Batch_size),
                       seeds)
            # starmap 函数是 multiprocessing.Pool 类的一个方法，它接受一个函数和一个可迭代对象，
            # 以及其他可选参数。它会将可迭代对象的每个元素作为参数传递给函数，并返回结果的列表。worker_run是一个函数
            results = self.pool.starmap(worker_run, args)

            #  collect the gradient(for training), loss(for logging only), returns(for logging only), and epi_length(for logging only) from workers         
            # 显示各个节点求梯度的进度
            for out in tqdm(results, desc='Worker node'):
                # grad是一个数组，loss是一个数，rets是一个数（平均回报），lens也是一个数（平均长度）
                grad, loss, rets, lens = out

                # store all values
                gradient.append(grad)

                batch_loss.append(loss)
                batch_rets.append(rets)
                batch_lens.append(lens)

            # simulate FedScsPG-attack (if needed) on server for demo
            if opts.attack_type == 'FedScsPG-attack' and opts.num_Byzantine > 0:
                for idx, _ in enumerate(self.master.parameters()):
                    tmp = []
                    # 这个循环提取各个拜占庭节点第n层神经网络的梯度，加到tmp数组中
                    for bad_worker in range(opts.num_Byzantine):
                        tmp.append(gradient[bad_worker][idx].view(-1))
                    tmp = torch.stack(tmp)
                    # 计算各个拜占庭节点第n层梯度的欧氏距离
                    # estimated_2sigma是各拜占庭节点第n层神经网络梯度的最大欧氏距离
                    estimated_2sigma = euclidean_dist(tmp, tmp).max()
                    # estimated_mean是各拜占庭节点第n层神经网络平均梯度
                    estimated_mean = tmp.mean(0)

                    # change the gradient to be estimated_mean + 3sigma (with a random direction rnd)
                    # gradient[0][idx]表示第0个节点的n层神经网络，rnd是和第0个节点的n层神经网络相同形状的随机张量，
                    # rnd / rnd.norm() 对 rnd 进行标准化，即将 rnd 的每个元素除以 rnd 的范数（模）。
                    rnd = torch.rand(gradient[0][idx].shape) * 2. - 1.
                    rnd = rnd / rnd.norm()
                    # attacked_gradient为修改后的第n层神经网络的梯度信息
                    attacked_gradient = estimated_mean.view(
                        gradient[bad_worker][idx].shape) + rnd * estimated_2sigma * 3. / 2.
                    # 为各个拜占庭节点分配攻击后的第n层梯度信息，再gradient数组里修改
                    for bad_worker in range(opts.num_Byzantine):
                        gradient[bad_worker][idx] = attacked_gradient

            # make the old policy as a copy of the current master node
            self.old_master.load_param_from_master(param)

            # 聚合时检测拜占庭节点和正常节点，得到good_set
            if opts.FedPG_BR:

                #mu_vec为各个节点所有神经网络梯度信息(self.world_wize,-1)
                # flatten the gradient vectors of each worker and put them together, shape [num_worker, -1]
                mu_vec = None

                for idx, item in enumerate(self.old_master.parameters()):
                    # stack gradient[idx] from all worker nodes
                    grad_item = []
                    # 每次循环往grad_item = []添加各节点第n层神经网络的参数
                    for i in range(self.world_size):
                        grad_item.append(gradient[i][idx])
                    grad_item = torch.stack(grad_item).view(self.world_size, -1)

                    # concat stacked grad vector
                    if mu_vec is None:
                        mu_vec = grad_item.clone()
                    else:
                        mu_vec = torch.cat((mu_vec, grad_item.clone()), -1)
                # 计算所有节点梯度的欧式距离，dist是一个10*10的矩阵
                # calculate the norm distance between each worker's gradient vector, shape [num_worker, num_worker]
                dist = euclidean_dist(mu_vec, mu_vec)
                # delta和sigma为Filtering hyperparameters（过滤超参数）
                # calculate C, Variance Bound V, thresold, and alpha
                V = 2 * np.log(2 * opts.num_worker / opts.delta)
                sigma = opts.sigma

                threshold = 2 * sigma * np.sqrt(V / Batch_size)
                alpha = opts.alpha

                # to find MOM: |dist <= threshold| > 0.5 * num_worker
                mu_med_vec = None
                '''
                这行代码的目的是确定哪些节点的梯度被认为是正常（非拜占庭节点）。
                dist <= threshold：生成一个布尔矩阵，矩阵的元素是 dist 中对应位置的值是否小于等于 threshold。
                (dist <= threshold).sum(-1)：对布尔矩阵的每一行进行求和，得到一个包含每个节点对应的小于等于 threshold 的数量的张量。
                (0.5 * self.world_size)：计算非拜占庭节点的阈值，这里采用总节点数的一半。
                (dist <= threshold).sum(-1) > (0.5 * self.world_size)：将每个节点小于等于 threshold 的数量与阈值进行比较，得到一个布尔张量。
                如果某个节点小于等于 threshold 的数量超过一半，那么该节点被认为是正常的（True），否则被认为是异常的（False）。'''
                k_prime = (dist <= threshold).sum(-1) > (0.5 * self.world_size)
                #k_prime为[world_size,1],里面为true or false
                # computes the mom of the gradients, mu_med_vec, and
                # filter the gradients it believes to be Byzantine and store the index of non-Byzantine graidents in Good_set
                if torch.sum(k_prime) > 0:
                    #聚合那些认为是真正节点的梯度，计算他们的平均值
                    # view(1, -1) 的作用是将张量的形状改变为一个行向量（1行）
                    mu_mean_vec = torch.mean(mu_vec[k_prime], 0).view(1, -1)
                    # 计算所有正常节点的梯度与均值向量之间的欧氏距离。根据最接近均值的节点的索引，获取这个节点的梯度向量。这个向量 mu_med_vec 将用于后续的处理。
                    # mu_med_vec是一行向量
                    mu_med_vec = mu_vec[k_prime][euclidean_dist(mu_mean_vec, mu_vec[k_prime]).argmin()].view(1, -1)
                    # applying R1 to filter
                    # 将所有节点的梯度与最接近均值节点的梯度之间的欧氏距离与阈值进行比较。如果某个节点的梯度与最接近均值节点的梯度之间的距离小于等于 1 * threshold，
                    # 则将其标记为正常的（True），否则标记为异常的（False）。最终，Good_set 表示哪些节点的梯度被认为是正常的。
                    # goodset是一个列向量
                    Good_set = euclidean_dist(mu_vec, mu_med_vec) <= 1 * threshold
                else:
                    Good_set = k_prime  # skip this step if k_prime is empty (i.e., all False)

                # 如果判断出的正常节点个数太少
                # avoid the scenarios that Good_set is empty or can have |Gt| < (1 − α)K.
                '''
                This filtering rule is designed based on Assumption 2 which implies that the maximum distance between any 
                two good agents is 2σ, and our assumption that at least half of the agents are good
                我们在附录 D 中表明，在这两个假设下，R2 保证所有好的代理都包含在 Gt 中
                '''
                # alpha为拜占庭节点比例
                if torch.sum(Good_set) < (1 - alpha) * self.world_size or torch.sum(Good_set) == 0:
                    # 之前是k_prime = (dist <= threshold).sum(-1) > (0.5 * self.world_size)
                    # re-calculate mom of the gradients
                    k_prime = (dist <= 2 * sigma).sum(-1) > (0.5 * self.world_size)
                    if torch.sum(k_prime) > 0:
                        mu_mean_vec = torch.mean(mu_vec[k_prime], 0).view(1, -1)
                        mu_med_vec = mu_vec[k_prime][euclidean_dist(mu_mean_vec, mu_vec[k_prime]).argmin()].view(1, -1)
                        # re-filter with R2
                        # 之前是Good_set = euclidean_dist(mu_vec, mu_med_vec) <= 1 * threshold
                        Good_set = euclidean_dist(mu_vec, mu_med_vec) <= 2 * sigma
                    else:
                        Good_set = torch.zeros(self.world_size, 1).to(opts.device).bool()

            # else will treat all nodes as non-Byzantine nodes
            # 采用普通聚合方式
            else:
                Good_set = torch.ones(self.world_size, 1).to(opts.device).bool()

            # calculate number of good gradients for logging
            N_good = torch.sum(Good_set)

            #下面开始聚合好的节点梯度信息
            # aggregate all detected non-Byzantine gradients to get mu
            if N_good > 0:
                #mu里的每个元素记录了正常节点神经网络各层的聚合后的梯度
                mu = []
                for idx, item in enumerate(self.old_master.parameters()):
                    grad_item = []
                    for i in range(self.world_size):
                        if Good_set[i]:  # only aggregate non-Byzantine gradients
                            grad_item.append(gradient[i][idx])
                    mu.append(torch.stack(grad_item).mean(0))
            else:  # if still all nodes are detected to be Byzantine, check the sigma. If siagma is set properly, this situation will not happen.
                mu = None

            # perform gradient update in master node
            # 在中心节点更新参数
            grad_array = []  # store gradients for logging

            if opts.FedPG_BR or opts.SVRPG:
                # 采用FedPG_BR更新参数
                if opts.FedPG_BR:
                    # for n=1 to Nt ~ Geom(B/B+b) do grad update
                    b = opts.b
                    #在n次伯努利试验中，试验k次才得到第一次成功的次数，这个次数所符合的概率分布即为几何分布，geometric生成几何分布
                    N_t = np.random.geometric(p=1 - Batch_size / (Batch_size + b))

                elif opts.SVRPG:
                    b = opts.b
                    N_t = opts.N
                # old_master更新N_t次

                '''old_master负责更新，且参数不变，在下次10个节点各自训练完成之后到中心聚合时再将master的参数赋值给old——master，master负责采样'''

                for n in tqdm(range(N_t), desc='Master node'):

                    # calculate new gradient in master node
                    self.optimizer.zero_grad()
                    # do_sample_for_training决定采取动作时是采取贪婪策略还是试探策略
                    '''
                        if fixed_action is not None:
                            action = torch.tensor(fixed_action, device = obs.device)
                        elif do_sample_for_training:
                            try:
                                action = policy.sample()
                            except:
                                print(logits,obs)
                        else:
                            action = policy.probs.argmax()
                        return action.item(), policy.log_prob(action)
                    '''
                    # sample b trajectory using the latest policy (\theta_n) of master node
                    # 这里的batch_ret是一个数组
                    weights, new_logp, batch_ret, batch_len, batch_states, batch_actions = self.master.collect_experience_for_training(
                        b,
                        opts.device,
                        record=True,
                        sample=opts.do_sample_for_training)

                    # calculate gradient for the new policy (\theta_n)
                    loss_new = -(new_logp * weights).mean()
                    self.master.logits_net.zero_grad()
                    loss_new.backward()

                    if mu:
                        # get the old log_p with the old policy (\theta_0) but fixing the actions to be the same as the sampled trajectory
                        # old_logp = []记录中心节点采取的动作对数概率
                        old_logp = []
                        '''
                        batch_states[]记录了采样各轨迹所有的状态
                        batch_actions[]记录了各轨迹所有的动作
                        old_master是用的
                        '''
                        for idx, obs in enumerate(batch_states):
                            # act in the environment with the fixed action
                            obs = env_wrapper(opts.env_name, obs)
                            _, old_log_prob = self.old_master.logits_net(
                                torch.as_tensor(obs, dtype=torch.float32).to(opts.device),
                                fixed_action=batch_actions[idx])
                            # store in the old_logp
                            old_logp.append(old_log_prob)
                        old_logp = torch.stack(old_logp)

                        # Finding the ratio (pi_theta / pi_theta__old):
                        # print(old_logp, new_logp)
                        '''
                        在深度学习和机器学习中，通常使用的是自然对数，即以e为底的对数。自然对数通常用符号ln表示。在PyTorch中，torch.log 函数默认使用自然对数。
                        ratios反应old——master和master差异程度
                        '''
                        ratios = torch.exp(old_logp.detach() - new_logp.detach())
                        ratios_step += 1

                        # calculate gradient for the old policy (\theta_0)
                        loss_old = -(old_logp * weights * ratios).mean()
                        self.old_master.logits_net.zero_grad()
                        loss_old.backward()
                        grad_old = [item.grad for item in self.old_master.parameters()]

                        # early stop if ratio is not within [0.995, 1.005]
                        if torch.abs(ratios.mean()) < 0.995 or torch.abs(ratios.mean()) > 1.005:
                            N_t = n
                            break

                        if tb_logger is not None:
                            tb_logger.add_scalar(f'params/ratios_{run_id}', ratios.mean(), ratios_step)
                        # 更新master
                        # adjust and set the gradient for latest policy (\theta_n)
                        for idx, item in enumerate(self.master.parameters()):
                            item.grad = item.grad - grad_old[idx] + mu[idx]  # if mu is None, use grad from master 
                            grad_array += (item.grad.data.view(-1).cpu().tolist())

                    # take a gradient step
                    # 只有master有优化器
                    self.optimizer.step()

            else:  # GOMDP in this case

                b = 0
                N_t = 0

                # perform gradient descent with mu vector
                for idx, item in enumerate(self.master.parameters()):
                    item.grad = mu[idx]
                    grad_array += (item.grad.data.view(-1).cpu().tolist())

                # take a gradient step
                self.optimizer.step()

            print('\nepoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f \t N_good: %d' %
                  (epoch, np.mean(batch_loss), np.mean(batch_rets), np.mean(batch_lens), N_good))

            # current step: number of trajectories sampled
            # step += max(Batch_size, b * N_t) if self.world_size > 1 else Batch_size + b * N_t
            step += round((Batch_size * self.world_size + b * N_t) / (
                    1 + self.world_size)) if self.world_size > 1 else Batch_size + b * N_t

            # Logging to tensorboard
            if (tb_logger is not None):

                # training log
                tb_logger.add_scalar(f'train/total_rewards_{run_id}', np.mean(batch_rets), step)
                tb_logger.add_scalar(f'train/epi_length_{run_id}', np.mean(batch_lens), step)
                tb_logger.add_scalar(f'train/loss_{run_id}', np.mean(batch_loss), step)
                # grad log
                tb_logger.add_scalar(f'grad/grad_{run_id}', np.mean(grad_array), step)
                # optimizer log
                tb_logger.add_scalar(f'params/lr_{run_id}', self.optimizer.param_groups[0]['lr'], step)
                tb_logger.add_scalar(f'params/N_t_{run_id}', N_t, step)

                # Byzantine filtering log
                if opts.FedPG_BR:

                    y_true = self.true_Byzantine
                    y_pred = (~ Good_set).view(-1).cpu().tolist()

                    dist_Byzantine = dist[:opts.num_Byzantine][:, :opts.num_Byzantine]
                    dist_good = dist[opts.num_Byzantine:][:, opts.num_Byzantine:]
                    dist_good_Byzantine = dist[:opts.num_Byzantine][:, opts.num_Byzantine:]

                    if opts.num_Byzantine > 0:
                        tb_logger.add_scalar(f'grad_norm_mean/Byzantine_{run_id}', torch.mean(dist_Byzantine), step)
                        tb_logger.add_scalar(f'grad_norm_max/Byzantine_{run_id}', torch.max(dist_Byzantine), step)
                        tb_logger.add_scalar(f'grad_norm_mean/Good_{run_id}', torch.mean(dist_good), step)
                        tb_logger.add_scalar(f'grad_norm_max/Good_{run_id}', torch.max(dist_good), step)
                        tb_logger.add_scalar(f'grad_norm_mean/Between_{run_id}', torch.mean(dist_good_Byzantine), step)
                        tb_logger.add_scalar(f'grad_norm_max/Between_{run_id}', torch.max(dist_good_Byzantine), step)

                        tb_logger.add_scalar(f'Byzantine/precision_{run_id}', metrics.precision_score(y_true, y_pred),
                                             step)
                        tb_logger.add_scalar(f'Byzantine/recall_{run_id}', metrics.recall_score(y_true, y_pred), step)
                        tb_logger.add_scalar(f'Byzantine/f1_score_{run_id}', metrics.f1_score(y_true, y_pred), step)

                    tb_logger.add_scalar(f'Byzantine/threshold_{run_id}', threshold, step)
                    tb_logger.add_scalar(f'grad_norm_mean/ALL_{run_id}', torch.mean(dist), step)
                    tb_logger.add_scalar(f'grad_norm_max/ALL_{run_id}', torch.max(dist), step)
                    tb_logger.add_scalar(f'Byzantine/N_good_pred_{run_id}', N_good, step)

                # for performance plot
            if run_id not in self.memory.steps.keys():
                self.memory.steps[run_id] = []
                self.memory.eval_values[run_id] = []
                self.memory.training_values[run_id] = []

            self.memory.steps[run_id].append(step)
            self.memory.training_values[run_id].append(np.mean(batch_rets))

            # do validating render默认为true eval_reward为平均回报长度
            eval_reward = self.start_validating(tb_logger, step, max_steps=opts.val_max_steps, render=opts.render,
                                                run_id=run_id)

            self.memory.eval_values[run_id].append(eval_reward)

            # save current model
            if not opts.no_saving:
                self.save(epoch, run_id)

    # validate the new model   
    def start_validating(self, tb_logger=None, id=0, max_steps=1000, render=False, run_id=0, mode='human'):
        print('\nValidating...', flush=True)

        val_ret = 0.0
        val_len = 0.0
        # def rollout(self, device, max_steps = 1000, render = False, env = None,
        # obs = None, sample = True, mode = 'human', save_dir = './', filename = '.'):
        #验证val_size次
        for _ in range(self.opts.val_size):
            #epi_ret是测试时一条轨迹的回报， epi_len是一条轨迹的长度，默认测试时候是1000，如果越接近1000，说明保持时间越长
            epi_ret, epi_len, _ = self.master.rollout(self.opts.device, max_steps=max_steps, render=render,
                                                      sample=False, mode=mode, save_dir='./outputs/',
                                                      filename=f'gym_{run_id}_{_}.gif')
            val_ret += epi_ret
            val_len += epi_len

        val_ret /= self.opts.val_size
        val_len /= self.opts.val_size

        print('\nGradient step: %3d \t return: %.3f \t ep_len: %.3f' %
              (id, np.mean(val_ret), np.mean(val_len)))

        if (tb_logger is not None):
            tb_logger.add_scalar(f'validate/total_rewards_{run_id}', np.mean(val_ret), id)
            tb_logger.add_scalar(f'validate/epi_length_{run_id}', np.mean(val_len), id)
            tb_logger.close()

        return np.mean(val_ret)

    # logging performance summary
    '''
    from scipy.interpolate import Rbf
    import numpy as np
    x = [1,3,5,7]
    y = [1,9,25,49]
    data=Rbf(x,y,function='linear')([1,2,3,4])
    print(data)
    上面是一个线性插值的例子，首先根据x（横坐标）和y（纵坐标）拟合出一条x平方的曲线，然后输出横坐标为【1，2，3，4】位置的的纵坐标
    '''
    def plot_graph(self, array):
        #plt.ioff()
        axes = plt.axes()
        axes.xaxis.set_visible(False)
        axes.yaxis.set_visible(False)
        fig = plt.figure(figsize=(8, 4))
        y = []

        for id in self.memory.steps.keys():
            x = self.memory.steps[id]
            # y是所有run_id的每个step上的数据值，是一个二维数组，一行为一个run——id，每一列为一个step
            y.append(Rbf(x, array[id], function='linear')(np.arange(self.opts.max_trajectories)))

        mean = np.mean(y, axis=0)
        # scipy.stats.norm函数 可以实现正态分布（也就是高斯分布），均值落在l,h间的可能性为90%
        l, h = st.norm.interval(0.90, loc=np.mean(y, axis=0), scale=st.sem(y, axis=0))

        plt.plot(mean)
        plt.fill_between(range(int(self.opts.max_trajectories)), l, h, alpha=0.5)

        axes.set_ylim([self.opts.min_reward, self.opts.max_reward])
        plt.xlabel("Number of Trajectories")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def log_performance(self):
        self.plot_graph(self.memory.eval_values)
        self.plot_graph(self.memory.training_values)
        # eval_img = self.plot_graph(self.memory.eval_values)
        # training_img = self.plot_graph(self.memory.training_values)
        # tb_logger.add_figure(f'validate/performance_until_{len(self.memory.steps.keys())}_runs', eval_img,
        #                      len(self.memory.steps.keys()))
        # tb_logger.add_figure(f'train/performance_until_{len(self.memory.steps.keys())}_runs', training_img,
        #                      len(self.memory.steps.keys()))



