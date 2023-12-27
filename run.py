#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import pprint
import numpy as np
'''
torch.utils.tensorboard.SummaryWriter 是 PyTorch 提供的用于创建 TensorBoard 日志的工具。
TensorBoard 是 TensorFlow 提供的一个可视化工具，用于查看和分析深度学习模型的训练过程和性能。

SummaryWriter 的作用是将训练过程中的各种信息记录到 TensorBoard 日志中，
以便用户可以通过 TensorBoard 界面进行可视化分析。
这包括训练损失、准确率、模型参数的直方图、学习率曲线等信息。'''
#from torch.utils.tensorboard import SummaryWriter

from agent import Agent
from worker import Worker
from options import get_options
from utils import get_inner_model

def run(opts):

    # Pretty print the run args
    pprint.pprint(vars(opts))
    
    # setup tensorboard
    # if not opts.no_tb:
    #     tb_writer = SummaryWriter(opts.log_dir)
    # else:
    tb_writer = None

    # Optionally configure tensorboard
    if not opts.no_saving and not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)

    # Configure for multiple runs    
    assert opts.multiple_run > 0
    #opts.seeds为从【0，opts.multiple_run-1】的数组,multiple_run默认为1，opts.max_trajectories = 100
    opts.seeds = (np.arange(opts.multiple_run) + opts.seed ).tolist()

    # Save arguments so exact configuration can always be found
    if not opts.no_saving:
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    
    # Figure out the RL
    agent = Agent(opts)
    
    # Do validation only
    if opts.eval_only:
        # Set the random seed
        torch.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        
        # Load data from load_path
        if opts.load_path is not None:
            agent.load(opts.load_path)
        
        agent.start_validating(tb_writer, 0, opts.val_max_steps, opts.render, mode = opts.mode)
        
    else:
        for run_id in opts.seeds:
            # Set the random seed
            torch.manual_seed(run_id)
            np.random.seed(run_id)
            
            nn_parms_worker = Worker(
                id = 0,
                is_Byzantine = False,
                env_name = opts.env_name,
                gamma = opts.gamma,
                hidden_units = opts.hidden_units, 
                activation = opts.activation, 
                output_activation = opts.output_activation,
                max_epi_len = opts.max_epi_len,
                opts = opts
            ).to(opts.device)
            
            # Load data from random policy
            model_actor = get_inner_model(agent.master.logits_net)
            model_actor.load_state_dict({**model_actor.state_dict(), **get_inner_model(nn_parms_worker.logits_net).state_dict()})
        
            # Start training here
            agent.start_training(tb_writer, run_id)
            agent.log_performance()
            if tb_writer:
                agent.log_performance(tb_writer)
            


if __name__ == "__main__":
    
    import warnings
    warnings.filterwarnings("ignore", category=Warning)
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    run(get_options())
