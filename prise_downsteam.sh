#!/bin/bash

# Set experiment name prefix
EXP_PREFIX="smaller_network"
DOWNSTREAM_EXP_NAME="smaller_network_finetune_1"
set seed
python train_prise.py exp_name=prise95_ep20 replay_buffer_num_workers=2 batch_size=0 stage=3 downstream_exp_name=prise95_ep20 downstream_task_name=-1 downstream_task_suite=libero_90 num_train_steps=483 eval_freq=23 max_traj_per_task=45 seed=95 checkpoint_name=snapshot_ckpt24000
