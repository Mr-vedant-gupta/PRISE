#!/bin/bash

# Set experiment name prefix
EXP_PREFIX="proper_run2"
DOWNSTREAM_EXP_NAME="prise_finetune_untrained"

python train_prise.py exp_name=${EXP_PREFIX} replay_buffer_num_workers=4 batch_size=64 stage=3 downstream_exp_name=${DOWNSTREAM_EXP_NAME} downstream_task_name=80 downstream_task_suite=libero_90 num_train_steps=6000 eval_freq=500 max_traj_per_task=45
