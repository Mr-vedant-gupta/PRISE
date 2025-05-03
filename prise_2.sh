#EXP_NAME="smaller_network"
#
#python train_prise.py exp_name=${EXP_NAME} stage=2 hidden_dim=64


#python train_prise.py exp_name=prise95_ep5 stage=2 checkpoint_name=snapshot_ckpt6000
#python train_prise.py exp_name=prise95_ep10 stage=2 checkpoint_name=snapshot_ckpt12000
#python train_prise.py exp_name=prise95_ep15 stage=2 checkpoint_name=snapshot_ckpt18000
#python train_prise.py exp_name=prise95_ep20 stage=2 checkpoint_name=snapshot_ckpt24000
#
#python train_prise.py exp_name=prise96_ep5 stage=2 checkpoint_name=snapshot_ckpt6000
#python train_prise.py exp_name=prise96_ep10 stage=2 checkpoint_name=snapshot_ckpt12000
#python train_prise.py exp_name=prise96_ep15 stage=2 checkpoint_name=snapshot_ckpt18000
#python train_prise.py exp_name=prise96_ep20 stage=2 checkpoint_name=snapshot_ckpt24000

#python train_prise.py exp_name=prise97_ep5 stage=2 checkpoint_name=snapshot_ckpt6000
#python train_prise.py exp_name=prise97_ep10 stage=2 checkpoint_name=snapshot_ckpt12000
#python train_prise.py exp_name=prise97_ep15 stage=2 checkpoint_name=snapshot_ckpt18000
python train_prise.py exp_name=prise97_ep20 stage=2 checkpoint_name=snapshot_ckpt24000

#python train_prise.py exp_name=prise97_ep10 replay_buffer_num_workers=2 batch_size=0 stage=3 downstream_exp_name=prise97_ep10 downstream_task_name=-1 downstream_task_suite=libero_90 num_train_steps=483 eval_freq=23 max_traj_per_task=45 seed=97 checkpoint_name=snapshot_ckpt12000
python train_prise.py exp_name=prise97_ep20 replay_buffer_num_workers=2 batch_size=0 stage=3 downstream_exp_name=prise97_ep20 downstream_task_name=-1 downstream_task_suite=libero_90 num_train_steps=483 eval_freq=23 max_traj_per_task=45 seed=97 checkpoint_name=snapshot_ckpt24000 port=12233
