set -x

export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HYDRA_FULL_ERROR=1

ref_model_path="agentica-org/DeepScaleR-1.5B-Preview"
output_dir=./outputs/
data_dir=./data/

mkdir -p $output_dir
export N_NODES=1
export N_GPUS=8
export HYDRA_FULL_ERROR=1
export ROLLOUT_TP_SIZE=1
max_prompt_length=1024
max_response_length=8192
ppo_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length)*2 ))
echo $ppo_max_token_len_per_gpu
train_batch_size=32
ppo_mini_batch_size=8
ppo_micro_batch_size=4
ulysses_sequence_parallel_size=2
base_name="1.5B"
rollout_n=8


exp_name="$base_name-LR$LR-ROLLOUT$rollout_n"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_dir/train.parquet \
    data.val_files=$data_dir/aime.parquet \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    reward_model.reward_manager=prime \
    actor_rollout_ref.model.path=$ref_model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.use_score=True \
    actor_rollout_ref.score.channel=1536 \
    actor_rollout_ref.score.optim.lr=1e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_micro_batch_size=$ppo_micro_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.actor.optim.no_load_optim=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    +actor_rollout_ref.rollout.swap_space=256 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    actor_rollout_ref.actor.optim.no_load_optim=True \
    trainer.logger=['console', 'wandb'] \
    trainer.project_name="$base_name" \
    trainer.experiment_name="$exp_name" \
    trainer.default_hdfs_dir=$output_dir \
    trainer.default_local_dir=$output_dir \
    trainer.experiment_name='15b_function_rm' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=20 \
    trainer.test_freq=1000000 \
    trainer.total_epochs=1 2>&1 | tee $output_dir/train.log
