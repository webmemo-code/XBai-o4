set -x

export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HYDRA_FULL_ERROR=1
# Set XFormers backend to avoid CUDA errors


output_dir=./outputs/Qwen3_32b_score
mkdir -p $output_dir
data_path=./data/

ref_model_path=Qwen3/Qwen3-32B

base_name="Qwen3-32B"
LR=1e-7
train_batch_size=256
ppo_mini_batch_size=32
val_batch_size=256
max_prompt_length=2048
max_response_length=32768
ulysses_sequence_parallel_size=8

ppo_max_token_len_per_gpu=32768

echo ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu

rollout_n=8
kl_coef=0.001
entropy_coeff=0.001
exp_name="$base_name-LR$LR-ROLLOUT$rollout_n"
echo $exp_name

# For async rollout mode, dataset should return raw chat.
rollout_mode="sync"
if [ "$rollout_mode" = "async" ]; then
    return_raw_chat="True"
    chat_scheduler=examples.ppo_trainer.naive_chat_scheduler.NaiveChatCompletionScheduler
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_path/train.parquet \
    data.val_files=$data_path/test.parquet \
    data.return_raw_chat=$return_raw_chat \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.use_score=True \
    actor_rollout_ref.score.channel=5120 \
    actor_rollout_ref.score.optim.lr=1e-06 \
    actor_rollout_ref.score.optim.name=AdamW \
    reward_model.reward_manager=prime \
    actor_rollout_ref.model.path=$ref_model_path \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.optim.name=AdamW \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    +actor_rollout_ref.rollout.chat_scheduler=$chat_scheduler \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=$output_dir \
    trainer.default_local_dir=$output_dir \
    trainer.project_name="$base_name" \
    trainer.experiment_name="$exp_name" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=8 \
    trainer.save_freq=5 \
    trainer.test_freq=1000000 \
    trainer.total_epochs=1 2>&1 | tee $output_dir/train.log
