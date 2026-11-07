#!/usr/bin/env bash
set -euo pipefail

# 教学友好版：支持单卡 / 多卡的最小预训练脚本
#
# 用法：
#   bash scripts/run_pretrain_demo.sh
#
# 常用改法：
#   DEVICE=cpu bash scripts/run_pretrain_demo.sh
#   BATCH_SIZE=256 bash scripts/run_pretrain_demo.sh
#   NPROC_PER_NODE=4 BATCH_SIZE=512 bash scripts/run_pretrain_demo.sh
#   HIDDEN_SIZE=768 NUM_ATTENTION_HEADS=12 HEAD_SIZE=64 bash scripts/run_pretrain_demo.sh
#   ARCHITECTURE_TYPE=moe bash scripts/run_pretrain_demo.sh
#   ARCHITECTURE_TYPE=moe MOE_INTERMEDIATE_SIZE=256 N_ROUTED_EXPERTS=8 bash scripts/run_pretrain_demo.sh
#
# 可选环境变量：
#   DATA_BIN, SAVE_DIR, DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE
#   LOG_INTERVAL, SAVE_INTERVAL, EVAL_INTERVAL
#   USE_COMPILE, ARCHITECTURE_TYPE
#   HIDDEN_SIZE, NUM_HIDDEN_LAYERS, NUM_ATTENTION_HEADS, HEAD_SIZE
#   NUM_KEY_VALUE_HEADS, INTERMEDIATE_SIZE, VOCAB_SIZE
#   MAX_SEQ_LEN, MAX_POSITION_EMBEDDINGS, ROPE_THETA, HIDDEN_ACT, DROPOUT
#   ATTN_GATE_TYPE, ATTN_GATE_INIT_BIAS, ENABLE_GATE_MONITOR
#   MOE_INTERMEDIATE_SIZE, N_SHARED_EXPERTS, N_ROUTED_EXPERTS,
#   NUM_EXPERTS_PER_TOK, MOE_AUX_LOSS_COEF（仅 ARCHITECTURE_TYPE=moe 生效）
#   NPROC_PER_NODE, MASTER_PORT
#
# 架构说明：
#   ARCHITECTURE_TYPE=dense 时使用 INTERMEDIATE_SIZE（FFN 维度）；
#   ARCHITECTURE_TYPE=moe   时使用 MOE_INTERMEDIATE_SIZE 及专家数量等 MoE 超参。
#
# 注意：BATCH_SIZE 是每个进程/每张卡的 batch，
#       全局 batch = BATCH_SIZE × NPROC_PER_NODE（× accumulation_steps）。

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

# 0）环境变量
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE="${ENV_FILE:-$ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
  echo "[info] loaded env file: $ENV_FILE"
fi

# 1) 训练数据与输出目录
DATA_BIN="${DATA_BIN:-$ROOT/data/pretrain_data/spongebob_pretrain_512.bin}"
SAVE_DIR="${SAVE_DIR:-$ROOT/pretrain_out/demo}"

# 2) 最核心的训练参数
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-2}"
# dense=标准 FFN，moe=混合专家
ARCHITECTURE_TYPE="${ARCHITECTURE_TYPE:-dense}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-3000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-12}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-12}"
HEAD_SIZE="${HEAD_SIZE:-64}"
NUM_KEY_VALUE_HEADS="${NUM_KEY_VALUE_HEADS:-4}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-2048}"
VOCAB_SIZE="${VOCAB_SIZE:-15000}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-32768}"
ROPE_THETA="${ROPE_THETA:-10000.0}"
HIDDEN_ACT="${HIDDEN_ACT:-silu}"
DROPOUT="${DROPOUT:-0.0}"

# MoE 专家超参（仅 ARCHITECTURE_TYPE=moe 时生效），默认值与 model/config_moe.py 一致
MOE_INTERMEDIATE_SIZE="${MOE_INTERMEDIATE_SIZE:-512}"
N_SHARED_EXPERTS="${N_SHARED_EXPERTS:-1}"
N_ROUTED_EXPERTS="${N_ROUTED_EXPERTS:-15}"
NUM_EXPERTS_PER_TOK="${NUM_EXPERTS_PER_TOK:-3}"
MOE_AUX_LOSS_COEF="${MOE_AUX_LOSS_COEF:-0.01}"

# gated attention
# 默认值与 model/config.py 的 GrandLineConfig 保持一致；
# 做门控实验时可覆盖为：ATTN_GATE_TYPE=head ENABLE_GATE_MONITOR=1
ATTN_GATE_TYPE="${ATTN_GATE_TYPE:-none}"
ATTN_GATE_INIT_BIAS="${ATTN_GATE_INIT_BIAS:-4.0}"
ENABLE_GATE_MONITOR="${ENABLE_GATE_MONITOR:-0}"



# 3) 分布式参数
# NPROC_PER_NODE=1 表示单卡；大于 1 表示单机多卡
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# 4）评测/swanlab记录参数
EVAL_BENCH="${EVAL_BENCH:-1}"
USE_SWANLAB="${USE_SWANLAB:-1}"
USE_COMPILE="${USE_COMPILE:-1}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-GrandLine-Pretrain}"

case "$ARCHITECTURE_TYPE" in
  dense|moe) ;;
  *)
    echo "[error] ARCHITECTURE_TYPE 只能是 dense 或 moe，当前为: $ARCHITECTURE_TYPE" >&2
    exit 1
    ;;
esac

if [[ -n "${GLOBAL_BATCH_SIZE:-}" ]]; then
  echo "[warn] GLOBAL_BATCH_SIZE 已废弃且会被忽略；请改用 BATCH_SIZE（每卡/每进程 batch）" >&2
fi

if [[ ! -f "$DATA_BIN" ]]; then
  echo "[error] data file not found: $DATA_BIN"
  exit 1
fi

mkdir -p "$SAVE_DIR"

TRAIN_ARGS=(
  --data_path "$DATA_BIN"
  --save_dir "$SAVE_DIR"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --architecture_type "$ARCHITECTURE_TYPE"
  --learning_rate "$LEARNING_RATE"
  --log_interval "$LOG_INTERVAL"
  --save_interval "$SAVE_INTERVAL"
  --eval_interval "$EVAL_INTERVAL"
  --hidden_size "$HIDDEN_SIZE"
  --num_hidden_layers "$NUM_HIDDEN_LAYERS"
  --num_attention_heads "$NUM_ATTENTION_HEADS"
  --num_key_value_heads "$NUM_KEY_VALUE_HEADS"
  --vocab_size "$VOCAB_SIZE"
  --max_seq_len "$MAX_SEQ_LEN"
  --max_position_embeddings "$MAX_POSITION_EMBEDDINGS"
  --rope_theta "$ROPE_THETA"
  --hidden_act "$HIDDEN_ACT"
  --dropout "$DROPOUT"
  --from_weight none
  --from_resume 0
  --use_swanlab "$USE_SWANLAB"
  --use_compile "$USE_COMPILE"
  --eval_bench "$EVAL_BENCH"
  --swanlab_project "$SWANLAB_PROJECT"
  --head_size "$HEAD_SIZE"
  # gated attention
  --attn_gate_type "$ATTN_GATE_TYPE"
  --attn_gate_init_bias "$ATTN_GATE_INIT_BIAS"
  --enable_gate_monitor "$ENABLE_GATE_MONITOR"
)

# 按架构追加互斥的 FFN / 专家超参
if [[ "$ARCHITECTURE_TYPE" == "dense" ]]; then
  TRAIN_ARGS+=(--intermediate_size "$INTERMEDIATE_SIZE")
else
  TRAIN_ARGS+=(
    --moe_intermediate_size "$MOE_INTERMEDIATE_SIZE"
    --n_shared_experts "$N_SHARED_EXPERTS"
    --n_routed_experts "$N_ROUTED_EXPERTS"
    --num_experts_per_tok "$NUM_EXPERTS_PER_TOK"
    --moe_aux_loss_coef "$MOE_AUX_LOSS_COEF"
  )
fi

echo "[info] data_path=$DATA_BIN"
echo "[info] save_dir=$SAVE_DIR"
echo "[info] epochs=$EPOCHS"
echo "[info] architecture_type=$ARCHITECTURE_TYPE"
echo "[info] batch_size_per_process=$BATCH_SIZE"
echo "[info] global_batch_size=$((BATCH_SIZE * NPROC_PER_NODE))"
echo "[info] use_compile=$USE_COMPILE"
echo "[info] learning_rate=$LEARNING_RATE"
echo "[info] log_interval=$LOG_INTERVAL"
echo "[info] save_interval=$SAVE_INTERVAL"
echo "[info] eval_interval=$EVAL_INTERVAL"
echo "[info] hidden_size=$HIDDEN_SIZE"
echo "[info] num_hidden_layers=$NUM_HIDDEN_LAYERS"
echo "[info] num_attention_heads=$NUM_ATTENTION_HEADS"
echo "[info] head_size=$HEAD_SIZE"
echo "[info] num_key_value_heads=$NUM_KEY_VALUE_HEADS"
if [[ "$ARCHITECTURE_TYPE" == "dense" ]]; then
  echo "[info] intermediate_size=$INTERMEDIATE_SIZE"
else
  echo "[info] moe_intermediate_size=$MOE_INTERMEDIATE_SIZE"
  echo "[info] n_shared_experts=$N_SHARED_EXPERTS"
  echo "[info] n_routed_experts=$N_ROUTED_EXPERTS"
  echo "[info] num_experts_per_tok=$NUM_EXPERTS_PER_TOK"
  echo "[info] moe_aux_loss_coef=$MOE_AUX_LOSS_COEF"
fi
echo "[info] vocab_size=$VOCAB_SIZE"
echo "[info] max_seq_len=$MAX_SEQ_LEN"
echo "[info] max_position_embeddings=$MAX_POSITION_EMBEDDINGS"
echo "[info] rope_theta=$ROPE_THETA"
echo "[info] hidden_act=$HIDDEN_ACT"
echo "[info] dropout=$DROPOUT"
# gated attention
echo "[info] attn_gate_type=$ATTN_GATE_TYPE"
echo "[info] attn_gate_init_bias=$ATTN_GATE_INIT_BIAS"
echo "[info] enable_gate_monitor=$ENABLE_GATE_MONITOR"
echo "[info] nproc_per_node=$NPROC_PER_NODE"

if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
  echo "[run] single GPU / single process"
  python train/pretrain.py \
    --device "$DEVICE" \
    "${TRAIN_ARGS[@]}"
else
  echo "[run] multi GPU with torchrun"
  echo "[info] per-rank batch=$BATCH_SIZE, global batch=$((BATCH_SIZE * NPROC_PER_NODE))"
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_port "$MASTER_PORT" \
    train/pretrain.py \
    "${TRAIN_ARGS[@]}"
fi
