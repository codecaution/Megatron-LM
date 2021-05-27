#! /bin/bash

# Runs the "1.5B" parameter model

workspace=/mnt/xiaonan/large_model_training/
GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=${PAI_HOST_IP_worker_0}
MASTER_PORT=${PAI_worker_0_http_PORT}
NNODES=${PAI_TASK_ROLE_TASK_COUNT_worker}
NODE_RANK=${PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
CHECKPOINT_DIR=$workspace/Megatron-LM-Results/checkpoints/GPT2_Medium/4x4/
LOG_DIR=$workspace/Megatron-LM-Results/logs/GPT2_Medium/4x4/

if [ ! -d "$CHECKPOINT_DIR" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p "$CHECKPOINT_DIR"
fi

if [ ! -d "$LOG_DIR" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p "$LOG_DIR"
fi

ModelSize=Medium

DATA_PATH=$workspace/Megatron-LM-Data/Data/GPT2Data/medium/gpt2medium_text_document
CHECKPOINT_PATH=$CHECKPOINT_DIR/rank_${NODE_RANK}
LOG_PATH=$LOG_DIR/rank_${NODE_RANK}.log

VOCAB_PATH=$workspace/Megatron-LM-Data/Data/GPT2Data/medium/gpt2medium-vocab.json
MERGE_PATH=$workspace/Megatron-LM-Data/Data/GPT2Data/medium/gpt2medium-merges.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

GPT2Small_ARGS="--num-layers 12 --hidden-size 768 --num-attention-heads 12"
GPT2Medium_ARGS="--num-layers 24 --hidden-size 1024 --num-attention-heads 16"
GPT2Large_ARGS="--num-layers 36 --hidden-size 1280 --num-attention-heads 20"
GPT2XL_ARGS="--num-layers 48 --hidden-size 1600 --num-attention-heads 25"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       ${GPT2Medium_ARGS} \
       --micro-batch-size 4 \
       --global-batch-size 512 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 320000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_PATH \
       --merge-file $MERGE_PATH \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 3e-4 \
       --lr-decay-style cosine \
       --adam-beta2 0.999 \
       --min-lr 1.0e-5 \
       --weight-decay 0 \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --clip-grad 1.0 \
       --lr-warmup-fraction 0.1\
       --checkpoint-activations \
       --log-interval 100 \
       --save-interval 3000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --DDP-impl torch \
       --fp16 | tee $LOG_PATH
