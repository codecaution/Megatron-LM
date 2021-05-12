#! /bin/bash

# Runs the "1.5B" parameter model

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=10.5.0.5
MASTER_PORT=7000
NNODES=4
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_DIR=/workspace/Results/checkpoints
LOG_DIR=/workspace/Results/logs

if [ ! -d "$CHECKPOINT_DIR" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p "$CHECKPOINT_DIR"
fi

if [ ! -d "$LOG_DIR" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p "$LOG_DIR"
fi

DATA_PATH=/workspace/Data/GPT2Data/xlarge/gpt2xl_text_document
CHECKPOINT_PATH=$CHECKPOINT_DIR/GPT2xl4x8_rank$NODE_RANK
LOG_PATH=$LOG_DIR/GPT2xl4x8_rank$NODE_RANK.log

VOCAB_PATH=/workspace/Data/GPT2Data/xlarge/gpt2xl-vocab.json
MERGE_PATH=/workspace/Data/GPT2Data/xlarge/gpt2xl-merges.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --num-layers 48 \
       --hidden-size 1600 \
       --num-attention-heads 25 \
       --micro-batch-size 8 \
       --global-batch-size 512 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 1000000 \
       --lr-decay-iters 50000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_PATH \
       --merge-file $MERGE_PATH \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00025 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-iters 2000\
       --checkpoint-activations \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 | tee $LOG_PATH
