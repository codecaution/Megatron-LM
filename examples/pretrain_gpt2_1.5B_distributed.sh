#! /bin/bash
# Runs the "1.5B" parameter model
ModelSize=GPT2XL_ARGS
Version=c4owt


GPU_PER_NODE_COUNT=$DLWS_NUM_GPU_PER_WORKER
NODE_COUNT=$DLWS_NUM_WORKER
# Path
workspace=/home/v-xiaonannie/xiaonan/large_model_training/

DATA_DIR=$workspace/Megatron-LM-Data/Data/C4_OWT_Data/
CHECKPOINT_DIR=$workspace/Megatron-LM-Results/checkpoints/hf-GPT2/${ModelSize}/${NODE_COUNT}x${GPU_PER_NODE_COUNT}__${Version}/
LOG_DIR=$workspace/Megatron-LM-Results/logs/hf-GPT2/${ModelSize}/${NODE_COUNT}x${GPU_PER_NODE_COUNT}_${Version}/

if [ ! -d "$CHECKPOINT_DIR" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p "$CHECKPOINT_DIR"
fi

if [ ! -d "$LOG_DIR" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p "$LOG_DIR"
fi

# Read Data Dir
DATA_PATH=$DATA_DIR/c4_openwebtext_text_document
VOCAB_PATH=$DATA_DIR/vocab.json
MERGE_PATH=$DATA_DIR/merges.txt
# Write log and checkpoints
CHECKPOINT_PATH=$CHECKPOINT_DIR
LOG_PATH=$LOG_DIR/rank_${NODE_RANK}.log

# DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
DISTRIBUTED_ARGS="--nproc_per_node $GPU_PER_NODE_COUNT --nnodes $NODE_COUNT --node_rank $NODE_RANK --master_addr $MASTER_IP --master_port $MASTER_PORT"

GPT2Small_ARGS="--num-layers 12 --hidden-size 768 --num-attention-heads 12"
GPT2Medium_ARGS="--num-layers 24 --hidden-size 1024 --num-attention-heads 16"
GPT2Large_ARGS="--num-layers 36 --hidden-size 1280 --num-attention-heads 20"
GPT2XL_ARGS="--num-layers 48 --hidden-size 1600 --num-attention-heads 25"

python -u -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       ${GPT2XL_ARGS} \
       --micro-batch-size 2 \
       --global-batch-size 1024 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 1000000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_PATH \
       --merge-file $MERGE_PATH \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 2e-4 \
       --lr-decay-style cosine \
       --adam-beta2 0.95 \
       --min-lr 1.0e-5 \
       --weight-decay 0.1 \
       --attention-dropout 0.1 \
       --hidden-dropout 0.1 \
       --clip-grad 1.0 \
       --lr-warmup-fraction 0.1\
       --checkpoint-activations \
       --log-interval 100 \
       --save-interval 5000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --DDP-impl torch \
       --fp16 | tee $LOG_PATH

