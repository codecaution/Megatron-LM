#! /bin/bash

# Runs the "345M" parameter model
export NCCL_DEBUG=WARN
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/jizhicfs/xiaonan/deepspeed/downloads/webtext2/my-gpt2_text_document
CHECKPOINT_PATH=/jizhicfs/xiaonan/logs/ep/
VOCAB_PATH=/jizhicfs/xiaonan/deepspeed/downloads/webtext2

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       ../pretrain_gpt.py \
       --num-layers 6 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 8 \
       --tensor-model-parallel-size 4 \
       --embedding-model-parallel-size 8 \
       --global-batch-size 16 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --use-cpu-initialization \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_PATH/gpt2-vocab.json \
       --merge-file $VOCAB_PATH/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 1 \
       --lr-decay-style cosine \
       --min-lr 1 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --recompute-activations \
       --recompute-granularity full \
       --no-async-tensor-model-parallel-allreduce \
       --no-gradient-accumulation-fusion \
       --log-interval 100 \
       --save-interval 500000 \
       --eval-interval 1000 \
       --eval-iters 10 2>&1 | tee -a /home/logs/GPT_8GPU_tp4_ep8.log
