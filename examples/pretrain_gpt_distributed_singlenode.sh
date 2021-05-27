#! /bin/bash

# Runs the "1.5B" parameter model
 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt2/hpcx-v2.8.0-gcc-MLNX_OFED_LINUX-5.2-1.0.4.0-ubuntu20.04-x86_64/nccl_rdma_sharp_plugin/lib/
 export UCX_IB_ENABLE_CUDA_AFFINITY=n 
 export NCCL_IB_PCI_RELAXED_ORDERING=1
 export UCX_IB_PCI_RELAXED_ORDERING=on 
 export UCX_NET_DEVICES=mlx5_0:1 
 export UCX_TLS=rc 
 export NCCL_SOCKET_IFNAME=eth0 
 export CUDA_DEVICE_ORDER=PCI_BUS_ID 
 export NCCL_NET_GDR_LEVEL=5 
 export LD_PRELOAD=/opt2/hpcx-v2.8.0-gcc-MLNX_OFED_LINUX-5.2-1.0.4.0-ubuntu20.04-x86_64/nccl_rdma_sharp_plugin/lib/libnccl-net.so 
 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt2/hpcx-v2.8.0-gcc-MLNX_OFED_LINUX-5.2-1.0.4.0-ubuntu20.04-x86_64/sharp/lib 
 export NCCL_TOPO_FILE=/opt2/msft/topo.xml 


GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=7000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
workspace=/home/xiaonan
CHECKPOINT_DIR=$workspace/Results/checkpoints
LOG_DIR=$workspace/Results/logs

if [ ! -d "$CHECKPOINT_DIR" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p "$CHECKPOINT_DIR"
fi

if [ ! -d "$LOG_DIR" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p "$LOG_DIR"
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt2/hpcx-v2.8.0-gcc-MLNX_OFED_LINUX-5.2-1.0.4.0-ubuntu20.04-x86_64/nccl_rdma_sharp_plugin/lib/
DATA_PATH=$workspace/Data/GPT2Data/xlarge/gpt2xl_text_document
CHECKPOINT_PATH=$CHECKPOINT_DIR/GPT2xl4x8_rank
LOG_PATH=$LOG_DIR/GPT2xl4x8_rank$NODE_RANK.log

VOCAB_PATH=$workspace/Data/GPT2Data/xlarge/gpt2xl-vocab.json
MERGE_PATH=$workspace/Data/GPT2Data/xlarge/gpt2xl-merges.txt

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
       --log-interval 10 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 | tee $LOG_PATH
