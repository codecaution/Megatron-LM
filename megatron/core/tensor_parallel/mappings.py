# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_embedding_model_parallel_world_size,
    get_embedding_model_parallel_rank,
    get_embedding_model_parallel_group,
)
from .utils import split_tensor_along_last_dim


def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(
        input_, group=get_tensor_model_parallel_group())

    return input_


def _exchange_across_embedding_parallel_group(input_, extend_input=True, reduce_output=True):
    """Gather tensors across embedding parallel group."""

    tensor_model_size = get_tensor_model_parallel_world_size()
    embedding_model_size = get_embedding_model_parallel_world_size()
    # Bypass the function if embedding parallel size is equal to tensor model parallel size.
    if tensor_model_size == embedding_model_size:
        return input_

    # Gather across embedding parallel group.
    num_pieces = embedding_model_size // tensor_model_size
    offset = get_embedding_model_parallel_rank() % tensor_model_size

    if extend_input == True:
        extended_input = torch.cat([input_]*num_pieces)
        num_splits = input_.size(0)
    else:
        extended_input = input_
        num_splits = input_.size(0) // num_pieces
    output = torch.empty_like(extended_input)
    input_splits = [0] * embedding_model_size
    for i in range(num_pieces):
        input_splits[i * tensor_model_size + offset] = num_splits
    torch.distributed.all_to_all_single(
        output, extended_input, input_splits, input_splits, group=get_embedding_model_parallel_group())
    if reduce_output:
        output = sum(torch.chunk(output, num_pieces))
    return output

'''
def _gather_across_embedding_parallel_group(input_, extend=True):
    """Gather tensors across embedding parallel group."""

    tensor_model_size = get_tensor_model_parallel_world_size()
    embedding_model_size = get_embedding_model_parallel_world_size()
    # Bypass the function if embedding parallel size is equal to tensor model parallel size.
    if tensor_model_size == embedding_model_size:
        return input_

    # Gather across embedding parallel group.
    num_pieces = embedding_model_size // tensor_model_size
    offset = get_embedding_model_parallel_rank() % tensor_model_size

    if extend == True:
        extended_input = torch.cat([input_]*num_pieces)
    else:
        extended_input = input_
    output = torch.empty_like(extended_input)
    input_splits = [0] * embedding_model_size
    for i in range(num_pieces):
        input_splits[i * tensor_model_size + offset] = input_.size(0)
    torch.distributed.all_to_all_single(
        output, extended_input, input_splits, input_splits, group=get_embedding_model_parallel_group())
    # if backward == True:
    #     output = output / num_pieces
    return output


def _scatter_across_embedding_parallel_group(input_):
    """Gather tensors across embedding parallel group."""

    tensor_model_size = get_tensor_model_parallel_world_size()
    embedding_model_size = get_embedding_model_parallel_world_size()
    # Bypass the function if embedding parallel size is equal to tensor model parallel size.
    if tensor_model_size == embedding_model_size:
        return input_

    # scatter across embedding parallel group.
    num_pieces = embedding_model_size // tensor_model_size
    offset = get_embedding_model_parallel_rank() % tensor_model_size

    output = torch.empty_like(input_)
    input_splits = [0] * embedding_model_size
    for i in range(num_pieces):
        input_splits[i * tensor_model_size +
                     offset] = input_.size(0)//num_pieces
    torch.distributed.all_to_all_single(
        output, input_, input_splits, input_splits, group=get_embedding_model_parallel_group())

    output = sum(torch.chunk(output, num_pieces))
    return output
'''

def _split_along_last_dim(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output


def _split_along_first_dim(input_):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert dim_size % world_size == 0, \
        "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = get_tensor_model_parallel_rank()
    dim_offset = rank * local_dim_size

    output = input_[dim_offset:dim_offset+local_dim_size].contiguous()

    return output


def _gather_along_last_dim(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(
        tensor_list, input_, group=get_tensor_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


def _gather_along_first_dim(input_):
    """Gather tensors and concatinate along the first dimension."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=torch.cuda.current_device())
    torch.distributed._all_gather_base(output, input_.contiguous(),
                                       group=get_tensor_model_parallel_group())

    return output


def _reduce_scatter_along_first_dim(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0, \
        "First dimension of the tensor should be divisible by tensor parallel size"

    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=torch.cuda.current_device())
    torch.distributed._reduce_scatter_base(output, input_.contiguous(),
                                           group=get_tensor_model_parallel_group())
    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _GatherFromEmbeddingModelParallelRegion(torch.autograd.Function):
    """gather the input from the embedding parallel region."""
    @staticmethod
    def symbolic(graph, input_):
        return _exchange_across_embedding_parallel_group(input_, extend_input=True, reduce_output=False)

    @staticmethod
    def forward(ctx, input_):
        return _exchange_across_embedding_parallel_group(input_, extend_input=True, reduce_output=False)

    @staticmethod
    def backward(ctx, grad_output):
        return _exchange_across_embedding_parallel_group(grad_output, extend_input=False, reduce_output=True)


class _ReduceFromEmbeddingModelParallelRegion(torch.autograd.Function):
    """scatter the input from the embedding parallel region."""
    @staticmethod
    def symbolic(graph, input_):
        return _exchange_across_embedding_parallel_group(input_, extend_input=False, reduce_output=True)

    @staticmethod
    def forward(ctx, input_):
        return _exchange_across_embedding_parallel_group(input_, extend_input=False, reduce_output=True)

    @staticmethod
    def backward(ctx, grad_output):
        return _exchange_across_embedding_parallel_group(grad_output, extend_input=True, reduce_output=False) / (get_embedding_model_parallel_world_size() // get_tensor_model_parallel_world_size())


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output)


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, tensor_parallel_output_grad=True):
        return _gather_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_, tensor_parallel_output_grad=True):
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        return _gather_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad

        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce
        # scattered and whereas if the computation is duplicated,
        # output gradients need to be scattered.
        if tensor_parallel_output_grad:
            return _reduce_scatter_along_first_dim(grad_output), None
        else:
            return _split_along_first_dim(grad_output), None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


# -----------------
# Helper functions.
# -----------------

def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def gather_from_embedding_model_parallel_region(input_):
    return _GatherFromEmbeddingModelParallelRegion.apply(input_)


def reduce_from_embedding_model_parallel_region(input_):
    return _ReduceFromEmbeddingModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)


def scatter_to_sequence_parallel_region(input_):
    return _ScatterToSequenceParallelRegion.apply(input_)


def gather_from_sequence_parallel_region(input_, tensor_parallel_output_grad=True):
    return _GatherFromSequenceParallelRegion.apply(input_, tensor_parallel_output_grad)


def reduce_scatter_to_sequence_parallel_region(input_):
    return _ReduceScatterToSequenceParallelRegion.apply(input_)
