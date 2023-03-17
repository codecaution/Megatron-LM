from megatron.core import tensor_parallel
from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_embedding_model_parallel_world_size,
    get_embedding_model_parallel_rank,
    get_embedding_model_parallel_group,
)
import torch
from tests.test_utilities import Utils
import numpy as np
import random
import torch.nn.init as init
from torch.nn.parameter import Parameter


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensor_parallel.model_parallel_cuda_manual_seed(seed)

def test_vocab_parallel_embedding():
    Utils.initialize_model_parallel(8, 1, 1)

    batch_size = 17
    seq_length = 23
    vocab_size = 48
    hidden_size = 16
    seed = 1234

    set_random_seed(seed)
    input_data = torch.LongTensor(
        size=(batch_size, seq_length)).random_(0, vocab_size).cuda()
    loss_weight = torch.randn([batch_size, seq_length, hidden_size]).cuda()

    set_random_seed(seed)
    embedding_original = torch.nn.Embedding(vocab_size, hidden_size).cuda()

    output = embedding_original(input_data)
    loss_original = torch.mul(output, loss_weight).sum()
    loss_original.backward()

    set_random_seed(seed)
    embedding_vocab_parallel = tensor_parallel.VocabParallelEmbedding(
        vocab_size, hidden_size, init_method=init.normal_, use_cpu_initialization=True).cuda()
    output = embedding_vocab_parallel(input_data)
    loss_vocab_parallel = torch.mul(output, loss_weight).sum()
    loss_vocab_parallel.backward()

    torch.distributed.barrier()
    # assert(torch.allclose(loss_original, loss_vocab_parallel))
    assert (torch.equal(loss_original, loss_vocab_parallel))

    weight_grad_orig = torch.split(embedding_original.weight.grad,
                                   vocab_size // get_tensor_model_parallel_world_size(),
                                   0)[get_tensor_model_parallel_rank()]

    assert (torch.equal(weight_grad_orig, embedding_vocab_parallel.weight.grad))

    Utils.destroy_model_parallel()


def test_embedding_parallel_embedding():
    batch_size = 32
    seq_length = 24
    vocab_size = 1024
    hidden_size = 128
    seed = 1234

    Utils.initialize_model_parallel(4, 1, 1)

    set_random_seed(seed)
    input_data1 = torch.LongTensor(
        size=(batch_size, seq_length)).random_(0, vocab_size).cuda()
    loss_weight1 = torch.randn([batch_size, seq_length, hidden_size]).cuda()

    set_random_seed(seed)
    embedding_vocab_parallel = tensor_parallel.VocabParallelEmbedding(
        vocab_size, hidden_size, init_method=init.normal_, use_cpu_initialization=True).cuda()
    parallel_output = embedding_vocab_parallel(input_data1)
    loss_vocab_parallel = torch.mul(parallel_output, loss_weight1).sum()
    loss_vocab_parallel.backward()

    Utils.destroy_model_parallel()

    Utils.initialize_model_parallel(4, 1, 8)

    set_random_seed(seed)
    input_data2 = torch.LongTensor(
        size=(batch_size, seq_length)).random_(0, vocab_size).cuda()
    loss_weight2 = torch.randn([batch_size, seq_length, hidden_size]).cuda()

    set_random_seed(seed)
    embedding_vocab_parallel_ep = tensor_parallel.VocabParallelEmbedding(
        vocab_size, hidden_size, init_method=init.normal_, use_cpu_initialization=True).cuda()
    parallel_output = embedding_vocab_parallel_ep(input_data2)
    loss_vocab_parallel_ep = torch.mul(parallel_output, loss_weight2).sum()
    loss_vocab_parallel_ep.backward()

    torch.distributed.barrier()
    # assert(torch.allclose(loss_original, loss_vocab_parallel))
    assert (torch.equal(loss_vocab_parallel, loss_vocab_parallel_ep))
    if get_embedding_model_parallel_rank() < 4:
        # print(torch.abs(embedding_vocab_parallel.weight.grad[:(vocab_size // get_embedding_model_parallel_world_size())] - embedding_vocab_parallel_ep.weight.grad).max())
        # assert (torch.allclose(embedding_vocab_parallel.weight.grad[:(vocab_size // get_embedding_model_parallel_world_size())], embedding_vocab_parallel_ep.weight.grad, atol=1e-12))
        print(f"xxxxxxx, {(embedding_vocab_parallel.weight.grad[:(vocab_size // get_embedding_model_parallel_world_size())] - embedding_vocab_parallel_ep.weight.grad).abs().max()}")
        assert (torch.equal(embedding_vocab_parallel.weight.grad[:(vocab_size // get_embedding_model_parallel_world_size())], embedding_vocab_parallel_ep.weight.grad))
    else:
        # print(torch.abs(embedding_vocab_parallel.weight.grad[(vocab_size // get_embedding_model_parallel_world_size()):] - embedding_vocab_parallel_ep.weight.grad).max())
        # assert (torch.allclose(embedding_vocab_parallel.weight.grad[(vocab_size // get_embedding_model_parallel_world_size()):], embedding_vocab_parallel_ep.weight.grad, atol=1e-12))
        print(f"xxxxxxx, {(embedding_vocab_parallel.weight.grad[(vocab_size // get_embedding_model_parallel_world_size()):] - embedding_vocab_parallel_ep.weight.grad).abs().max()}")
        assert (torch.equal(embedding_vocab_parallel.weight.grad[(vocab_size // get_embedding_model_parallel_world_size()):], embedding_vocab_parallel_ep.weight.grad))
    Utils.destroy_model_parallel()

    
def test_linear_with_grad_accumulation_and_async_allreduce():
    Utils.initialize_model_parallel(8, 1, 1)

    batch_size = 16
    seq_length = 24
    vocab_size = 48
    hidden_size = 16
    seed = 1234

    set_random_seed(seed)
    input_data = torch.LongTensor(
        size=(batch_size, seq_length, hidden_size)).random_(0, vocab_size).cuda()

    set_random_seed(seed)
    embedding_original = torch.nn.Embedding(vocab_size, hidden_size).cuda()
    input_data = embedding_original(input_data)
    
    embedding_original_rank_weight = torch.chunk(embedding_original.weight, get_tensor_model_parallel_world_size(), 0)[get_tensor_model_parallel_rank()]

    output = torch.matmul(input_data, embedding_original_rank_weight.t())
    
    set_random_seed(seed)
    embedding_vocab_parallel = tensor_parallel.VocabParallelEmbedding(
        vocab_size, hidden_size, init_method=init.normal_, use_cpu_initialization=True).cuda()
    parallel_output = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(input_data, embedding_vocab_parallel.weight, bias=None,
                                                                                        gradient_accumulation_fusion=False,
                                                                                        async_grad_allreduce=False,
                                                                                        sequence_parallel_enabled=False,
                                                                                        parallel_logits=True)

    torch.distributed.barrier()
    # output = torch.chunk(output, get_tensor_model_parallel_world_size(), -1)[get_tensor_model_parallel_rank()]
    assert (torch.equal(output, parallel_output))
    print(output.size())
    # assert (torch.allclose(output, parallel_output, atol=1e-6))
    Utils.destroy_model_parallel()

class IdentityLayer3D(torch.nn.Module):
    def __init__(self, m, n, k):
        super(IdentityLayer3D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n, k))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self):
        return self.weight
    
def test_linear_with_grad_accumulation_and_async_allreduce_ep():
    Utils.initialize_model_parallel(4, 1, 1)

    batch_size = 32
    seq_length = 32
    vocab_size = 8
    hidden_size = 128
    seed = 1234

    set_random_seed(seed)
    input_data = torch.randn([batch_size, seq_length, hidden_size]).cuda().double()
    input_data.requires_grad = True
    bias1 = torch.ones([vocab_size//4]).cuda().double()
    bias1.requires_grad = True
    loss_weight = torch.randn([seq_length, batch_size, vocab_size//4]).cuda().double()

    set_random_seed(seed)
    embedding_vocab_parallel = tensor_parallel.VocabParallelEmbedding(
        vocab_size, hidden_size, init_method=init.normal_, use_cpu_initialization=True).cuda().double()
    

    parallel_output = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(input_data, embedding_vocab_parallel.weight, bias=bias1,
                                                                                        gradient_accumulation_fusion=False,
                                                                                        async_grad_allreduce=False,
                                                                                        sequence_parallel_enabled=False,
                                                                                        parallel_logits=True)
    loss_vocab_parallel = torch.mul(parallel_output, loss_weight).sum()
    loss_vocab_parallel.backward()
    
    torch.distributed.barrier()
    Utils.destroy_model_parallel()

    # embedding parallel
    Utils.initialize_model_parallel(4, 1, 8)

    set_random_seed(seed)
    input_data2 = torch.randn([batch_size, seq_length, hidden_size]).cuda().double()
    input_data2.requires_grad = True
    bias2 = torch.ones([vocab_size//4]).cuda().double()
    bias2.requires_grad = True

    set_random_seed(seed)
    embedding_vocab_parallel_ep = tensor_parallel.VocabParallelEmbedding(
        vocab_size, hidden_size, init_method=init.normal_, use_cpu_initialization=True).cuda().double()
    

    parallel_output_ep = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(input_data2, embedding_vocab_parallel_ep.weight, bias=bias2,
                                                                                        gradient_accumulation_fusion=False,
                                                                                        async_grad_allreduce=False,
                                                                                        sequence_parallel_enabled=False,
                                                                                        parallel_logits=True)

    loss_vocab_parallel_ep = torch.mul(parallel_output_ep, loss_weight).sum()
    loss_vocab_parallel_ep.backward()
    
    torch.distributed.barrier()

    print(torch.abs(parallel_output - parallel_output_ep).max())
    print(parallel_output.size(), parallel_output_ep.size())
    # assert (torch.equal(parallel_output, parallel_output_ep))
    # assert (torch.equal(input_data.grad, input_data2.grad))
    # assert (torch.equal(bias1.grad, bias2.grad))
    assert (torch.allclose(parallel_output, parallel_output_ep, atol=1e-12))
    assert (torch.allclose(input_data.grad, input_data2.grad, atol=1e-12))
    assert (torch.allclose(bias1.grad, bias2.grad, atol=1e-12))

    if get_embedding_model_parallel_rank() < 4:
        assert (torch.allclose(embedding_vocab_parallel.weight.grad[:(vocab_size // get_embedding_model_parallel_world_size())], embedding_vocab_parallel_ep.weight.grad, atol=1e-12))
    else:
        assert (torch.allclose(embedding_vocab_parallel.weight.grad[(vocab_size // get_embedding_model_parallel_world_size()):], embedding_vocab_parallel_ep.weight.grad, atol=1e-12))

    Utils.destroy_model_parallel()
#     loss = torch.mul(output, loss_weight).sum()
#     loss.backward()
    
