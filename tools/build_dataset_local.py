from megatron import get_args
from megatron import print_rank_0
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples
from megatron.data.gpt_dataset import _build_train_valid_test_datasets

def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    # args = get_args()
    
    train_iters = 1000000
    global_batch_size = 1024
    eval_interval = 1000
    eval_iters = 10
    consumed_train_samples = 0
    consumed_valid_samples = 0
    seq_length = 1024
    seed = 1234
    data_impl = 'mmap'
    mmap_warmup = False
    split = "949,50,1"
    data_path = ["/home/v-xiaonannie/Data/C4_OWT_Data/c4_openwebtext_text_document"]

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Data loader only on rank 0 of each model parallel group.
    # Number of train/valid/test samples.

    train_samples = train_iters * global_batch_size
    
    eval_iters = (train_iters // eval_interval + 1) * eval_iters
    test_iters = eval_iters
    train_val_test_num_samples = [train_samples,
                                    eval_iters * global_batch_size,
                                    test_iters * global_batch_size]
    
    print_rank_0(' > datasets target sizes (minimum size):')
    print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
    print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
    print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

    # Build the datasets.
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
        train_val_test_num_samples)

    # Build dataloders.
    # train_dataloader = build_pretraining_data_loader(
    #     train_ds, consumed_train_samples)
    # valid_dataloader = build_pretraining_data_loader(
    #     valid_ds, consumed_valid_samples)
    # test_dataloader = build_pretraining_data_loader(test_ds, 0)

    # # Flags to know if we need to do training/validation/testing.
    # do_train = train_dataloader is not None and train_iters > 0
    # do_valid = valid_dataloader is not None and eval_iters > 0
    # do_test = test_dataloader is not None and eval_iters > 0
    # # Need to broadcast num_tokens and num_type_tokens.
    # flags = torch.cuda.LongTensor(
    #     [int(do_train), int(do_valid), int(do_test)])

    # return train_data_iterator, valid_data_iterator, test_data_iterator
    return

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    # args = get_args()
    train_iters = 1000000
    global_batch_size = 1024
    eval_interval = 1000
    eval_iters = 10
    consumed_train_samples = 0
    consumed_valid_samples = 0
    seq_length = 1024
    seed = 1234
    data_impl = 'mmap'
    mmap_warmup = False
    split = "949,50,1"
    data_path = ["/home/v-xiaonannie/Data/C4_OWT_Data/c4_openwebtext_text_document"]

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=data_path,
        data_impl=data_impl,
        splits_string=split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=seq_length,
        seed=seed,
        skip_warmup=(not mmap_warmup))
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds

def build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                    train_valid_test_num_samples,
                                    seq_length, seed, skip_warmup):
    """Build train, valid, and test datasets."""


    print(data_prefix)
    print(len(data_prefix))
    # Single dataset.
    if len(data_prefix) == 1:
        return _build_train_valid_test_datasets(data_prefix[0],
                                                data_impl, splits_string,
                                                train_valid_test_num_samples,
                                                seq_length, seed, skip_warmup)

    # Blending dataset.
    # Parse the values.
    output = get_datasets_weights_and_num_samples(data_prefix,
                                                  train_valid_test_num_samples)

    prefixes, weights, datasets_train_valid_test_num_samples = output

    # Build individual datasets.
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    for i in range(len(prefixes)):
        train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
            prefixes[i], data_impl, splits_string,
            datasets_train_valid_test_num_samples[i],
            seq_length, seed, skip_warmup)
        if train_ds:
            train_datasets.append(train_ds)
        if valid_ds:
            valid_datasets.append(valid_ds)
        if test_ds:
            test_datasets.append(test_ds)

    # Blend.
    blending_train_dataset = None
    if train_datasets:
        blending_train_dataset = BlendableDataset(train_datasets, weights)
    blending_valid_dataset = None
    if valid_datasets:
        blending_valid_dataset = BlendableDataset(valid_datasets, weights)
    blending_test_dataset = None
    if test_datasets:
        blending_test_dataset = BlendableDataset(test_datasets, weights)

    return (blending_train_dataset, blending_valid_dataset,
            blending_test_dataset)

if __name__ == "__main__":

    build_train_valid_test_data_iterators(train_valid_test_datasets_provider)
