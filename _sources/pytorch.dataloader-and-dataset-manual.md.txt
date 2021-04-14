# PyTorch Dataloader and Dataset Manual

Reference: [PyTorch Tensor Manual](pytorch.tensor-manual.md)
and [PyTorch Official Docs](https://pytorch.org/docs/stable/data.html)

## Data Loading in Torch

PyTorch uses two classes that provide the necessary boilerplate and other useful utilities required to load datasets in a manner amenable to feeding into a neural network. A key class in question is the `torch.utils.data.DataLoader` which wraps a `Dataset` object in a variety of modes, that is application specific. The options that can be configured on a `DataLoader` are reflected in its signature: `python dataloader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, *, prefetch_factor=2, persistent_workers=False)`

> They entry point to understanding various ways of loading Data in PyTorch is to understand the various arguments fed into a `DataLoader` object and their use.
> We will refer to the class as `DataLoader` and the object as `dataloader`.

## The `dataset` argument

The most important argument of `DataLoader` is the `dataset` object which can be of two types.

1. **map-style** datasets: This is a dataset that can be accessed as one would a key-value store , or a map. The keys can be any hashable object, but PyTorch supports integer keys out of the box. In general we would access an element of this dataset simply by indexing the `dataset` object like so, `dataset[idx]` to obtain the `idx`th object.

   > To create a **map-style** dataset we extend the abstract class `torch.utils.data.Dataset` and implement the `__get_item__()` function and optionally the `__len__()` function.

2. **iterable-style**: This type of dataset is best used in situations where random reads are expensive or the notion of a batch is somewhat ambiguous or undefined. Some typical scenarios are listed below.

   - This type of dataset is especially impactful in situations where we're reading from a source of data that is itself stochastic (in Deep RL, this is fairly common).
   - where random reads are expensive (reading from HDFS) or
   - a large file that can't be loaded into memory (huge text file).
   - a large file that is still being written to (like a log file).

> To create an **iterable-style** dataset, we extend the abstract class `torch.utils.data.IterableDataset` and implement the `__iter__()` function that represents an iterable over data samples. We explicitly make use of the `yield` keyword in python to ensure this is the case.

## Batch-able Datasets

Batch-able datasets are the most common and are easily handled by PyTorch. As noted above, **map-style** datasets are meant to handle this kind of data.

> To create a **map-style** dataset we extend the abstract class `torch.utils.data.Dataset` and implement the `__get_item__()` function and optionally the `__len__()` function.

## Default Dataloaders

PyTorch has a few useful defaults that can speed up your implementation.

### [`torch.utils.data.TensorDataset(*tensors)`](https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset)

Let each data point be a tensor of arbitrary dimensions `(d_1, d_2, ..., d_k)`. A dataset formed by creating a tensor of `N` such samples of dimension `(N, d_1, d_2, ..., d_k)` can be can be loaded using this wrapper. Accessing the $i$th sample through the `TensorDataset` object is done using `tensor_dataset[i - 1]` (since tensors use 0 based indexing).

### [`torch.utils.data.ConcatDataset(datasets)`](https://pytorch.org/docs/stable/data.html#torch.utils.data.ConcatDataset)

This creates a dataset by concatenating two datasets. This is very handy when we wish to leverage different datasources with the same type of data, differently organized.

### [`torchvision.datasets.DatasetFolder(root, loader, extensions)`](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.DatasetFolder)

A class that implements a generic dataset for samples arranged using the directory structure shown below.

```text
        root/
        class_1/
                sample_1.ext
                sample_2.ext
                .
                .
                .
        class_2/
                sample_1.ext
                sample_2.ext
                .
                .
```

## The `batch_size` argument and Automatic Batching

This tells the `dataloader` object how many samples to extract from the `dataset`.

> Passing a `batch_size=None` implies that we're dealing with a dataset for which Automatic Batching is disabled.
> PyTorch automatically creates batches using a default `collate_fn` of size `batch_size`.

## Automatic Batching with `collate_fn`

In a scenario when a `DataLoader` is provided with `batch_size`, `batch_sampler` or `shuffle=True` the `collate_fn` has the following default behavior.

1. Prepend a new dimension as a batch dimension. This is always the first dimension.
2. Convert Numpy arrays and Python numerics to PyTorch tensors.
3. Preserves the structure of the `dataset.__get_item__()` function. for eg. if this function returns a dictionary, the keys are the same but the values are batched tensors (or lists). The same applies to tuples, or lists.

> A custom `collate_fn` can be passed in if we wish to collate along a different dimension, padding sequences of various lengths, or adding support for special data types.

## The `sampler` and `batch_sampler` argument

The `sampler` argument provides a means of specifying which indices or sequence of indices need to be used to create batches. The samplers above work under the assumption that the sampler returns a single index into a dataset. We can use `torch.utils.data.BatchSampler(sampler, batch_size, drop_last)` to convert these into a batch sampler instead, which is returns a list or a sequential collection that can be used to index into a `Dataset`.

> The use of a `sampler` or `batch_sampler` and a `not None`, `batch_size` enables automatic batching. This triggers the default behavior of `collate_fn` when `collate_fn=None` is passed. Moreover, if the indices into your dataset are non-integer, it is mandatory to implement a `BatchSampler` or `Sampler`.

The difference between a `sampler` and `batch_sampler` is that a `sampler` is expected to provide a **single key** and a `batch_sampler` is expected to provide `batch_size` number of keys.

On passing `shuffle=True` to the `DataLoader`, a sampler that shuffles the dataset every epoch is automatically constructed.

> A sampler can be created by extending `torch.utils.data.Sampler` and overriding the `__iter__()` function to yield a set (or single) indices. Optionally the `__len__()` function can be overriden.

PyTorch provides some interesting default samplers that can help speed up our implementation.

1. `torch.utils.data.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)`. The `shuffle=True` creates a RandomSampler object implicitly.
2. `torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)`. Here `weights` is a `double` tensor that provides a sample weight for each sample in the dataset.

Read this [discussion](https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2) on the PyTorch forums for a useful guide to implement weighted sampling on Datasets.
