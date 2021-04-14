# PyTorch Tensor Manual

## Tensors

Tensors are a sequential multidimensional array. To get started fire up a python interpreter and begin with `import torch`.

- To create a tensor of ones of size $n=10$ enter, `a = torch.ones(10)`.
- To create a tensor of a single number enter, `p = torch.tensor(4.0)`.
- To check the type of a tensor enter, `p.dtype`.
- Torch can also create a tensor from a list, `v = torch.tensor([1,2,3,4,5.])`
- To create a tensor with zeros. `pts = torch.zeros(6)` creates a tensor with 6 zeros.

Notice the $\color{red}{.}$ at the end instead of $5$ without a period. This will make sure that the tensor is of type `float32`. The whole tensor should be of a uniform type, so the integers in the beginning will also be **upcast** to `float32`.

> PyTorch tensors (or numpy arrays) are views over contiguous memory blocks that contain [**unboxed**](https://stackoverflow.com/questions/13055/what-is-boxing-and-unboxing-and-what-are-the-trade-offs) C numeric types rather than python objects.

Tensors thus require a fixed amount of memory per item stored and a small amount of overhead to store dimensions and the data type associated with the tensor.

### Tensors of more than 1 dimension

We can also create a 2-D tensor from a list of lists.

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
```

- Here we can access the first row by indices starting as zero like so, `points[0]`.
- We can also ask the tensor about its shape by leveraging the `.shape` attribute like so, `points.shape`.
- If we wish to create an $m\times n$ tensor of ones or zeros, we can specify the dimensions as arguments as shown,
  `a = torch.ones(m, n)` and `b = torch.zeros(m, n)`.
- Here, we need to use two indices to access a single element.
- The code `points[i, j]` accesses the $i,j$th element in the `points` tensor.
- We can access a single row of the data set by using a single index.
- `points[i]` returns a new 1D tensor of the $i$th element. **This is a view of the underlying data.**
  > A new chunk of memory was not allocated. Only pointers were manipulated to gain access to this view.

### Indexing Tensors

To access rows and columns we can use slicing as in `numpy`.

For the `points` data we can access all the x-coords by using `:` like so, `points[:, 0]`.
Here we obtain all the rows of the first column.

Similarly we can also obtain all the rows between `2` to `8` like so,`points[2:8, :]`.

We can also add a dimension where required by doing `points[None]`. This is similar to `np.newaxis` in numpy.

> In PyTorch since many dimensions are present in common use cases, it is important to ensure that the order and meaning of these dimensions are tracked.

### Broadcasting

A smaller tensor can be broadcast over a larger tensor while performing an operation. For example, consider a tensor of shape `(3, 5, 5)` which can be through of as 3, 2-D arrays of shape `(5, 5)`. We wish to multiply each `(5, 5)` tensor with three numbers stored in a tensor of shape (3)`. We can't multiply these shapes directly, but instead must align the smaller tensor to the larger one by following these rules.

#### Broadcasting Rules

To ensure that two tensors are "broadcastable" start from the trailing dimension for both tensors.

Every dimension from the trailing dimension must

1. Either be equal.
2. One of them must be 1.
3. If one of the tensors has a smaller dimension the last dimension that lines up with that of the larger tensor must be equal or 1.

In order to multiply the tensors of shape `(3, 5, 5)` and `(3)` we must add dimensions to the end of `(3)` using `unsqueeze` or using `None`. This is demonstrated in the snippet below.

```python
a = torch.randn(3,5,5)
b = torch.randn(3)
# multiplies each 5*5 array in a by one value in b
a * b[:, None, None] # or b.unsqueeze(-1).unsqueeze(-1), yes thats two .unsqueeze(-1)'s
```

## Tensor element types

Using standard python data types are not going to cut it. Python makes use of the OOP paradigm for even fundamental objects like integers and floating point numbers. This means using tensors in PyTorch or numpy arrays is going to yield a massive speedup because we're not keeping track of unwanted attributes that add to the overhead.

Python stores numbers as objects with reference counting (for garbage collection) and so on. This can't scale to millions of numbers.

Lists in pythons are meant for sequential collections of objects: Objects can be very diverse in size and type. This makes allocating space to store them contiguously for faster access difficult. This means that python stores them non-contiguously. This makes it very inefficient to do dot products or to map functions on to arrays.

Python is interpreted and not compiled. For each command in python, we pay the price of converting into machine code. This is slower than C or C++ which is compiled into fast machine code.

PyTorch prefers the compiled paradigm to implement tensors and arrays which makes these problems go away.

PyTorch also prefers homohenous arrays i.e. arrays that contain elements of the same type.

The only information PyTorch needs is data type being stored and how many such values to determine the required space needed.

### Numeric Data Types in Pytorch

#### Floating Point Types

`torch.float32` or `torch.float`

`torch.float64` or `torch.double`

`torch.float16` or `torch.half`

#### Integer Types

`torch.int8`

`torch.uint8`

`torch.int16` or `torch.short`

`torch.int32` or `torch.int`

`torch.int64` or `torch.long`

#### Boolean Types

`torch.bool`

#### General Info about types

> `torch.float16` or `torch.half` are available mostly on GPU's and are not native to modern(intel) CPU's. Can be used to reduce the footprint of a model for a minor reduction in accuracy.
> `torch.bool` tensors are produced when predicates are applied to tensors, such as `a > 0`. The `torch.where(condition, x, y)` makes use of these `bool` tensors for the `condition` parameter as illustrated below.

```python
a = torch.tensor(range(10),
                 dtype=torch.float32).reshape(2,5)
a
## Out
# tensor([[0, 1, 2, 3, 4],
#        [5, 6, 7, 8, 9]])
b = torch.ones(2, 5) * 100

 torch.where(a > 4., a, b)
## Out
# tensor([[100., 100., 100., 100., 100.],
#        [  5.,   6.,   7.,   8.,   9.]])
```

### Managing a Tensors Type using `dtype`

We can manipulate a tensors data type using the `dtype` attribute or argument.

If we wish to construct a tensor from a list, we can pass in the `dtype` argument as seen below.

`t = torch.tensor([[1, 2, 3, 4]], dtype=torch.short)`

We can use this to construct a tensor using any of the other standard functions used to construct constant tensors as well.

```python
t = torch.ones(5, 5, dtype=torch.double)
s = torch.zeros(5, 5, dtype=torch.uint8)
```

We can also use casting methods as seen below.

```python
d = torch.ones(4, 2).double()
a = torch.zero(1,2).short()
```

The more handy `.to()` is also plausible. This is useful because it can do more than just change the type.

```python
d = torch.ones(4, 2).to(dtype=torch.double)
```

> From PyTorch 1.3 onwards, when mixing types, the lower type is cast to the higher one. For eg. when multiplying a float tensor and a short tensor, the result will be a float tensor.

## Tensor Ops

Refer to this cheatsheet for a full list of operations: <https://pytorch.org/tutorials/beginner/ptcheat.html>.

## Tensor Indexing and Storage

Three attributes of a tensor define how it is stored in memory. Size, offset and stride.

Common ways of storing and allocating contiguous blocks of multidimensional memory are the following:

> **Row Major Order**: Memory is allocated as contiguous rows.
> **Column Major Order**: Memory is allocated as contiguous columns.
> _This determines the order of the indices considered when iterating over elements of the array. As demonstrated [here](https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays) the order plays a huge role in the efficiency with which the entire array is traversed._

### Size

Tuple that indicates how many elements are present across each dimension.

### Offset

Indicates how many storage blocks need to be skipped to obtain a particular element.

### Stride

Tuple that indicates the the number of storage elements that need to be skipped to obtain the next element along each dimension.

Examples:

Consider the 3-D Array below:

```python
 # Choosing small dims
rands = torch.randn(2, 3, 4)
rands.storage()

''' Out
-0.2829638719558716
 -1.1649523973464966
 1.3321882486343384
 1.3248190879821777
 -0.3670113682746887
 1.4062683582305908
 -1.4656304121017456
 -0.1982385367155075
 0.659064531326294
 0.6156697869300842
 1.385794758796692
 -0.8989276885986328
 -0.7524341940879822
 0.6367485523223877
 1.1766417026519775
 -0.19177700579166412
 -0.02886893041431904
 1.017943263053894
 -1.3301843404769897
 0.005868543870747089
 -0.21604008972644806
 0.825073778629303
 -0.7951344847679138
 0.4247194528579712
[torch.FloatStorage of size 24]
'''
```

This output indicates that we have allocated a contiguous block of 24 elements.

To look at what the strides for this tensor look like, we use the `stride()` function.

```python
rands.stride()

## Out
# (12, 4, 1)
```

This means that the offset of the second 3x4 matrix in this array of random numbers is 12 as indicated by,

```python
rands[1].storage_offset()
## Out
# 12
rands[1]

''' Out
tensor([[-0.7524,  0.6367,  1.1766, -0.1918],
        [-0.0289,  1.0179, -1.3302,  0.0059],
        [-0.2160,  0.8251, -0.7951,  0.4247]])
'''
rands[1].stride()
# Out
# (4, 1)
```

Also notice that the strides of the first 2-D array indicates that the tensor is stored as contiguous rows which is row major order.

### Sub-Tensors are views

If we choose to reassign the first element of our tensor from the previous example we are left with a view into the original tensors storage. Thus re-assigning any element in this view will alter the original storage as well as demonstrated below.

```python
t = rands[1]

t[2, 3] = 888

rands
'''
Out:
tensor([[[-2.8296e-01, -1.1650e+00,  1.3322e+00,  1.3248e+00],
         [-3.6701e-01,  1.4063e+00, -1.4656e+00, -1.9824e-01],
         [ 6.5906e-01,  6.1567e-01,  1.3858e+00, -8.9893e-01]],

        [[-7.5243e-01,  6.3675e-01,  1.1766e+00, -1.9178e-01],
         [-2.8869e-02,  1.0179e+00, -1.3302e+00,  5.8685e-03],
         [-2.1604e-01,  8.2507e-01, -7.9513e-01,  8.8800e+02]]])
'''
```

This may not be desirable thus we can clone the original tensor which allocates a new and different memory block. We can do this by using, `python rands2 = rands.clone()`

## N-dimensional Indexing for Row Major arrays

Suppose we have an n-dimensional array of dimensions $d_1, \cdots, d_n$.

To access an individual element we need to provide an index of $n$ numbers given as $a_1, \cdots, a_n$.

The offset of this element can be computed as follows
$offset = a_n + d_n (a_{n-1} + d_{n-1}(a_{n-2} + d_{n-2}(\cdots)))=a_n + \sum_{i=1}^{n - 1}\lbrack\Pi_{j=i+1}^n d_j \rbrack a_i$

For 3-D arrays this generalizes to:

$offset = a_3 + a_2 \cdot  d_3 + a_1 (d_2 \cdot d_3)$ for an element with an index $(a_1, a_2, a_3)$.

and `offset = a_3 + a_2 * stride[1] + a_1 * stride[0] * stride[1]` in PyTorch.

### PyTorch mixes Row Major and Column Major indexing

This is done to avoid unnecessary allocation of memory.

Example:

If we create a 2-D array and transposing it, using `.transpose()` or `.t()` as a shorthand PyTorch simply creates a new tensor object with the strides reversed.

```python
In [35]: rand_2d = torch.rand(2,5)

In [36]: rand_2d
'''
Out[36]:
tensor([[0.1995, 0.3417, 0.2173, 0.1970, 0.3338],
        [0.7093, 0.3297, 0.5090, 0.3352, 0.6757]])
'''
rand_2d.stride()

# Out[37]: (5, 1)

In [38]: rand_2d_t = rand_2d.t()

In [39]: rand_2d_t.stride()
# Out [40]: (1, 5)
In [40]: rand_2d_t[1,1]=9009

In [41]: rand_2d

'''
Out[41]:
tensor([[1.9947e-01, 3.4166e-01, 2.1729e-01, 1.9695e-01, 3.3377e-01],
        [7.0934e-01, 9.0090e+03, 5.0896e-01, 3.3519e-01, 6.7570e-01]])
'''
```

> From a PyTorch perspective, the transposed array in the above example is **not contiguous**. The reasoning being that the convention in PyTorch is to consider row major order arrays as contiguous.

### Making an Array Contiguous

Certain operations are only permitted on contiguous arrays in PyTorch. It is important to convert an array to be contiguous if it is not using the `.contiguous()`function on an array.

If the array is already contiguous, this operation does nothing else it allocates a new memory block that is contiguous and reassigns the tensor to it.

## Moving Tensors to the GPU

Tensors in PyTorch can live on the GPU as well.

GPU's use the data local processing paradigm.

They are massively parallel integrated circuit's that accelerate the processing of data especially for operations like dot products and convolutions that are common to Deep Learning.

GPU's are associated with a large amount of RAM that is specific to the GPU and is not shared by the CPU.

Furthermore this RAM is equipped with several processing units that operate on segments of memory local to a particular block of RAM.

Operations that need to occur on a specific range of indices within tensors are executed in parallel local to the memory regions they map to.

We can assign a tensor to a GPU in PyTorch by using the `.to(device="cuda")` function on any tensor.

While creating a tensor, we can also use any tensor creation method or function and specify the device as seen in this example `t = torch.ones(50, device="cuda")`

On devices with more than 1 GPU we can specify which device the tensor is allocated to by using the following convention. `t = torch.ones(50, device="cuda:0")`.

GPU's are zero indexed thus if we have two GPU's we can allocate a tensor on the second GPU by doing `t = torch.randn(3, 5, 5, device="cuda:1")`

This form of addressing GPU's also works with the `to` function as seen, `t = torch.ones(1,2,3).to(device="cuda:0")` function.

To transfer a tensor back to the CPU, we can do simply say `device="cpu"` in any using the `.to` attribute.

Tensors by default are allocated to the CPU using constructors.

### Transferring Numpy arrays to PyTorch

We can use the `t = torch.from_numpy(np.arange(10))` constructor to turn a numpy array into a tensor. This will preserve the dimensions and shape as in numpy.

We can also use the `t.numpy()` function on a tensor to turn it back into a numpy array.

> Numpy arrays are by default `float64`. In PyTorch we generally use `float32` converting numpy arrays to `float32` using the `.to(dtype=torch.float32)` function is generally a good idea.

### Saving and loading Tensors

We can use the `torch.save(tensor, "tensor.t")` function.

We can also load the tensor using `torch.load(tensor, "tensor.t")`

Resoures: Deep Learning with PyTorch, Eli Stevens, Luca Antiga, Thomas Viehman
