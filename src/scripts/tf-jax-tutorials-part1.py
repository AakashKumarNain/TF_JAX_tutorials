import numpy as np
import tensorflow as tf

seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)


# Tensors

"""
What is a `Tensor` anyway?<br>
Although the meaning of `Tensor` is much diverse than what we typically use in
ML, whenever we say `tensor` in ML, we mean that it is a **`multi-dimensional array`**
where all the values have a uniform `dtype`. There are many ways to create a TF
tensor. We will take a look at a few of them, a few important ones.
`tf.constant(..)`: This is the simplest way yet with some `gotchas` to create a
tensor object. First, let's try to create a tensor with it, and then we will
look at the gotchas later on.
"""


# A zero rank tensor. A zero rank tensor is nothing but a single value
x = tf.constant(5.0)
print(x)


"""As you can see above, that the tensor object has a `shape` and a `dtype`.
There are other attributes/properties as well that are associated with
a tensor object:
1. Shape: The length (number of elements) of each of the axes of a tensor.
2. Rank: Number of axes. For example, a matrix is a tensor of rank 2.
3. Axis or Dimension: A particular dimension of a tensor.
4. Size: The total number of items in the tensor.
"""


# We can convert any tensor object to `ndarray` by calling the `numpy()` method
y = tf.constant([1, 2, 3], dtype=tf.int8).numpy()
print(f"`y` is now a {type(y)} object and have a value == {y}")


"""A few important things along with some gotchas:

1. People confuse `tf.constant(..)` with an operation that creates a `constant`
tensor. There is no such relation. This is related to how we embed a node
in a `tf.Graph`

2. Any tensor in TensorFlow is **immutable** by default i.e. you cannot change
the values of a tensor once created. You always create a new one. This is different
from `numpy` and `pytorch` where you can actually modify the values. We will see
an example on this in a bit

3. One of the closest member to `tf.constant` is the `tf.convert_to_tensor()`
method with a few difference which we will see later on

4. `tf.constant(..)` is just one of the many ways to create a tensor.
There are many other methods as well
"""

# Immutability check

# Rank-1 tensor
x = tf.constant([1, 2], dtype=tf.int8)
# Try to modify the values
try:
    x[1] = 3
except Exception as ex:
    print(type(ex).__name__, ex)

# tf.constant(..) is no special. Let's create a tensor using a diff method
x = tf.ones(2, dtype=tf.int8)
print(x)

try:
    x[0] = 3
except Exception as ex:
    print("\n", type(ex).__name__, ex)


# Check all the properties of a tensor object
print(f"Shape of x : {x.shape}")
print(f"Another method to obtain the shape using `tf.shape(..)`: {tf.shape(x)}")

print(f"\nRank of the tensor: {x.ndim}")
print(f"dtype of the tensor: {x.dtype}")
print(f"Total size of the tensor: {tf.size(x)}")
print(f"Values of the tensor: {x.numpy()}")


"""Not able to do assignment in Tensor objects is a bit (more than bit TBH) frustrating.
What's the solution then?<br>
The best way that I have figured out, that has always worked for my use case is to
create a mask or to use [tf.tensor_scatter_nd_update](https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update). Let's look at an example.<br>

Original tensor -> `[1, 2, 3, 4, 5]` <br>
Output tensor we want -> `[1, 200, 3, 400, 5]`<br>
"""

# Create a tensor first. Here is another way
x = tf.cast([1, 2, 3, 4, 5], dtype=tf.float32)
print("Original tensor: ", x)

mask = x % 2 == 0
print("Original mask: ", mask)

mask = tf.cast(mask, dtype=x.dtype)
print("Mask casted to original tensor type: ", mask)

# Some kind of operation on an tensor that is of same size
# or broadcastable to the original tensor. Here we will simply
# use the range object to create that tensor
temp = tf.cast(tf.range(1, 6) * 100, dtype=x.dtype)

# Output tensor
# Input tensor -> [1, 2, 3, 4, 5]
# Mask -> [0, 1, 0, 1, 0]
out = x * (1 - mask) + mask * temp
print("Output tensor: ", out)

# Another way to achieve the same thing
indices_to_update = tf.where(x % 2 == 0)
print("Indices to update: ", indices_to_update)

# Update the tensor values
updates = [200.0, 400.0]
out = tf.tensor_scatter_nd_update(x, indices_to_update, updates)
print("\nOutput tensor")
print(out)

# This works!
arr = np.random.randint(5, size=(5,), dtype=np.int32)
print("Numpy array: ", arr)
print(
    "Accessing numpy array elements based on a  condition with irregular strides",
    arr[[1, 4]],
)

# This doesn't work
try:
    print(
        "Accessing tensor elements based on a  condition with irregular strides",
        x[[1, 4]],
    )
except Exception as ex:
    print(type(ex).__name__, ex)


"""What now? If you want to extract multiple elements from a tensor with irregular
strides, or not so well defined strides, then [tf.gather](https://www.tensorflow.org/api_docs/python/tf/gather)
and [tf.gather_nd](https://www.tensorflow.org/api_docs/python/tf/gather_nd) are
your friends. Let;s try it again!"""


print("Original tensor: ", x.numpy())
# Using the indices that we used for mask
print("\nIndices to update: ", indices_to_update.numpy())

# This works!
print("\n Accesing tensor elements using gather")
print("\n", tf.gather(x, indices_to_update).numpy())


"""There is another method `tf.convert_to_tensor(..)` to create a tensor. This is
very similar to `tf.constant(..)` but with a few subtle differences:
1. Whenever you pass a non tf.Tensor object like a Python list or a ndarray to
an op, `convert_to_tensor(..)` is always called automatically
2. It doesn't take `shape` as an input argument.
3. It allows to pass even `symbolic tensors`. We will take a look at it in a bit.

When to use `tf.convert_to_tensor(..)`? It's up to your mental model!
"""

#  An example with a python list
y = tf.convert_to_tensor([1, 2, 3])
print("Tensor from python list: ", y)

#  An example with a ndarray
y = tf.convert_to_tensor(np.array([1, 2, 3]))
print("Tensor from ndarray: ", y)

#  An example with symbolic tensors
with tf.compat.v1.Graph().as_default():
    y = tf.convert_to_tensor(
        tf.compat.v1.placeholder(shape=[None, None, None], dtype=tf.int32)
    )
print("Tensor from python list: ", y)


# ### Other kind of Tensor objects available

# String as a tensor object with dtype==tf.string
string = tf.constant("abc", dtype=tf.string)
print("String tensor: ", string)

# String tensors are atomic and non-indexable.
# This doen't work as expected!
print("\nAccessing second element of the string")
try:
    print(string[1])
except Exception as ex:
    print(type(ex).__name__, ex)


# #### Ragged tensors
# In short, a tensor with variable numbers of elements along some axis.

# This works!
y = [[1, 2, 3], [4, 5], [6]]

ragged = tf.ragged.constant(y)
print("Creating ragged tensor from python sequence: ", ragged)

# This won't work
print("Trying to create tensor from above python sequence\n")
try:
    z = tf.constant(y)
except Exception as ex:
    print(type(ex).__name__, ex)


# #### Sparse tensors

"""Let's say you have a an array like this one
[[1 0 0]
 [0 2 0]
 [0 0 3]]
If there are too many zeros in your `huge` tensor, then it is wise to use `sparse`
tensors instead of `dense` one. Let's say how to create this one. We need to specify:
1. Indices where our values are
2. The values
3. The actual shape
"""

sparse_tensor = tf.SparseTensor(
    indices=[[0, 0], [1, 1], [2, 2]], values=[1, 2, 3], dense_shape=[3, 3]
)
print(sparse_tensor)

# You can convert sparse tensors to dense as well
print("\n", tf.sparse.to_dense(sparse_tensor))
