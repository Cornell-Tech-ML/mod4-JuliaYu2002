from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw
    first_view = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    permutation = first_view.permute(0, 1, 2, 4, 3, 5).contiguous()
    re_view = permutation.view(batch, channel, new_height, new_width, kh * kw)
    return (re_view, new_height, new_width)


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, smaller: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D"""
    new_tensor, new_height, new_width = tile(input, smaller)
    batch, channel, height, width, _ = new_tensor.shape
    meaned = new_tensor.mean(4)
    return meaned.view(batch, channel, height, width)

# TODO: 4.4

def argmax():
    pass

class Max(Function):
    pass

def max():
    pass

def softmax(input: Tensor, dim: int) -> Tensor:
    pass

def logsoftmax(input: Tensor, dim: int) -> Tensor:
    pass

def maxpool2d(input: Tensor, smaller: tuple[int, int]) -> Tensor:
    pass

def dropout(input: Tensor, dim: float, ignore: bool = False) -> Tensor:
    pass