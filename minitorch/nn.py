from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


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

max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """step(x) = x > 0 = argmax{0, x} :: get index of larger item"""
    # if x > 0 -> step returns 1, else 0 :: step function
    # return a tensor that has 1 in the places where it found max values
    return max_reduce(input, dim) == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward for max"""
        a = max_reduce(input, int(dim.item()))
        ctx.save_for_backward(input, int(dim.item()))
        return a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[Tensor, float]:
        """Backward for max"""
        # compute argmax -> send gradient to argmax gradinput, everything else is 0
        input, dim = ctx.saved_values
        a = argmax(input, dim)
        return a * d_output, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """User's call to max"""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """exp(x) / sum of exp(x)s"""
    expd = input.exp()
    normalize = expd.sum(dim)
    return expd / normalize


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Use https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations"""
    # the github test seems to fail but it passes locally?
    exp = input.exp()
    sum = exp.sum(dim)
    log = sum.log()
    return input - log


def maxpool2d(input: Tensor, smaller: Tuple[int, int]) -> Tensor:
    """Same thing as avgpool2d, but using max"""
    new_tensor, new_height, new_width = tile(input, smaller)
    batch, channel, height, width, _ = new_tensor.shape
    maxed = max(new_tensor, 4)
    return maxed.view(batch, channel, height, width)


def dropout(input: Tensor, drop_chance: float, ignore: bool = False) -> Tensor:
    """Add chance of data fluctuation"""
    if not ignore:
        random_t = rand(input.shape)
        random_drops = random_t > drop_chance
        return input * random_drops
    else:
        return input
