import numpy as np
import torch
from torch.autograd import Function
from torch.nn import functional as F
from torch.nn.grad import _grad_input_padding
from torch.nn.modules.utils import _pair


class Conv2DLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(Conv2DLayer, self).__init__()
        self.kernel = kernel
        self.weight = torch.nn.Parameter(torch.from_numpy(np.ones([in_channels, out_channels, kernel, kernel],
                                                                  dtype=np.float32)))
        self.conv = Conv2D().apply

        # self.gdParam = torch.nn.Parameter(torch.from_numpy(np.array(2, dtype=np.float32)))

        self.loss = torch.nn.MSELoss(reduction='sum')

    def forward(self, layerIn, target):
        x = self.conv(layerIn, self.weight)

        loss = self.loss(x, target)

        return x, loss


class Conv2D(Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return F.conv2d(x, w, padding=1)

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_variables
        x_grad = w_grad = None
        if ctx.needs_input_grad[0]:
            x_grad = conv2d_input(x.shape, w, grad_output, padding=1)
        if ctx.needs_input_grad[1]:
            weight_size = w.shape
            w_grad = conv2d_weight(x, weight_size, grad_output, padding=1)
            # graddrop
            in_channels = x.shape[1]
            out_channels = grad_output.shape[1]
            groups = 1

            w_grad.sum(dim=0).view(in_channels // groups, out_channels, w_grad.shape[2], w_grad.shape[3]) \
                .transpose(0, 1).narrow(2, 0, weight_size[2]).narrow(3, 0, weight_size[3])

        return x_grad, w_grad


def conv2d_input(input_size, weight, grad_output, stride=1, padding=0, dilation=1, groups=1):
    r"""
    Computes the gradient of conv2d with respect to the input of the convolution.
    This is same as the 2D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.
    Args:
        input_size : Shape of the input gradient tensor
        weight: weight tensor (out_channels x in_channels/groups x kH x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
    Examples::
        >>> input = torch.randn(1,1,3,3, requires_grad=True)
        >>> weight = torch.randn(1,1,1,2, requires_grad=True)
        >>> output = F.conv2d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv2d_input(input.shape, weight, grad_output)
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    kernel_size = (weight.shape[2], weight.shape[3])

    if input_size is None:
        raise ValueError("grad.conv2d_input requires specifying an input_size")

    grad_input_padding = _grad_input_padding(grad_output, input_size, stride,
                                             padding, kernel_size, dilation)

    return torch.conv_transpose2d(grad_output, weight, None, stride, padding, grad_input_padding, groups, dilation)


def conv2d_weight(input, weight_size, grad_output, stride=1, padding=0, dilation=1, groups=1):
    r"""
    Computes the gradient of conv2d with respect to the weight of the convolution.
    Args:
        input: input tensor of shape (minibatch x in_channels x iH x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
    Examples::
        >>> input = torch.randn(1,1,3,3, requires_grad=True)
        >>> weight = torch.randn(1,1,1,2, requires_grad=True)
        >>> output = F.conv2d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_weight = torch.autograd.grad(output, filter, grad_output)
        >>> F.grad.conv2d_weight(input, weight.shape, grad_output)
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    in_channels = input.shape[1]
    out_channels = grad_output.shape[1]
    min_batch = input.shape[0]

    grad_output = grad_output.contiguous().repeat(1, in_channels // groups, 1,
                                                  1)
    grad_output = grad_output.contiguous().view(
        grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2],
        grad_output.shape[3])

    input = input.contiguous().view(1, input.shape[0] * input.shape[1],
                                    input.shape[2], input.shape[3])

    grad_weight = torch.conv2d(input, grad_output, None, dilation, padding,
                               stride, in_channels * min_batch)

    grad_weight = grad_weight.contiguous().view(min_batch,
                                                grad_weight.shape[1] // min_batch,
                                                grad_weight.shape[2],
                                                grad_weight.shape[3])

    return grad_weight


x = torch.autograd.Variable(torch.from_numpy(np.array([[[[-1, 3, -5, 2],
                                                         [-3, 5, 5, -1],
                                                         [-1, -2, -3, -4]]],
                                                       [[[2, 3, 4, 5],
                                                         [2, 4, 3, 5],
                                                         [5, 6, 9, 0]]],
                                                       [[[1, 2, 3, 4],
                                                         [1, -4, 3, 5],
                                                         [2, 3, 4, 5]]]], dtype=np.float32)), requires_grad=True)
gt = torch.from_numpy(np.array([[[[2, 3, 4, 5],
                                  [2, 4, 3, 5],
                                  [5, 6, 9, 0]]],
                                [[[2, 3, 4, 5],
                                  [2, 4, 3, 5],
                                  [5, 6, 9, 0]]],
                                [[[1, 2, 3, 4],
                                  [1, -4, 3, 5],
                                  [2, 3, 4, 5]]]], dtype=np.float32))
# wParam = torch.nn.Parameter(torch.from_numpy(np.array([[-1, -2, 3], [4, -5, 6]], dtype=np.float32)))
# fct = GradDrop(3, 2, bias=False)
# fct.weight = wParam

fct = Conv2DLayer(1, 1, 3)

print("forward...")
y, loss = fct(x, gt)
fct.zero_grad()
print("backward...")
loss.backward(torch.ones_like(loss))

# test = gradcheck(fct, (x, gt), eps=1e-6, atol=1e-4)

print(fct.wParam.grad.data)
print(x.grad.data)
