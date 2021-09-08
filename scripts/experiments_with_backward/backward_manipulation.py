import numpy as np
import torch
from ml.modules.losses import MaskedL2Loss, MaskedL1Loss


class GradDrop(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        self.save_for_backward(x)

        return x

    @staticmethod
    def backward(self, grad_output):
        l = 0.1
        batches = [1, 2]

        grad_norm = torch.norm(grad_output, dim=[-1, -2])

        x_sign = torch.sign(self.saved_tensors[0])
        g_i = x_sign * grad_output

        g_i_batches = torch.split(g_i, batches, dim=0)

        g_is = []
        for gi_batch in g_i_batches:
            g_is.append(torch.sum(gi_batch, dim=0))

        sum_g_is = torch.stack(g_is, dim=0)

        p = 1 / 2 * (1 + sum_g_is / torch.norm(sum_g_is, dim=0))
        p[p != p] = 0

        random_threshold = torch.normal(0.5, 0.5, size=[1])

        full_mask = (p > random_threshold).float() * (sum_g_is > 0).float() + \
                    (p < random_threshold).float() * (sum_g_is < 0).float()

        split_grads = torch.split(grad_output, batches, dim=0)
        split_grad_list = []
        for i, split_grad in enumerate(split_grads):
            split_grad_list.append(split_grad * full_mask[i])

        grad = torch.cat(split_grad_list, dim=0)

        return grad


class Conv2DLayer(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel=3):
        super(Conv2DLayer, self).__init__()
        self.kernel = kernel
        self.wParam = torch.nn.Parameter(torch.from_numpy(np.ones([in_channels, out_channels, kernel, kernel],
                                                                  dtype=np.float32)))
        self.fct = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=1, bias=False)
        self.fct.weight = self.wParam

        self.gdParam = torch.nn.Parameter(torch.from_numpy(np.array(2, dtype=np.float32)))

        self.graddrop = GradDrop().apply

        self.loss1 = MaskedL1Loss()
        self.loss2 = MaskedL2Loss()

    def forward(self, layerIn, target):
        x = self.fct(layerIn)
        x = self.graddrop(x)

        loss = (self.loss2(x, target))

        return x, loss


x = torch.autograd.Variable(torch.from_numpy(np.array([[[[-1, 3, -5, 2],
                                                         [-3, 5, 5, -1],
                                                         [-1, -2, -3, -4]]],
                                                       [[[-2, 3, 4, 5],
                                                         [2, 4, 3, 5],
                                                         [5, 6, 9, 0]]],
                                                       [[[-1, 2, 3, 4],
                                                         [3, -4, 3, 5],
                                                         [2, 3, 4, 5]]]], dtype=np.float32)), requires_grad=True)
gt = torch.from_numpy(np.array([[[[2, 0, 0, 0],
                                  [0, 4, 0, 0],
                                  [0, -2, 0, 0]]],
                                [[[-50, 0, 0, 5],
                                  [0, 0, 0, 8],
                                  [0, 6, 0, 0]]],
                                [[[10, 0, 0, 3],
                                  [0, -3, 3, 2],
                                  [0, 0, 4, 0]]]], dtype=np.float32))
# wParam = torch.nn.Parameter(torch.from_numpy(np.array([[-1, -2, 3], [4, -5, 6]], dtype=np.float32)))
# fct = GradDrop(3, 2, bias=False)
# fct.weight = wParam

fct = Conv2DLayer()

print("forward...")
y, loss = fct(x, gt)
fct.zero_grad()
print("backward...")
# loss.backward(torch.ones_like(loss))
loss.backward()

# test = gradcheck(fct, (x, gt), eps=1e-6, atol=1e-4)

print(fct.wParam.grad.data)
print(x.grad.data)
