import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

# Initialization model
def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0, stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    temp = torch.nn.PixelShuffle(33)(temp)
    return temp


class ConvLSTM(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel):

        super().__init__()
        pad_x = 1
        self.conv_xf = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x)
        self.conv_xi = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x)
        self.conv_xo = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x)
        self.conv_xj = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x)

        pad_h = 1
        self.conv_hf = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hi = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_ho = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hj = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)

    def forward(self, x, h, c):

        if h is None and c is None:
            i = F.sigmoid(self.conv_xi(x))
            o = F.sigmoid(self.conv_xo(x))
            j = F.tanh(self.conv_xj(x))
            c = i * j
            h = o * c
        else:
            f = F.sigmoid(self.conv_xf(x) + self.conv_hf(h))
            i = F.sigmoid(self.conv_xi(x) + self.conv_hi(h))
            o = F.sigmoid(self.conv_xo(x) + self.conv_ho(h))
            j = F.tanh(self.conv_xj(x) + self.conv_hj(h))
            c = f * c + i * j
            h = o * F.tanh(c)

        return h, h, c


# Define RB
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        res = x
        x = res + input
        return x


# Define MADUN Stage
class BasicBlock(torch.nn.Module):
    def __init__(self, args):
        super(BasicBlock, self).__init__()
        channels = args.channels
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(channels, channels + 1, 3, 3)))
        self.RB1 = ResidualBlock(channels, channels, 3, bias=True)
        self.RB2 = ResidualBlock(channels, channels, 3, bias=True)
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, channels, 3, 3)))
        self.ConvLSTM = ConvLSTM(channels, channels, 3)

    def forward(self, x, z, PhiWeight, PhiTWeight, PhiTb, h, c):
        x = x - self.lambda_step * PhiTPhi_fun(x, PhiWeight, PhiTWeight)
        x_input = x + self.lambda_step * PhiTb
        x_a = torch.cat([x_input, z], 1)
        x_D = F.conv2d(x_a, self.conv_D, padding=1)
        x = self.RB1(x_D)
        x, h, c = self.ConvLSTM(x, h, c)
        x_backward = self.RB2(x)
        x_G = F.conv2d(x_backward, self.conv_G, padding=1)
        x_pred = x_input + x_G

        return x_pred, x_backward, h, c


# Define MADUN
class MADUN(torch.nn.Module):
    def __init__(self, LayerNo, args, n_input, n_output):
        super(MADUN, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.n_input = n_input
        self.n_output = n_output

        for i in range(LayerNo):
            onelayer.append(BasicBlock(args))

        self.fcs = nn.ModuleList(onelayer)
        self.fe = nn.Conv2d(1, args.channels, 3, padding=1, bias=True)

    def forward(self, Phix, Phi):

        PhiWeight = Phi.contiguous().view(self.n_input, 1, 33, 33)
        PhiTWeight = Phi.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None)  # 64*1089*3*3
        PhiTb = torch.nn.PixelShuffle(33)(PhiTb)
        x = PhiTb
        [h, c] = [None, None]
        z = self.fe(x)

        for i in range(self.LayerNo):
            x, z, h, c = self.fcs[i](x, z, PhiWeight, PhiTWeight, PhiTb, h, c)

        x_final = x

        return x_final