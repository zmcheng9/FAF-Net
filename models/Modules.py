import torch
import numpy as np
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch import nn

import math
import torch
import numpy as np
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import nn
from torch.nn import init


class cross_att(nn.Module):

    def __init__(self, num_heads, dim_Q, dim_K, dim_hid, Dropout=False, ln=False):  # 5, 512, 512, 500
        super(cross_att, self).__init__()
        self.dim_hid = dim_hid
        self.num_heads = num_heads
        self.dim_K = dim_K
        self.fc_q = nn.Linear(dim_Q, dim_hid)
        self.fc_k = nn.Linear(dim_K, dim_hid)
        self.fc_v = nn.Linear(dim_K, dim_hid)   # (512, 512*m)
        self.Dropout = Dropout
        self.do = nn.Dropout(p=0.2)
        if ln:
            self.ln0 = nn.LayerNorm(dim_hid)
            self.ln1 = nn.LayerNorm(dim_hid)
        self.fc_o = nn.Linear(dim_hid, dim_K)

        self.amp_res1 = nn.Sequential(nn.Conv2d(dim_K, dim_K, kernel_size=1, padding=0, bias=False))
        self.amp_res2 = nn.Sequential(nn.ReLU(),
                                      nn.Conv2d(dim_K, dim_K, kernel_size=3, padding=1, bias=False))
        self.amp_res3 = nn.Sequential(nn.ReLU(),
                                      nn.Conv2d(dim_K, num_heads, kernel_size=1, padding=0, bias=False),
                                      nn.ReLU())

        self.pha_res1 = nn.Sequential(nn.Conv2d(dim_K, dim_K, kernel_size=1, padding=0, bias=False))
        self.pha_res2 = nn.Sequential(nn.ReLU(),
                                      nn.Conv2d(dim_K, dim_K, kernel_size=3, padding=1, bias=False))
        self.pha_res3 = nn.Sequential(nn.ReLU(),
                                      nn.Conv2d(dim_K, num_heads, kernel_size=1, padding=0, bias=False),
                                      nn.ReLU())
        self.down1 = nn.Linear(4225, 65 * 33)
        self.down2 = nn.Linear(4225, 65 * 33)

    def forward(self, Q, K):

        bsz, h, w = K.shape[0], K.shape[2], K.shape[3]
        qrt_fts = K
        K = K.permute(0, 2, 3, 1)
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(K)

        Q = Q.view(bsz, -1, self.num_heads, self.dim_hid // self.num_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.num_heads, self.dim_hid // self.num_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.num_heads, self.dim_K).permute(0, 2, 1, 3)

        A = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.dim_K)

        # get Fliter
        amp_Filter = self.amp_res1(qrt_fts)
        amp_Filter = self.amp_res2(amp_Filter) + amp_Filter
        amp_Filter = self.amp_res3(amp_Filter).view(self.num_heads, -1)
        amp_Filter = self.down1(amp_Filter).view(self.num_heads, h, -1)
        pha_Filter = self.pha_res1(qrt_fts)
        pha_Filter = self.pha_res2(pha_Filter) + pha_Filter
        pha_Filter = self.pha_res3(pha_Filter).view(self.num_heads, -1)
        pha_Filter = self.down2(pha_Filter).view(self.num_heads, h, -1)

        A = A.view(self.num_heads, h, w)
        A = torch.fft.rfft2(A, dim=(-2, -1), norm='backward')
        amp_A = torch.abs(A)
        pha_A = torch.angle(A)

        # flitering
        amp_A = amp_Filter * amp_A
        pha_A = pha_Filter * pha_A
        A_real = amp_A * torch.cos(pha_A)
        A_imag = amp_A * torch.sin(pha_A)
        A = torch.complex(A_real, A_imag)

        A = torch.fft.irfft2(A, s=(h, w), dim=(-2, -1), norm='backward').view(bsz, self.num_heads, 1, -1)
        A = torch.softmax(A, dim=-1)

        if self.Dropout:
            A = self.do(A)
        x = torch.matmul(A, V)
        x = x.reshape(1, self.num_heads * self.dim_K)
        x = self.fc_o(x)  # [1, 512]

        return x
