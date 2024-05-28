import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, theta=0.7):
        super(CDC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, padding=0)
            return out_normal - self.theta * out_diff

class SFIA(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1,
                 padding=1, theta=0.7,pooling = False):
        super(SFIA, self).__init__()
        self.cdc = CDC(in_channels*2, in_channels*2, kernel_size, stride, padding, theta)
        self.bn = nn.BatchNorm2d(in_channels*2)
        self.pooling = pooling
        if not pooling:
            self.conv1x1 = nn.Conv2d(in_channels*2, 2, kernel_size=1,bias=False)
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def forward(self, spa, freq):
        # input: spatial and frequency features
        # output: spatial and frequency attention
        x = torch.cat([spa, freq], dim=1)
        x = F.relu(self.bn(self.cdc(x)))
        if self.pooling:
            x_a = torch.mean(x, dim=1, keepdim=True)
            x_m,_ = torch.max(x, dim=1, keepdim=True)
            x = torch.cat([x_a,x_m],dim=1)
        else:
            x = self.conv1x1(x)
        x = F.sigmoid(x)
        spa_att, freq_att = torch.chunk(x, 2, dim=1)
        return spa_att, freq_att
    
class ChannelAttention(nn.Module):
    def __init__(self, num_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // ratio, num_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.excitation(self.avg_pool(x))
        max_out = self.excitation(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SE(nn.Module):
    def __init__(self, inc=2048*2, ouc=2048, reduction_ratio=16):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.convblk = nn.Sequential(
            nn.Conv2d(inc, ouc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ouc),
            nn.ReLU()
        )
        self.excitation = nn.Sequential(
            nn.Linear(ouc,ouc//reduction_ratio,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ouc//reduction_ratio,ouc,bias=False),
            nn.Sigmoid()
        )

    def forward(self, spa, freq):
        input_tensor = self.convblk(torch.cat((spa, freq), dim=1))
        batch_size, num_channels, H, W = input_tensor.size()
        out = self.squeeze(input_tensor).view(batch_size,num_channels)
        out = self.excitation(out).view(batch_size,num_channels,1,1)
        out = input_tensor*out
        return out

class CAT(nn.Module):
    def __init__(self, inc=2048*2, ouc=2048):
        super(CAT, self).__init__()
        self.convblk = nn.Conv2d(inc, ouc, 1, 1, 0, bias=False)
    def forward(self, spa, freq):
        return self.convblk(torch.cat((spa, freq), dim=1))
    
class CMIA(nn.Module):
    def __init__(self, in_channels, dim):
        super(CMIA, self).__init__()
        # self.cdc = CDC(in_channels*2, in_channels)
        self.cdc = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1) # use 1x1 conv
        self.conv_spa_v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.conv_freq_v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.ln = nn.LayerNorm(dim)
        self.relu = nn.ReLU(inplace=True)

        self.to_qk = nn.Linear(dim, dim * 2, bias = False)
        self.spa_fc = nn.Linear(dim, dim)
        self.freq_fc = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = dim ** -0.5

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x_spa, x_freq):
        b,c,h,w = x_spa.size()
        v_spa, v_freq = self.conv_spa_v(x_spa), self.conv_freq_v(x_freq)
        x = torch.cat([x_spa, x_freq], dim=1)
        x = self.ln(rearrange(self.cdc(x), 'b c h w -> b c (h w)'))

        q, k = self.to_qk(x).chunk(2, dim = -1)
        att = torch.matmul(q.transpose(-1, -2), k) * self.scale
        att_spa = self.softmax(self.spa_fc(att))
        att_freq = self.softmax(self.freq_fc(att))
        
        atted_spa = rearrange(torch.matmul(v_spa.flatten(2), att_spa), 'b c (h w) -> b c h w', h=h, w=w)
        atted_freq = rearrange(torch.matmul(v_freq.flatten(2), att_freq), 'b c (h w) -> b c h w', h=h, w=w)
        
        out_spa = x_spa + atted_spa
        out_freq = x_freq + atted_freq
        return out_spa, out_freq


if __name__ == '__main__':
    # test code
    spa = torch.randn(2,6,224,224)
    freq = torch.randn(2,6,224,224)
    sfia = SFIA(in_channels=6,pooling=True)
    spa_att, freq_att = sfia(spa, freq)
    print(spa_att.shape, freq_att.shape)
    print(spa_att)
    print(freq_att)
    print(spa_att.sum(dim=1))
    print(freq_att.sum(dim=1))