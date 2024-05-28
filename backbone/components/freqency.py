# fine-grained frequency extraction
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# utils
def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

class BlockDCT(nn.Module):
    def __init__(self, block_size=8):
        super(BlockDCT, self).__init__()
        self.block_size = block_size
        self._DCT = nn.Parameter(torch.tensor(DCT_mat(block_size)).float(), requires_grad=False)
        self._DCT_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(block_size)).float(), 0, 1), requires_grad=False)

    def image_to_blocks(self, x):
        """
        将 tensor 分割为 block_size x block_size 的块。
        """
        B, C, H, W = x.shape
        unfold = F.unfold(x, kernel_size=self.block_size, stride=self.block_size) # (B, C*bsize*bsize, N,)
        blocks = unfold.permute(0,2,1).view(B, -1, C, self.block_size, self.block_size)  # (B, N, C, block_size, block_size)
        return blocks
    
    def blocks_to_image(self, blocks, output_size):
        """
        将多个 block_size x block_size 的块重建为原始 tensor。
        """
        # 调整形状并使用 fold 重建图像
        blocks_ = blocks.permute(0,2,3,4,1).contiguous().view(blocks.shape[0], -1, blocks.shape[1]) # (B, C*bsize*bsize, N)
        fold = F.fold(blocks_, output_size=output_size, kernel_size=self.block_size, stride=self.block_size) # (B, C, H, W)
        return fold

    def DCT_transform(self, blocks):
        """
        对块进行 DCT 变换。
        """
        x_freq = self._DCT @ blocks @ self._DCT_T
        return x_freq

    def forward(self, x):
        B,C,H,W = x.shape
        blocks = self.image_to_blocks(x)
        x_freq = self.DCT_transform(blocks)
        fold = self.blocks_to_image(x_freq, output_size=(H,W))
        return fold

# fine-grained multi-scale frequency module
class FMFM(nn.Module):
    def __init__(self, block_sizes=(8,12,16),in_channels=3,out_channels=6):
        super(FMFM, self).__init__()
        self.block_sizes = block_sizes
        for block_size in block_sizes:
            setattr(self, f'block_dct_{block_size}', BlockDCT(block_size))
        self.out_conv = nn.Conv2d(in_channels=in_channels*3, out_channels=out_channels, kernel_size=1, bias=False)
    def forward(self, x):
        out = []
        for block_size in self.block_sizes:
            out.append(getattr(self, f'block_dct_{block_size}')(x))
        out = torch.cat(out, dim=1)
        out = self.out_conv(out)
        return out
    
# texture enhancement module
class Conv_Filter(nn.Module):
    def __init__(self, kernel):
        super(Conv_Filter, self).__init__()
        kernel = np.stack([kernel, kernel, kernel], axis=0)
        kernel = np.expand_dims(kernel, axis=1)
        self.kernel = nn.Parameter(data=torch.from_numpy(kernel), requires_grad=False)
    def forward(self, x):
        out = F.conv2d(x, self.kernel, stride=1, padding=1, groups=3)
        return out

class TEM(nn.Module):
    def __init__(self,):
        super(TEM, self).__init__()
        kernel1 = np.array([[0, 0, 0],
                            [1, -1, 0],
                            [0, 0, 0]], dtype=np.float32)
        kernel2 = np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 0]], dtype=np.float32)
        kernel3 = np.array([[0, 1, 0],
                            [0, -1, 0],
                            [0, 0, 0]], dtype=np.float32)
        kernel4 = np.array([[0, 0, 1],
                            [0, -1, 0],
                            [0, 0, 0]], dtype=np.float32)
        self.conv_1 = Conv_Filter(kernel1)
        self.conv_2 = Conv_Filter(kernel2)
        self.conv_3 = Conv_Filter(kernel3)
        self.conv_4 = Conv_Filter(kernel4)

    def forward(self, x):
        out1 = self.conv_1(x).unsqueeze(1)
        out2 = self.conv_2(x).unsqueeze(1)
        out3 = self.conv_3(x).unsqueeze(1)
        out4 = self.conv_4(x).unsqueeze(1)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out_max = torch.max(out, dim=1)[0]
        out_min = torch.min(out, dim=1)[0]
        residual = out_max - out_min
        rgb_res = torch.cat([x, residual], dim=1)
        return rgb_res
    
class TEM_CONV(nn.Module):
    def __init__(self,outc=3):
        super(TEM_CONV, self).__init__()
        kernel1 = np.array([[0, 0, 0],
                            [1, -1, 0],
                            [0, 0, 0]], dtype=np.float32)
        kernel2 = np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 0]], dtype=np.float32)
        kernel3 = np.array([[0, 1, 0],
                            [0, -1, 0],
                            [0, 0, 0]], dtype=np.float32)
        kernel4 = np.array([[0, 0, 1],
                            [0, -1, 0],
                            [0, 0, 0]], dtype=np.float32)
        self.conv_1 = Conv_Filter(kernel1)
        self.conv_2 = Conv_Filter(kernel2)
        self.conv_3 = Conv_Filter(kernel3)
        self.conv_4 = Conv_Filter(kernel4)
        self.conv_1x1 = nn.Conv2d(in_channels=6, out_channels=outc, kernel_size=1, stride=1, padding=0, bias=False)
        self.act = nn.Hardtanh(min_val=-3.0, max_val=3.0)

    def forward(self, x):
        out1 = self.conv_1(x).unsqueeze(1)
        out2 = self.conv_2(x).unsqueeze(1)
        out3 = self.conv_3(x).unsqueeze(1)
        out4 = self.conv_4(x).unsqueeze(1)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out_max = torch.max(out, dim=1)[0]
        out_min = torch.min(out, dim=1)[0]
        residual = out_max - out_min
        rgb_res = torch.cat([x, residual], dim=1)
        rgb_res = self.conv_1x1(rgb_res)
        rgb_res = self.act(rgb_res)
        return rgb_res