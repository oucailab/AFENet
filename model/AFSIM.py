import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from AFENet.model.Attention import CrossAttention


# Adaptive Frequency and Spatial feature Interaction Module(AFSIM)
class AFSIModule(nn.Module):

    def __init__(self, dim, num_heads, bias, in_dim=3):
        super(AFSIModule, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        rdim = self.get_reduction_dim(dim)

        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, rdim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(rdim, 2, 1, bias=False),
        )
        # Define learnable parameters for gating
        self.alpha_h = torch.nn.Parameter(torch.tensor(0.5))
        self.alpha_w = torch.nn.Parameter(torch.tensor(0.5))

        self.CA_low = CrossAttention(dim // 2, num_head=num_heads, bias=bias)
        self.CA_high = CrossAttention(dim // 2, num_head=num_heads, bias=bias)

        self.conv2_1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2_2 = nn.Conv2d(dim, dim // 2, 1)

    def forward(self, s_high, s_low, image, x):

        f_high, f_low = self.fft(image)

        f_high = self.conv2_1(f_high)
        f_low = self.conv2_2(f_low)

        high_feature = self.CA_low(f_high, s_high)
        low_feature = self.CA_high(f_low, s_low)

        return high_feature, low_feature

    def get_reduction_dim(self, dim):
        if dim < 8:  # 最小维度保护
            return max(2, dim)
        log_dim = math.log2(dim)
        reduction = max(2, int(dim // log_dim))
        return reduction

    def shift(self, x):
        """shift FFT feature map to center"""
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))

    def unshift(self, x):
        """converse to shift operation"""
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(-int(h / 2), -int(w / 2)), dims=(2, 3))

    def fft(self, x):
        """obtain high/low-frequency features from input"""
        x = self.conv1(x)
        mask = torch.zeros(x.shape).to(x.device)
        h, w = x.shape[-2:]
        threshold = F.adaptive_avg_pool2d(x, 1)

        threshold = self.rate_conv(threshold).sigmoid()

        # 这个阈值用于确定频谱中心的大小，即决定多大范围的频率被认为是低频。
        # 加入了两个可学习参数帮助确定h和w
        blended_threshold_h = self.alpha_h * threshold[:, 0, :, :] + (1 - self.alpha_h) * threshold[:, 1, :, :]
        blended_threshold_w = self.alpha_w * threshold[:, 0, :, :] + (1 - self.alpha_w) * threshold[:, 1, :, :]

        # Calculate the dimensions of the mask based on the blended thresholds
        for i in range(mask.shape[0]):
            h_ = (h // 2 * blended_threshold_h[i]).round().int()  # Convert to int after rounding
            w_ = (w // 2 * blended_threshold_w[i]).round().int()  # Convert to int after rounding

            # Apply the mask based on blended h and w
            mask[i, :, h // 2 - h_:h // 2 + h_, w // 2 - w_:w // 2 + w_] = 1

        # 对于mask的每个元素，根据阈值在频谱的中心位置创建一个正方形窗口，窗口内的值设为1，表示这部分是低频区域。
        fft = torch.fft.fft2(x, norm='forward', dim=(-2, -1))
        fft = self.shift(fft)
        # 对x执行FFT变换，得到频谱，并通过shift方法将低频分量移动到中心。
        fft_high = fft * (1 - mask)

        high = self.unshift(fft_high)
        high = torch.fft.ifft2(high, norm='forward', dim=(-2, -1))
        high = torch.abs(high)

        fft_low = fft * mask
        low = self.unshift(fft_low)
        low = torch.fft.ifft2(low, norm='forward', dim=(-2, -1))
        low = torch.abs(low)

        return high, low
