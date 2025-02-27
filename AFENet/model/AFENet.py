import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

import numbers

from einops import rearrange

from AFENet.model.AFSIM import AFSIModule


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class ReLUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), )

    def forward(self, x):
        return self.op(x)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LocalBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalBranch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

        # Branch 1: 3x1 convolution with BN and ReLU
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 51 convolution with BN and ReLU
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (5, 1), padding=(2, 0)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # Depthwise 3x3 convolution
        self.dwconv3x3 = nn.Conv2d(in_channels, out_channels, (3, 3), padding=1, groups=out_channels)

    def forward(self, x):
        # Pass input through each branch
        x = self.conv1(x)
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = out1 + out2
        # Apply depthwise convolution
        out = self.dwconv3x3(out)
        return out + x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


# Spatial Conv Module
class SCModule(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        return attn1, attn2


# Selective feature Fusion Module
class SFFModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv3 = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, high_feature, low_feature, x):
        out = torch.cat([high_feature, low_feature], dim=1)
        avg_attn = torch.mean(out, dim=1, keepdim=True)

        max_attn, _ = torch.max(out, dim=1, keepdim=True)

        agg = torch.cat([avg_attn, max_attn], dim=1)

        sig = self.conv_squeeze(agg)

        sig = sig.sigmoid()

        out = high_feature * sig[:, 0, :, :].unsqueeze(1) + low_feature * sig[:, 1, :, :].unsqueeze(1)
        out = self.conv3(out)
        result = x * out

        return result


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


## Adaptive Frequency Enhancement Block (AFEB)
class AFEBlock(nn.Module):
    """
    AFEBlock integrates Adaptive Frequency and Spatial feature Interaction Module (AFSIM)
    with Spatial Conv Module (SCM) and Selective Feature Fusion Module (SFFModule) to enhance
    feature representation by combining high and low frequency features.
    """
    def __init__(self, dim, num_heads, bias, in_dim=3):
        super(AFEBlock, self).__init__()
        self.AFSIM = AFSIModule(dim, num_heads, bias, in_dim)
        self.SCM = SCModule(dim)
        self.fusion = SFFModule(dim)
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.proj_2 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()

    def forward(self, image, x):
        _, _, H, W = x.size()
        image = F.interpolate(image, (H, W), mode='bilinear')
        shortcut = x.clone()

        x = self.proj_1(x)
        x = self.activation(x)
        s_high, s_low = self.SCM(x)

        high_feature, low_feature = self.AFSIM(s_high, s_low, image, x)
        out = self.fusion(high_feature, low_feature, x)

        result = self.proj_2(out)

        result = shortcut + result
        return result


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Decoder(nn.Module):
    """
    Decoder module that reconstructs the feature maps from the encoded features.
    It uses AFEBlocks to enhance features at different levels and TransformerBlocks
    for further feature refinement.
    """
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 decoder=True,
                 heads=[1, 2, 4, 8],
                 num_blocks=[4, 6, 6, 8],
                 LayerNorm_type='WithBias',
                 num_classes=6,
                 ):
        super(Decoder, self).__init__()
        self.decoder = decoder
        if self.decoder:
            self.AFEB1 = AFEBlock(decode_channels * 2 ** 3, num_heads=heads[2], bias=bias)
            self.AFEB2 = AFEBlock(decode_channels * 2 ** 2, num_heads=heads[2], bias=bias)
            self.AFEB3 = AFEBlock(decode_channels * 2 ** 1, num_heads=heads[2], bias=bias)

        self.up4 = nn.Conv2d(decode_channels * 2 ** 3, decode_channels * 2 ** 2, 1)
        self.up3 = nn.Conv2d(decode_channels * 2 ** 2, decode_channels * 2 ** 1, 1)
        self.up2 = nn.Conv2d(decode_channels * 2 ** 1, decode_channels * 2 ** 0, 1)

        self.reduce_level3 = nn.Conv2d(int(decode_channels * 2 ** 3), int(decode_channels * 2 ** 2), kernel_size=1,
                                       bias=bias)
        self.reduce_level2 = nn.Conv2d(int(decode_channels * 2 ** 2), int(decode_channels * 2 ** 1), kernel_size=1,
                                       bias=bias)
        self.reduce_level1 = nn.Conv2d(int(decode_channels * 2 ** 1), int(decode_channels * 2 ** 0), kernel_size=1,
                                       bias=bias)

        self.TB1 = nn.Sequential(*[
            TransformerBlock(dim=int(decode_channels * 2 ** 2), num_heads=heads[2],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.TB2 = nn.Sequential(*[
            TransformerBlock(dim=int(decode_channels * 2 ** 1), num_heads=heads[1],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.TB3 = nn.Sequential(*[
            TransformerBlock(dim=int(decode_channels * 1), num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, x, res1, res2, res3, res4, h, w):

        # AFEB1
        inp_dec_level4 = self.AFEB1(x, res4)
        inp_dec_level4 = self.up4(inp_dec_level4)
        inp_dec_level3 = F.interpolate(inp_dec_level4, scale_factor=2, mode='bilinear', align_corners=False)
        inp_dec_level3 = torch.cat([inp_dec_level3, res3], 1)
        inp_dec_level3 = self.reduce_level3(inp_dec_level3)
        # TB1
        inp_dec_level3 = self.TB1(inp_dec_level3)
        # AFEB2
        out_dec_level3 = self.AFEB2(x, inp_dec_level3)
        out_dec_level3 = self.up3(out_dec_level3)
        inp_dec_level2 = F.interpolate(out_dec_level3, scale_factor=2, mode='bilinear', align_corners=False)
        inp_dec_level2 = torch.cat([inp_dec_level2, res2], 1)
        inp_dec_level2 = self.reduce_level2(inp_dec_level2)
        # TB2
        inp_dec_level2 = self.TB2(inp_dec_level2)
        # AFEB3
        out_dec_level2 = self.AFEB3(x, inp_dec_level2)
        out_dec_level2 = self.up2(out_dec_level2)
        inp_dec_level1 = F.interpolate(out_dec_level2, scale_factor=2, mode='bilinear', align_corners=False)
        inp_dec_level1 = torch.cat([inp_dec_level1, res1], 1)
        inp_dec_level1 = self.reduce_level1(inp_dec_level1)
        # TB3
        out = self.TB3(inp_dec_level1)
        # out = self.fre4(x, inp_dec_level1)

        # out = self.decoder_level1(out)
        out = self.segmentation_head(out)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        return out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class AFENet(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='swsl_resnet18',
                 pretrained=True,
                 num_classes=6,
                 ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, num_classes=num_classes)

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        if self.training:
            x = self.decoder(x, res1, res2, res3, res4, h, w)
            return x
        else:
            x = self.decoder(x, res1, res2, res3, res4, h, w)
            return x


