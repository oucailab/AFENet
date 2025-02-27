import torch
from thop import profile
from thop import clever_format

from AFENet.model.AFENet import AFENet

if __name__ == '__main__':
    input = torch.rand(1, 3, 512, 512)
    model = AFENet(decode_channels=64, num_classes=6, pretrained=True)
    output = model(input)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops:', flops)
    print('params:', params)
