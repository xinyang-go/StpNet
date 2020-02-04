import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Conv3x3(nn.Conv2d):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__(in_channels, out_channels, 3, stride, 1, groups=groups)


class SpConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(SpConv3x3, self).__init__()
        self.conv3x1 = nn.Conv2d(in_channels, out_channels, (3, 1), stride, (1, 0), groups=groups)
        self.conv1x3 = nn.Conv2d(in_channels, out_channels, (1, 3), stride, (0, 1), groups=groups)

    def forward(self, x):
        return self.conv1x3(x) + self.conv3x1(x)


class DwConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, expand=2, conv3x3=SpConv3x3,
                 ext_hidden=True, bn=True, relu=True):
        super(DwConv2d, self).__init__()
        mid_channels = ((int(in_channels * expand) - 1) | 7) + 1 if ext_hidden else in_channels * expand
        if mid_channels != in_channels:
            self.add_module("1x1_ex", nn.Conv2d(in_channels, mid_channels, 1, 1))
            if bn:
                self.add_module("bn_ex", nn.BatchNorm2d(mid_channels))
            self.add_module("rl_ex", nn.LeakyReLU(inplace=True))
        self.add_module("3x3_dw", conv3x3(mid_channels, mid_channels, stride, groups=mid_channels))
        self.add_module("1x1_pw", nn.Conv2d(mid_channels, out_channels, 1, 1))
        if bn:
            self.add_module("bn_pw", nn.BatchNorm2d(out_channels))
        if relu:
            self.add_module("rl_ex", nn.LeakyReLU(inplace=True))


class StpBlock(nn.Module):
    def __init__(self, in_channel, expands=(2, 2)):
        super(StpBlock, self).__init__()
        self.convs = nn.ModuleList([DwConv2d(in_channel, in_channel, expand=e) for e in expands])

    def forward(self, x):
        for i in range(len(self.convs)):
            v = x.clone()
            for j in range(i + 1):
                v += self.convs[j](x)
            x = v
        return x


class StpNet(nn.Sequential):
    def __init__(self, config, num_classes=1000):
        super(StpNet, self).__init__()
        self.add_module("conv0", DwConv2d(3, config[0], 2, expand=1, conv3x3=Conv3x3, ext_hidden=False))
        last_channel = config[0]
        for i, cfg in enumerate(config[1:]):
            seq = nn.Sequential()
            if cfg[2]:
                seq.add_module("downsample", DwConv2d(last_channel, cfg[0], 2, conv3x3=Conv3x3))
            elif last_channel != cfg[0]:
                seq.add_module("conv", nn.Conv2d(last_channel, cfg[0], 1))

            seq.add_module(f"stp", StpBlock(cfg[0], cfg[1]))

            self.add_module(f"block{i}", seq)
            last_channel = cfg[0]
        self.add_module("pool&flatten", nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
        ))
        self.add_module("classifier", nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        ))


def create_stpnet(num_classes=1000):
    return StpNet([
        16,  # channels begin with
        # channels expands down
        [24, (1,), False],  # stride 2
        [32, (4, 4), True],  # stride 4
        [64, (4, 4, 4), True],  # stride 8
        [128, (2, 2, 2, 2), True],  # stride 16
        [256, (2, 2, 2, 2), True],  # stride 32
    ], num_classes=1000)
