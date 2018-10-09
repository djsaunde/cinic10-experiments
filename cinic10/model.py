import torch.nn as nn
import torch.nn.functional as F

from fastai.conv_learner import ConvLearner, num_cpus, accuracy

from .load import load_cinic10


def conv_2d(ni, nf, stride=1, ks=3):
    return nn.Conv2d(
        in_channels=ni, out_channels=nf, kernel_size=ks, stride=stride, padding=ks // 2, bias=False
    )


def bn_relu_conv(ni, nf):
    return nn.Sequential(
        nn.BatchNorm2d(ni), nn.ReLU(inplace=True), conv_2d(ni, nf)
    )


class BasicBlock(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(ni)
        self.conv1 = conv_2d(ni, nf, stride)
        self.conv2 = bn_relu_conv(nf, nf)
        self.shortcut = lambda x: x
        if ni != nf:
            self.shortcut = conv_2d(ni, nf, stride, 1)

    def forward(self, x):
        x = F.relu(self.bn(x), inplace=True)
        r = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x) * 0.2
        return x.add_(r)


def make_group(n, ni, nf, stride):
    start = BasicBlock(ni, nf, stride)
    rest = [BasicBlock(nf, nf)] * (n - 1)
    return [start] + rest


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class WideResNet(nn.Module):
    def __init__(self, n_groups, n, n_classes, k=1, n_start=16):
        super().__init__()
        # Increase channels to n_start using conv layer
        layers = [conv_2d(3, n_start)]
        n_channels = [n_start]

        # Add groups of BasicBlock(increase channels & downsample)
        for i in range(n_groups):
            n_channels.append(n_start * (2 ** i) * k)
            stride = 2 if i > 0 else 1
            layers += make_group(
                n, n_channels[i], n_channels[i + 1], stride
            )

        # Pool, flatten & add linear layer for classification
        layers += [
            nn.BatchNorm2d(n_channels[3]), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
            Flatten(), nn.Linear(n_channels[3], n_classes)
        ]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


def wrn_22():
    return WideResNet(n_groups=3, n=3, n_classes=10, k=6)


def get_learner(arch, batch_size=128):
    """Create a FastAI learner using the given model"""
    data = load_cinic10(batch_size=batch_size)
    learn = ConvLearner.from_model_data(arch.cuda(), data)
    learn.crit = nn.CrossEntropyLoss()
    learn.metrics = [accuracy]
    return learn


def get_TTA_accuracy(learn):
    """Calculate accuracy with Test Time Agumentation(TTA)"""
    preds, targs = learn.TTA()
    preds = 0.6 * preds[0] + 0.4 * preds[1:].sum(0)