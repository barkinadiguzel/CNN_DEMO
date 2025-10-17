class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d(in_channels, growth_rate, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.Sequential(*[
            DenseLayer(in_channels + i * growth_rate, growth_rate)
            for i in range(n_layers)
        ])

    def forward(self, x):
        return self.layers(x)

class DenseNetSmall(nn.Module):
    def __init__(self, num_classes=1000, growth_rate=32):
        super(DenseNetSmall, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.block1 = DenseBlock(64, growth_rate, 4)
        self.block2 = DenseBlock(64 + 4 * growth_rate, growth_rate, 4)
        self.fc = nn.Linear(64 + 8 * growth_rate, num_classes)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = torch.mean(x, [2, 3])
        return self.fc(x)
