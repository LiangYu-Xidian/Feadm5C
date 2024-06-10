import torch.nn as nn
import torch
import feature

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=stride, bias=False)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=256, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.conv1 = nn.Conv2d(in_channels=41, out_channels=self.in_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(64 * block.expansion + 68, num_classes)
            self.dropout1 = nn.Dropout(p=0.3)
            self.fc2 = nn.Linear(num_classes, 64)
            self.dropout2 = nn.Dropout(p=0.75)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 1)
            self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))
        return nn.Sequential(*layers)

    def forward(self, x, x2):
        temp = torch.concat((x2[:, 0, 0, :], x2[:, 0, 1, :],
                             x2[:, 0, 2, :], x2[:, 0, 3, :]), 1)
        x = self.conv1(x)
        x = self.layer1(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = torch.concat((x, temp), 1)
            x = self.fc1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            x = torch.relu(x)
            x = self.fc4(x)
            x = self.sigmoid(x)
            x = torch.squeeze(x, -1)
        return x


class ResNetBiLSTM(nn.Module):
    def __init__(self, block, blocks_num, input_size, hidden_size, layer_num, num_classes=64, include_top=True):
        super(ResNetBiLSTM, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.conv1 = nn.Conv2d(in_channels=41, out_channels=self.in_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.lstm = nn.LSTM(input_size, hidden_size, layer_num, batch_first=True, bidirectional=True)
        self.w_omiga = torch.randn(128, 2 * self.hidden_size, 1, requires_grad=True).to(device)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(self.in_channel * block.expansion, num_classes)
            self.dropout1 = nn.Dropout(p=0.2)
            self.fc2 = nn.Linear(num_classes + 2 * hidden_size, 128)
            self.dropout2 = nn.Dropout(p=0.25)
            self.fc3 = nn.Linear(128, 32)
            self.fc4 = nn.Linear(32, 1)
            self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.layer1(x1)
        h0 = torch.zeros(2 * self.layer_num, x2.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(2 * self.layer_num, x2.size(0), self.hidden_size).to(device)
        temp = torch.concat((x2[:, 0, 0, :], x2[:, 0, 1, :],
                             x2[:, 0, 2, :], x2[:, 0, 3, :]), 1)
        temp = torch.reshape(temp, (-1, 68, 1))
        out, _ = self.lstm(temp, (h0, c0))
        H = torch.nn.Tanh()(out)
        weights = torch.nn.Softmax(dim=-1)(torch.bmm(H, self.w_omiga).squeeze()) \
            .unsqueeze(dim=-1).repeat(1, 1, self.hidden_size * 2)
        out = torch.mul(out, weights).sum(dim=-2)
        if self.include_top:
            x1 = self.avgpool(x1)
            x1 = torch.flatten(x1, 1)
            x1 = self.fc1(x1)
            x = torch.concat((x1, out), 1)
            x = torch.relu(x)
            x = self.fc2(x)
            x = torch.relu(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            x = torch.relu(x)
            x = self.fc4(x)
            x = self.sigmoid(x)
            x = torch.squeeze(x, -1)
        return x


def resnet(num_classes=256, include_top=True):
    return ResNet(BasicBlock, [2], num_classes=num_classes, include_top=include_top)


class CNN(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(CNN, self).__init__()
        self.in_channel = 32
        self.conv1 = nn.Conv2d(in_channels=41, out_channels=self.in_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32 + 68, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x2):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        temp = torch.concat((x2[:, 0, 0, :], x2[:, 0, 1, :],
                             x2[:, 0, 2, :], x2[:, 0, 3, :]), 1)
        x = torch.concat((x, temp), 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = torch.squeeze(x, -1)
        return x


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num, class_num=1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.lstm = nn.LSTM(input_size, hidden_size, layer_num, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_size + 68, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, class_num)
        self.dropout = nn.Dropout(p=0.3)
        self.w_omiga = torch.randn(128, 2 * self.hidden_size, 1, requires_grad=True).to(device)

    def forward(self, x1, x2):
        h0 = torch.zeros(2 * self.layer_num, x1.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(2 * self.layer_num, x1.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x1, (h0, c0))

        H = torch.nn.Tanh()(out)
        weights = torch.nn.Softmax(dim=-1)(torch.bmm(H, self.w_omiga).squeeze()) \
            .unsqueeze(dim=-1).repeat(1, 1, self.hidden_size * 2)
        out = torch.mul(out, weights).sum(dim=-2)
        temp = torch.concat((x2[:, 0, 0, :], x2[:, 0, 1, :],
                             x2[:, 0, 2, :], x2[:, 0, 3, :]), 1)
        x = torch.concat((out, temp), 1)
        out = self.fc1(x)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = self.fc3(out)
        out = torch.sigmoid(out)
        out = torch.squeeze(out, -1)
        return x, out


class kmerBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num, class_num=1):
        super(kmerBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.lstm = nn.LSTM(input_size, hidden_size, layer_num, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_size, 64)
        self.fc2 = nn.Linear(64, class_num)

    def forward(self, x1, x2):
        temp = torch.concat((x2[:, 0, 0, :], x2[:, 0, 1, :],
                             x2[:, 0, 2, :], x2[:, 0, 3, :]), 1)
        h0 = torch.zeros(2 * self.layer_num, x1.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(2 * self.layer_num, x1.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x1, (h0, c0))
        out = out[:, -1, :]
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        out = torch.squeeze(out, -1)
        return out


class MLP(nn.Module):
    def __init__(self, input_size=41 * 11 * 17, hidden_size=64, class_num=1):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, class_num)

    def forward(self, x, x2):
        out = torch.flatten(x, 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = torch.sigmoid(out)
        out = torch.squeeze(out, -1)
        return out
