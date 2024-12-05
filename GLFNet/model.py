import torch
import torch.nn as nn
import torch.nn.functional as F



def Conv3x3BNReLU(in_channels, out_channels, stride, groups):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                  groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


def Conv1x1BNReLU(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )



def Conv1x1BN(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels)
    )

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = (in_channels * expansion_factor)

        self.bottleneck = nn.Sequential(

            Conv1x1BNReLU(in_channels, mid_channels),

            Conv3x3BNReLU(mid_channels, mid_channels, stride, groups=mid_channels),

            Conv1x1BN(mid_channels, out_channels)
        )

        self.shortcut = Conv1x1BN(in_channels, out_channels)


    def forward(self, x):
        out = self.bottleneck(x)
        out = (out + self.shortcut(x)) if self.stride == 1 else out

        return out

class SEBlock(nn.Module):
    def __init__(self, channel, r=5):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale
        y = torch.mul(x, y)
        return y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class YYJOK_gao(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=1, stride=1, padding=0)
        self.mn = InvertedResidual(in_channels=10, out_channels=10, expansion_factor=2, stride=1)
        self.se = SEBlock(10)
        self.In = nn.InstanceNorm2d(10)
        self.sa = SpatialAttention()

    def forward(self, x):
        x_out_2 = self.conv1(x)
        x_out_2 = self.In(x_out_2)
        x_out_2 = F.elu(x_out_2)

        x_out_3 = self.mn(x_out_2)
        x_out_3 = x_out_3 + x_out_2

        x_out_4_1 = self.sa(x_out_2) * x_out_2
        x_out_4_1 = x_out_4_1 + x_out_2
        x_out_4_2 = self.se(x_out_4_1)
        x_out_4_2 = x_out_4_2 + x_out_4_1

        return x_out_4_2, x_out_3

class YYJOK_nose(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=1, stride=1, padding=0)
        self.mn = InvertedResidual(in_channels=10, out_channels=10, expansion_factor=4, stride=1)
        self.se = SEBlock(10)
        self.sa = SpatialAttention()
        self.In = nn.InstanceNorm2d(10)

    def forward(self, x):
        x_out_2 = self.conv1(x)
        x_out_2 = self.In(x_out_2)
        x_out_2 = F.elu(x_out_2)

        x_out_3 = self.mn(x_out_2)
        x_out_3 = x_out_3 + x_out_2

        x_out_4_1 = self.sa(x_out_2) * x_out_2
        x_out_4_1 = x_out_4_1 + x_out_2
        x_out_4_2 = self.se(x_out_4_1)
        x_out_4_2 = x_out_4_2 + x_out_4_1

        return x_out_4_2, x_out_3

class YYJOK(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(1, 2), stride=(1, 2), padding=0)
        self.mn = InvertedResidual(in_channels=10, out_channels=10, expansion_factor=2, stride=1)

        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)

        self.cf1 = nn.Conv1d(1, 1, kernel_size=5, stride=3)
        self.cf2 = nn.Conv1d(1, 1, kernel_size=5, stride=3)
        self.fc1 = nn.Linear(576, 6)
        self.gao = YYJOK_gao()
        self.nose = YYJOK_nose()
        self.bn = nn.BatchNorm2d(10)
        self.se = SEBlock(10)
        self.sa = SpatialAttention()
        self.xishu1 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        self.xishu2 = torch.nn.Parameter(torch.Tensor([0.5]))  # 1 - lamda
        self.xishu3 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        self.xishu4 = torch.nn.Parameter(torch.Tensor([0.5]))  # 1 - lamda
        self.xishu5 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        self.xishu6 = torch.nn.Parameter(torch.Tensor([0.5]))  # 1 - lamda

        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x_input_size = x.size(0)
        x_out_1 = x[:, :, :, :52]
        y_out = x[:, :, :, 52:142]

        x_tf_1, x_mo_1 = self.gao(x_out_1)
        y_tf_1, y_mo_1 = self.nose(y_out)

        m2 = x_tf_1 * self.xishu1 + y_tf_1 * self.xishu2
        m1 = x_mo_1 * self.xishu3 + y_mo_1 * self.xishu4

        m3 = self.mn(m2)
        m3 = m3 + m2

        m32 = self.sa(m1) * m1
        m32 = m32 + m1
        m4 = self.se(m32)
        m4 = m32 + m4
        m = m4 * self.xishu5 + m3 * self.xishu6

        n = m.reshape(x_input_size, -1)
        n = n.reshape(-1, 1, 5200)

        x = self.cf1(n)
        x = self.cf2(x)
        x = x.reshape(-1, 576)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.drop(x)
        output = F.log_softmax(x, dim=1)

        return output


best_accuracy = 0.0
def accuracy(output, target):
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == target).sum().item()
    total = target.size(0)
    return correct / total
def train_network(model,train_loader,test_loader, LR,pth):
    global best_accuracy_1,best_accuracy_2,best_accuracy_3
    model.train()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # optimize all cnn parameters
    total_accuracy_train = 0.0  # 初始化总训练准确率
    for batch_idx, ( data1, target) in enumerate(train_loader):
        b_x1, b_y = data1.cuda(),  target.cuda()
        output= model(b_x1)  # fake_img:64*(144*16*16) output:64*15    # cnn output
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, b_y)
        print(output.size())
        print(b_y.size())
        accuracy_train = accuracy(output, b_y)
        loss.backward()
        optimizer.step()
        total_accuracy_train += accuracy_train  # 累积每个批次的准确率
    average_accuracy_train = total_accuracy_train / len(train_loader)
    if pth == 1:

       if 50 < average_accuracy_train * 100 < 100:
           if average_accuracy_train * 100 > best_accuracy:
                best_accuracy_3 = average_accuracy_train * 100
                torch.save(model.state_dict(), 'train.pth')

    print('Average Training Accuracy: {:.2f}%'.format(average_accuracy_train * 100))

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for  data2, target in test_loader:
            data_nose, target =  data2.cuda(), target.cuda()
            out= model( data_nose)
            out = F.log_softmax(out, dim=1)
            test_loss += F.nll_loss(out, target, reduction='sum')  # 将一批的损失相加
            pred = out.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            target_cpu = target.cpu().numpy() if target.is_cuda else target.numpy()
            pred_cpu = pred.cpu().numpy() if pred.is_cuda else pred.numpy()

    test_loss /= len(test_loader.dataset)  # 平均损失
    test_loss = test_loss.item()
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))

    return 1. * test_loss / len(test_loader.dataset), 100. * correct / len(test_loader.dataset), pred_cpu, target_cpu



