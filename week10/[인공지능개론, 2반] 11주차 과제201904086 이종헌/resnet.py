#!/usr/bin/env python
# coding: utf-8

# 실행 환경과 인코딩 지정

# • PyTorch 및 torchvision의 주요 모듈을 임포트합니다.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models


# • CUDA(즉, GPU) 사용 가능 여부를 확인해, 연산 장치를 설정합니다.
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


# • 학습 epoch 수 (3회)와 배치 크기(128)를 지정합니다. (실제 실험시 300회로 늘릴 수 있음)
EPOCHS     = 3 #300
BATCH_SIZE = 128


# • CIFAR-10 학습 데이터를 불러오고, 데이터 증강(랜덤 크롭, 수평 뒤집기), 텐서 변환, 정규화 후 DataLoader로 만든다.
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./.data',
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                       transforms.RandomCrop(32, padding=4),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))])),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./.data',
                   train=False, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))])),
    batch_size=BATCH_SIZE, shuffle=True)


## ResNet 모델 정의
# • ResNet의 기본 블록(BasicBlock)을 정의합니다.
#    o 3x3 컨볼루션 → 배치정규화 → ReLU → 3x3 컨볼루션 → 배치정규화 → shortcut(잔차 연결) → ReLU
#    o 입력과 출력의 크기가 다르면 1x1 컨볼루션으로 shortcut을 맞춥니다.
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# • ResNet 전체 네트워크를 정의합니다.
#   o 입력(3 채널) → conv1 → layer1(16 채널, 2 블록) → layer2(32 채널, 2 블록, 다운샘플)
#       → layer3(64 채널, 2 블록, 다운샘플) → 평균풀링 → FC(10 클래스)
#   o _make_layer 는 지정한 블록 수만큼 BasicBlock 을 쌓습니다.
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


## 학습 준비
# • 모델을 생성하고, GPU/CPU 로 보냅니다.
# • SGD 옵티마이저와 러닝레이트 스케줄러를 설정합니다.
# • 모델 구조를 출력합니다
model = ResNet().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

print(model)


## 학습 및 평가 함수
# • 모델을 학습 모드로 전환하고, 각 배치마다
#   o 데이터를 장치로 이동
#   o 옵티마이저 초기화
#   o 순전파, 손실계산, 역전파, 가중치 업데이트를 수행합니다.
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

# • 평가 모드로 전환, 기울기 계산 비활성화
# • 전체 손실과 정확도를 계산해 반환합니다.
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # 배치 오차를 합산
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()

            # 가장 높은 값을 가진 인덱스가 바로 예측값
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


## 학습 및 평가 실행
# • 지정한 epoch 수만큼 학습과 평가를 반복합니다.
#   o epoch 마다 러닝레이트 스케줄러 갱신
#   o 학습 및 평가 후 결과를 출력합니다.
for epoch in range(1, EPOCHS + 1):
    scheduler.step()
    train(model, train_loader, optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, test_loader)
    
    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
          epoch, test_loss, test_accuracy))
