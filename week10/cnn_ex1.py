#!/usr/bin/env python
# coding: utf-8

## 라이브러리 임포트
# • PyTorch 및 torchvision: 딥러닝 모델, 데이터셋, 이미지 변환에 사용
# • matplotlib, seaborn: 시각화(그래프, 이미지, 혼동행렬)
# • numpy: 수치 연산
# • sklearn: 혼동행렬 계산
# • os: 폴더 생성 등 파일 시스템 제어
# • heapq: 모델 성능 기준으로 힙(우선순위 큐) 관리
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import heapq


## 환경 설정 및 하이퍼파라미터
# • GPU 사용 가능 여부 확인 후, DEVICE에 저장
# • 학습 epoch, 배치 크기, 최대 모델 수 설정
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
EPOCHS     = 10
BATCH_SIZE = 64
MAX_MODEL = 5 # 보관할 최대 모델 수


## 모델/플롯 저장 폴더 생성
# • 모델과 그래프 이미지를 저장할 폴더가 없으면 새로 생성
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('plots'):
    os.makedirs('plots')


## 학습/테스트 기록용 리스트 및 모델 힙
# • 학습 손실, 테스트 손실, 테스트 정확도 기록
# • 모델 성능(에러율) 기준으로 힙에 저장
train_losses = []
test_losses = []
test_accuracies = []
model_heap = []


## 데이터 샘플 시각화 함수
# • 배치에서 이미지를 추출해 16개 샘픙르 2x8 그리드로 시각화
def show_samples(loader,num_samples=16):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    fig = plt.figure(figsize=(16, 4))
    for i in range(num_samples):
        ax = fig.add_subplot(2, 8, i + 1)
        ax.imshow(images[i].squeeze().numpy(), cmap='gray')
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('plots/sample_images.png')
    plt.show()


## 학습 곡선 시각화 함수
# • 학습/테스트 손실, 테스트 정확도 변화 곡선을 1 행 2 열로 시각화
def plot_learning_curves():
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.tight_layout()
    plt.savefig('plots/learning_curves.png')
    plt.show()


## 혼동 행렬 시각화 함수
# • 테스트셋 전체 예측값과 실제값으로 혼동행렬을 계산, 시각화
def plot_confusion_matrix(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('plots/confusion_matrix.png')
    plt.show()


## 데이터셋 로딩 및 전처리
# • MNIST 데이터셋을 불러오고, 텐서 변환 및 정규화 적용
#• DataLoader 로 배치 단위로 로딩, 셔플
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./.data', train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./.data', train=False, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True)


## 데이터셋 정보 출력
# • 데이터셋 크기와 배치 개수 출력
print(f"Training dataset size: {len(train_loader.dataset)}")
print(f"Test dataset size: {len(test_loader.dataset)}")
print(f"Training batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")



## CNN 모델 클래스 정의
# • 3 개의 컨볼루션+배치정규화+ReLU+MaxPool 블록
# • 2 개의 완전연결(FC) 레이어, 드롭아웃 적용
# • 마지막 출력은 10 개 클래스(0~9)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 3 * 3)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


## 모델, 옵티마이저 선언
# • 모델을 선언하고, DEVICE(GPU/CPU)로 이동
# • Adam 옵티마이저 사용, 학습률 0.001
model = Net().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)


## 학습 함수 정의
# • 각 배치마다 순전파→손실계산→역전파→가중치갱신
# • 일정 간격마다 진행상황 출력
# • epoch 별 평균 손실과 정확도 기록
def train(model, train_loader, optimizer, epoch):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        epoch_loss += loss.item()
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    # epoch이 끝난 후 한 번만 계산
    epoch_loss = epoch_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    train_losses.append(epoch_loss)
    print(f'Training Accuracy: {train_accuracy:.2f}%')


## 평가 함수 정의
# • 평가 시에는 gradient 계산 비활성화
# • 전체 손실, 정확도 계산하여 반환
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * correct / len(test_loader.dataset)
        test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    return test_loss, test_accuracy


## 학습 및 평가 루프
# • 학습 전 샘플 이미지 출력
# • 각 epoch 마다 학습, 평가, 성능 정보(에러율 기준) 저장
# • 성능 좋은 모델 5 개만 힙에 유지, 파일로 저장
print("Showing sample images from the dataset:")
show_samples(train_loader, num_samples=16)

for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
        epoch, test_loss, test_accuracy))
    error_rate = 100 - test_accuracy
    model_info = {
        'epoch': epoch,
        'error_rate': error_rate,
        'accuracy': test_accuracy,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    heapq.heappush(model_heap, (error_rate, model_info))
    while len(model_heap) > MAX_MODEL:
        heapq.heappop(model_heap)
    print("\n 현재 저장된 모델들의 에러율:")
    for err_rate, info in sorted(model_heap):
        print(f"Epoch {info['epoch']}: 에러율 {err_rate:.2f}%, 정확도 {info['accuracy']:.2f}%")
for i, (err_rate, info) in enumerate(sorted(model_heap)):
    torch.save(info, f'models/mnist_cnn_top{i+1}_epoch_{info["epoch"]}.pt')


## 학습 곡선 및 혼동 행렬 출력
# • 학습/테스트 곡선, 혼동 행렬 시각화 및 저장
print("\nPlotting learning curves...")
plot_learning_curves()
print("\nGenerating confusion matrix...")
plot_confusion_matrix(model, test_loader)


## 최종 저장 모델 정보 출력
# • 성능 상위 5 개 모델의 epoch, 에러율, 정확도 출력

print("\n 최종 저장된 모델 정보:")
for i, (err_rate, info) in enumerate(sorted(model_heap)):
    print(f"Top {i+1} - Epoch {info['epoch']}: 에러율 {err_rate:.2f}%, 정확도 {info['accuracy']:.2f}%")