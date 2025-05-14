#!/usr/bin/env python
# coding: utf-8

# # 뉴럴넷으로 패션 아이템 구분하기
# Fashion MNIST 데이터셋과 앞서 배운 인공신경망을 이용하여 패션아이템을 구분해봅니다.

import torch                    # PyTorch 기본 라이브러리
import torch.nn as nn          # 신경망 모델 정의를 위한 모듈
import torch.optim as optim    # 최적화 알고리즘을 위한 모듈
import torch.nn.functional as F # 활성화 함수 등을 위한 모듈
from torchvision import transforms, datasets  # 데이터 변환 및 데이터셋 로드를 위한 모듈
import os                      # 파일 시스템 작업을 위한 모듈
import matplotlib.pyplot as plt # 그래프 시각화를 위한 모듈
import numpy as np             # 수치 연산을 위한 모듈
from sklearn.metrics import confusion_matrix  # 혼동 행렬 계산을 위한 모듈
import seaborn as sns          # 시각화를 위한 모듈

# GPU 사용 가능 여부 확인 및 디바이스 설정
USE_CUDA = torch.cuda.is_available()  # GPU 사용 가능 여부 확인
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")  # GPU 또는 CPU 선택


# 학습 파라미터 설정
EPOCHS = 30        # 전체 데이터셋 학습 횟수
BATCH_SIZE = 64    # 한 번에 처리할 데이터 수


# 모델 저장 경로 설정
MODEL_PATH = './model/fashion_mnist_model.pth'  # 학습된 모델을 저장할 경로



# ## 데이터셋 불러오기

# 데이터 전처리 설정
transform = transforms.Compose([
    transforms.ToTensor()  # 이미지를 텐서로 변환
])

# Fashion MNIST 데이터셋 로드
trainset = datasets.FashionMNIST(
    root      = './.data/',    # 데이터 저장 경로
    train     = True,          # 훈련 데이터셋 사용
    download  = True,          # 데이터 다운로드
    transform = transform      # 데이터 변환 적용
)

testset = datasets.FashionMNIST(
    root      = './.data/',    # 데이터 저장 경로
    train     = False,         # 테스트 데이터셋 사용
    download  = True,          # 데이터 다운로드
    transform = transform      # 데이터 변환 적용
)

# 데이터 로더 설정
train_loader = torch.utils.data.DataLoader(
    dataset     = trainset,    # 훈련 데이터셋
    batch_size  = BATCH_SIZE,  # 배치 크기
    shuffle     = True,        # 데이터 섞기
)

test_loader = torch.utils.data.DataLoader(
    dataset     = testset,     # 테스트 데이터셋
    batch_size  = BATCH_SIZE,  # 배치 크기
    shuffle     = True,        # 데이터 섞기
)


# ## 뉴럴넷으로 Fashion MNIST 학습하기
# 입력 `x` 는 `[배치크기, 색, 높이, 넓이]`로 이루어져 있습니다.
# `x.size()`를 해보면 `[64, 1, 28, 28]`이라고 표시되는 것을 보실 수 있습니다.
# Fashion MNIST에서 이미지의 크기는 28 x 28, 색은 흑백으로 1 가지 입니다.
# 그러므로 입력 x의 총 특성값 갯수는 28 x 28 x 1, 즉 784개 입니다.
# 우리가 사용할 모델은 3개의 레이어를 가진 인공신경망 입니다. 

# 신경망 모델 정의
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3개의 완전연결층 정의
        self.fc1 = nn.Linear(784, 256)  # 입력층 -> 은닉층1
        self.fc2 = nn.Linear(256, 128)  # 은닉층1 -> 은닉층2
        self.fc3 = nn.Linear(128, 10)   # 은닉층2 -> 출력층

    def forward(self, x):
        x = x.view(-1, 784)            # 입력 데이터 평탄화
        x = F.relu(self.fc1(x))        # 첫 번째 층 + ReLU 활성화
        x = F.relu(self.fc2(x))        # 두 번째 층 + ReLU 활성화
        x = self.fc3(x)                # 출력층
        return x


# ## 모델 준비하기
# `to()` 함수는 모델의 파라미터들을 지정한 곳으로 보내는 역할을 합니다.
# 일반적으로 CPU 1개만 사용할 경우 필요는 없지만,
# GPU를 사용하고자 하는 경우 `to("cuda")`로 지정하여 GPU로 보내야 합니다.
# 지정하지 않을 경우 계속 CPU에 남아 있게 되며 빠른 훈련의 이점을 누리실 수 없습니다.
# 최적화 알고리즘으로 파이토치에 내장되어 있는 `optim.SGD`를 사용하겠습니다.

# 모델 및 최적화 알고리즘 초기화
model        = Net().to(DEVICE)
optimizer    = optim.SGD(model.parameters(), lr=0.01)


# ## 학습하기

def train(model, train_loader, optimizer):
    model.train()  # 학습 모드 설정
    for batch_idx, (data, target) in enumerate(train_loader): # 배치 인덱스와 데이터, 타겟 가져오기
        data, target = data.to(DEVICE), target.to(DEVICE)  # 데이터를 GPU로 이동
        optimizer.zero_grad()  # 그래디언트 초기화
        output = model(data)   # 순전파
        loss = F.cross_entropy(output, target)  # 손실 계산
        loss.backward()        # 역전파
        optimizer.step()       # 파라미터 업데이트


# ## 테스트하기

def evaluate(model, test_loader):
    model.eval() # 평가 모드로 설정
    test_loss = 0
    correct = 0
    all_preds = [] # 예측값 저장
    all_targets = [] # 실제값 저장
    with torch.no_grad(): # 그래디언트 계산 비활성화
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE) 
            output = model(data) 
            
            test_loss += F.cross_entropy(output, target,
                                       reduction='sum').item()
            
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # 예측값과 실제값 저장
            all_preds.extend(pred.cpu().numpy()) # 예측값 저장
            all_targets.extend(target.cpu().numpy()) # 실제값 저장

    test_loss /= len(test_loader.dataset) # 평균 손실 계산
    test_accuracy = 100. * correct / len(test_loader.dataset) # 정확도 계산
    return test_loss, test_accuracy, all_preds, all_targets


# ## 코드 돌려보기
# 자, 이제 모든 준비가 끝났습니다. 코드를 돌려서 실제로 훈련이 되는지 확인해봅시다!

# 모델 저장 디렉토리 생성
os.makedirs('./model', exist_ok=True)

# 학습 기록을 저장할 리스트
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# 클래스 이름 정의
CLASSES = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# 학습 및 평가
for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer)
    test_loss, test_accuracy, all_preds, all_targets = evaluate(model, test_loader)
    
    # 학습 기록 저장
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
          epoch, test_loss, test_accuracy))
    
    # 매 에포크마다 모델 저장
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': test_loss,
        'accuracy': test_accuracy
    }, MODEL_PATH)

print('모델이 저장되었습니다:', MODEL_PATH)

# 학습 커브 그리기
plt.figure(figsize=(12, 4))

# 손실 그래프
plt.subplot(1, 2, 1)
plt.plot(test_losses, 'b-', label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(test_accuracies, 'r-', label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('./model/learning_curves.png')
# plt.show()  # 이 줄은 제거

# 혼동 행렬 계산 및 시각화
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(CLASSES.values()),
            yticklabels=list(CLASSES.values()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('./model/confusion_matrix.png')
plt.show()  # 마지막에 한 번만 호출