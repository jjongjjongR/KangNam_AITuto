#!/usr/bin/env python
# coding: utf-8

# # 4.3 오버피팅과 정규화 (Overfitting and Regularization)
# 머신러닝 모델
# 과적합(Overfitting)

from matplotlib import pyplot as plt
import torch # 파이토치 라이브러리 임포트
import torch.nn as nn # 신경망 모듈 임포트
import torch.optim as optim # 최적화 모듈 임포트
import torch.nn.functional as F # 활성화 함수 모듈 임포트
from torchvision import transforms, datasets # 데이터 변환 및 데이터셋 임포트


USE_CUDA = torch.cuda.is_available() # GPU 사용 가능 여부 확인
DEVICE = torch.device("cuda" if USE_CUDA else "cpu") # GPU 또는 CPU 장치 선택


EPOCHS = 50 # 에포크 수
BATCH_SIZE = 64 # 배치 크기

# 모델 저장 경로 설정
MODEL_PATH = './model/overfitting_and_regularization.pth'  # 학습된 모델을 저장할 경로


# ## 데이터셋에 노이즈 추가하기
# ![original.png](./assets/original.png)
# ![horizontalflip.png](./assets/horizontalflip.png)

train_loader = torch.utils.data.DataLoader( # 훈련 데이터셋 로더 생성
    datasets.FashionMNIST('./.data', # 데이터셋 경로
                   train=True, # 훈련 데이터셋 사용
                   download=True, # 데이터 다운로드
                   transform=transforms.Compose([ # 데이터 변환 적용
                       transforms.RandomHorizontalFlip(), # 수평 뒤집기
                       transforms.ToTensor(), # 텐서로 변환
                       transforms.Normalize((0.1307,), (0.3081,)) # 정규화
                   ])),
    batch_size=BATCH_SIZE, shuffle=True) # 배치 크기와 섞기 여부 설정

test_loader = torch.utils.data.DataLoader( # 테스트 데이터셋 로더 생성
    datasets.FashionMNIST('./.data', # 데이터셋 경로
                   train=False, # 테스트 데이터셋 사용
                   transform=transforms.Compose([ # 데이터 변환 적용
                       transforms.ToTensor(), # 텐서로 변환
                       transforms.Normalize((0.1307,), (0.3081,)) # 정규화
                   ])),
    batch_size=BATCH_SIZE, shuffle=True) # 배치 크기와 섞기 여부 설정


# ## 뉴럴넷으로 Fashion MNIST 학습하기
# 입력 `x` 는 `[배치크기, 색, 높이, 넓이]`로 이루어져 있습니다.
# `x.size()`를 해보면 `[64, 1, 28, 28]`이라고 표시되는 것을 보실 수 있습니다.
# Fashion MNIST에서 이미지의 크기는 28 x 28, 색은 흑백으로 1 가지 입니다.
# 그러므로 입력 x의 총 특성값 갯수는 28 x 28 x 1, 즉 784개 입니다.
# 우리가 사용할 모델은 3개의 레이어를 가진 뉴럴네트워크 입니다. 

class Net(nn.Module): # 뉴럴넷 모델 정의
    def __init__(self, dropout_p=0.2): # 드롭아웃 확률 초기화
        super(Net, self).__init__() # 부모 클래스 초기화
        self.fc1 = nn.Linear(784, 256) # 첫 번째 선형 레이어
        self.fc2 = nn.Linear(256, 128) # 두 번째 선형 레이어
        self.fc3 = nn.Linear(128, 10) # 세 번째 선형 레이어
        # 드롭아웃 확률
        self.dropout_p = dropout_p

    def forward(self, x): # 순전파 정의
        x = x.view(-1, 784) # 입력 데이터 뷰 변환
        x = F.relu(self.fc1(x)) # 활성화 함수 적용
        # 드롭아웃 추가
        x = F.dropout(x, training=self.training,
                      p=self.dropout_p)
        x = F.relu(self.fc2(x)) # 활성화 함수 적용
        # 드롭아웃 추가
        x = F.dropout(x, training=self.training,
                      p=self.dropout_p)
        x = self.fc3(x) # 출력 레이어
        return x


# ## 모델 준비하기 
# `to()` 함수는 모델의 파라미터들을 지정한 곳으로 보내는 역할을 합니다.
# 일반적으로 CPU 1개만 사용할 경우 필요는 없지만,
# GPU를 사용하고자 하는 경우 `to("cuda")`로 지정하여 GPU로 보내야 합니다.
# 지정하지 않을 경우 계속 CPU에 남아 있게 되며 빠른 훈련의 이점을 누리실 수 없습니다.
# 최적화 알고리즘으로 파이토치에 내장되어 있는 `optim.SGD`를 사용하겠습니다.

model        = Net(dropout_p=0.2).to(DEVICE) # 모델 생성 및 GPU로 이동
optimizer    = optim.SGD(model.parameters(), lr=0.01) # 최적화 알고리즘 설정


# ## 학습하기

def train(model, train_loader, optimizer): # 학습 함수 정의
    model.train() # 모델을 학습 모드로 설정
    for batch_idx, (data, target) in enumerate(train_loader): # 배치 인덱스와 데이터, 타겟 가져오기
        data, target = data.to(DEVICE), target.to(DEVICE) # 데이터와 타겟을 GPU로 이동
        optimizer.zero_grad() # 그래디언트 초기화
        output = model(data) # 모델 출력 계산
        loss = F.cross_entropy(output, target) # 크로스 엔트로피 손실 계산 (분류 문제에서 많이 쓰는 손실 함수)
        loss.backward() # 역전파 계산
        optimizer.step() # 파라미터 업데이트


# ## 테스트하기
def evaluate(model, test_loader): # 평가 함수 정의
    model.eval() # 모델을 평가 모드로 설정
    test_loss = 0 # 손실 초기화
    correct = 0 # 맞춘 갯수 초기화
    with torch.no_grad(): # 그래디언트 계산 비활성화
        for data, target in test_loader: # 데이터와 타겟 가져오기
            data, target = data.to(DEVICE), target.to(DEVICE) # 데이터와 타겟을 GPU로 이동
            output = model(data) # 모델 출력 계산
            test_loss += F.cross_entropy(output, target, # 손실 계산
                                         reduction='sum').item()
            # 맞춘 갯수 계산
            pred = output.max(1, keepdim=True)[1] # 최대 값 인덱스 추출
            correct += pred.eq(target.view_as(pred)).sum().item() # 맞춘 갯수 누적

    test_loss /= len(test_loader.dataset) # 손실 평균 계산
    test_accuracy = 100. * correct / len(test_loader.dataset) # 정확도 계산
    return test_loss, test_accuracy # 손실과 정확도 반환


# ## 코드 돌려보기
# 자, 이제 모든 준비가 끝났습니다. 코드를 돌려서 실제로 훈련이 되는지 확인해봅시다!

for epoch in range(1, EPOCHS + 1): # 에포크 반복
    train(model, train_loader, optimizer) # 학습
    test_loss, test_accuracy = evaluate(model, test_loader) # 평가
    
    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format( # 결과 출력
          epoch, test_loss, test_accuracy))




    ##############



#     # 매 에포크마다 모델 저장
#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': test_loss,
#         'accuracy': test_accuracy
#     }, MODEL_PATH)

# print('모델이 저장되었습니다:', MODEL_PATH)

# # 학습 커브 그리기
# plt.figure(figsize=(12, 4))

# # 손실 그래프
# plt.subplot(1, 2, 1)
# plt.plot(test_losses, 'b-', label='Test Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# # 정확도 그래프
# plt.subplot(1, 2, 2)
# plt.plot(test_accuracies, 'r-', label='Test Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.legend()

# plt.tight_layout()
# plt.savefig('./model/learning_curves.png')
# plt.show()

# # 혼동 행렬 계산 및 시각화
# cm = confusion_matrix(all_targets, all_preds)
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=list(CLASSES.values()),
#             yticklabels=list(CLASSES.values()))
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.savefig('./model/confusion_matrix.png')
# plt.show()

# # 저장된 모델 불러오기
# def load_model(model_path):
#     if os.path.exists(model_path):
#         checkpoint = torch.load(model_path)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         epoch = checkpoint['epoch']
#         loss = checkpoint['loss']
#         accuracy = checkpoint['accuracy']
#         print(f'모델을 불러왔습니다. (에포크: {epoch}, 손실: {loss:.4f}, 정확도: {accuracy:.2f}%)')
#         return model, optimizer
#     else:
#         print('저장된 모델이 없습니다.')
#         return None, None

# # 저장된 모델 불러오기 예시
# loaded_model, loaded_optimizer = load_model(MODEL_PATH)

# # 불러온 모델로 테스트
# if loaded_model is not None:
#     test_loss, test_accuracy, all_preds, all_targets = evaluate(loaded_model, test_loader)
#     print(f'불러온 모델 테스트 결과 - 손실: {test_loss:.4f}, 정확도: {test_accuracy:.2f}%')

