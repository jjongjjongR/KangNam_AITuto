#!/usr/bin/env python
# coding: utf-8

# # Fashion MNIST 데이터셋 알아보기

from torchvision import datasets, transforms, utils
from torch.utils import data

import matplotlib.pyplot as plt
import numpy as np


# ## Fashion MNIST 데이터셋

transform = transforms.Compose([ 
    transforms.ToTensor() # 이미지를 텐서로 변환
])


trainset = datasets.FashionMNIST(
    root      = './.data/', # 데이터 저장 경로
    train     = True, # 훈련 데이터셋 사용
    download  = True, # 데이터 다운로드
    transform = transform # 데이터 변환 적용
)
testset = datasets.FashionMNIST(
    root      = './.data/', # 데이터 저장 경로
    train     = False, # 테스트 데이터셋 사용
    download  = True, # 데이터 다운로드
    transform = transform # 데이터 변환 적용
)


batch_size = 16 # 배치 크기

train_loader = data.DataLoader(
    dataset     = trainset, # 훈련 데이터셋
    batch_size  = batch_size # 배치 크기
)
test_loader = data.DataLoader(
    dataset     = testset, # 테스트 데이터셋
    batch_size  = batch_size # 배치 크기
)


dataiter       = iter(train_loader) # 훈련 데이터셋 반복자
images, labels = next(dataiter) # 다음 배치 데이터 가져오기


# ## 멀리서 살펴보기
img   = utils.make_grid(images, padding=0) # 이미지 그리드 생성
npimg = img.numpy() # 그리드를 numpy 배열로 변환
plt.figure(figsize=(10, 7)) # 그림 크기 설정
plt.imshow(np.transpose(npimg, (1,2,0))) # 이미지 표시
plt.show() # 그림 표시


print(labels) # 레이블 출력


CLASSES = { # 클래스 이름 정의
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


for label in labels: # 레이블 반복
    index = label.item() # 레이블을 정수로 변환
    print(CLASSES[index]) # 클래스 이름 출력


# ## 가까이서 살펴보기
idx = 1 # 인덱스

item_img = images[idx] # 이미지 가져오기
item_npimg = item_img.squeeze().numpy() # 이미지 크기 조정
plt.title(CLASSES[labels[idx].item()]) # 타이틀 설정
print(item_npimg.shape) # 이미지 크기 출력
plt.imshow(item_npimg, cmap='gray') # 이미지 표시
plt.show() # 그림 표시

