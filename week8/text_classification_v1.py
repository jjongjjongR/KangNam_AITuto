#!/usr/bin/env python
# coding: utf-8

# # 프로젝트 1. 영화 리뷰 감정 분석
# **RNN 을 이용해 IMDB 데이터를 가지고 텍스트 감정분석을 해 봅시다.**
# 이번 책에서 처음으로 접하는 텍스트 형태의 데이터셋인 IMDB 데이터셋은 50,000건의 영화 리뷰로 이루어져 있습니다.
# 각 리뷰는 다수의 영어 문장들로 이루어져 있으며, 평점이 7점 이상의 긍정적인 영화 리뷰는 2로, 평점이 4점 이하인 부정적인 영화 리뷰는 1로 레이블링 되어 있습니다. 영화 리뷰 텍스트를 RNN 에 입력시켜 영화평의 전체 내용을 압축하고, 이렇게 압축된 리뷰가 긍정적인지 부정적인지 판단해주는 간단한 분류 모델을 만드는 것이 이번 프로젝트의 목표입니다.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm

# 하이퍼파라미터
BATCH_SIZE = 64
lr = 0.001
EPOCHS = 10
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("다음 기기로 학습합니다:", DEVICE)

# 데이터 로딩하기
print("데이터 로딩중...")
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
vocab.set_default_index(vocab["<unk>"])

def text_pipeline(x):
    return vocab(tokenizer(x))

def label_pipeline(x):
    return int(x) - 1

def process_batch(batch, batch_size):
    label_list, text_list, lengths = [], [], []
    for i, (_label, _text) in enumerate(batch):
        if i >= batch_size:
            break
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        if len(processed_text) == 0:
            processed_text = torch.tensor([vocab["<pad>"]], dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(len(processed_text))
    
    if not text_list:  # 빈 배치 처리
        return None, None, None
    
    max_length = max(lengths)
    padded_text_list = []
    for text in text_list:
        if len(text) < max_length:
            padding = torch.full((max_length - len(text),), vocab["<pad>"], dtype=torch.int64)
            padded_text = torch.cat([text, padding])
        else:
            padded_text = text
        padded_text_list.append(padded_text)
    
    text_tensor = torch.stack(padded_text_list)
    label_tensor = torch.tensor(label_list, dtype=torch.int64)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64)
    
    return label_tensor.to(DEVICE), text_tensor.to(DEVICE), lengths_tensor.to(DEVICE)

def train(model, optimizer, train_iter, batch_size):
    model.train()
    total_loss = 0
    batch = []
    progress_bar = tqdm(desc='Training')
    
    for i, data in enumerate(train_iter):
        batch.append(data)
        if len(batch) == batch_size:
            label, text, lengths = process_batch(batch, batch_size)
            if label is not None:
                optimizer.zero_grad()
                logit = model(text, lengths)
                loss = F.cross_entropy(logit, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                progress_bar.update(1)
                progress_bar.set_postfix({'loss': f'{total_loss/(i+1):.4f}'})
            batch = []
    
    # 마지막 배치 처리
    if batch:
        label, text, lengths = process_batch(batch, batch_size)
        if label is not None:
            optimizer.zero_grad()
            logit = model(text, lengths)
            loss = F.cross_entropy(logit, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': f'{total_loss/(i+1):.4f}'})

def evaluate(model, test_iter, batch_size):
    model.eval()
    corrects, total_loss = 0, 0
    total_samples = 0
    batch = []
    progress_bar = tqdm(desc='Evaluating')
    
    with torch.no_grad():
        for i, data in enumerate(test_iter):
            batch.append(data)
            if len(batch) == batch_size:
                label, text, lengths = process_batch(batch, batch_size)
                if label is not None:
                    logit = model(text, lengths)
                    loss = F.cross_entropy(logit, label, reduction='sum')
                    total_loss += loss.item()
                    corrects += (logit.max(1)[1] == label).sum().item()
                    total_samples += len(label)
                    progress_bar.update(1)
                    progress_bar.set_postfix({'accuracy': f'{100.0 * corrects / total_samples:.2f}%'})
                batch = []
        
        # 마지막 배치 처리
        if batch:
            label, text, lengths = process_batch(batch, batch_size)
            if label is not None:
                logit = model(text, lengths)
                loss = F.cross_entropy(logit, label, reduction='sum')
                total_loss += loss.item()
                corrects += (logit.max(1)[1] == label).sum().item()
                total_samples += len(label)
                progress_bar.update(1)
                progress_bar.set_postfix({'accuracy': f'{100.0 * corrects / total_samples:.2f}%'})
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_accuracy = 100.0 * corrects / total_samples if total_samples > 0 else 0
    return avg_loss, avg_accuracy

vocab_size = len(vocab)
n_classes = 2

print("[단어수]: %d [클래스] %d" % (vocab_size, n_classes))


class BasicGRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(BasicGRU, self).__init__()
        print("Building Basic GRU model...")
        self.n_layers = n_layers
        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers,
                          batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, text, lengths):
        embedded = self.embed(text)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        hidden = self.dropout(hidden[-1])
        return self.out(hidden)


train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

model = BasicGRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_val_loss = None
for e in range(1, EPOCHS+1):
    print(f'\nEpoch {e}/{EPOCHS}')
    train(model, optimizer, train_iter, BATCH_SIZE)
    val_loss, val_accuracy = evaluate(model, test_iter, BATCH_SIZE)
    print(f"[Epoch {e}] Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")
    
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss

model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
test_loss, test_acc = evaluate(model, test_iter, BATCH_SIZE)
print(f'\nFinal Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%')

