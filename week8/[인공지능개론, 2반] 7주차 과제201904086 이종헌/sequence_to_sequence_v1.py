#!/usr/bin/env python
# coding: utf-8

# # Seq2Seq 기계 번역
# 이번 프로젝트에선 임의로 Seq2Seq 모델을 아주 간단화 시켰습니다.
# 한 언어로 된 문장을 다른 언어로 된 문장으로 번역하는 덩치가 큰 모델이 아닌
# 영어 알파벳 문자열("hello")을 스페인어 알파벳 문자열("hola")로 번역하는 Mini Seq2Seq 모델을 같이 구현해 보겠습니다.

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

# 학습 데이터 정의
training_pairs = [
    ("I go to bed.", "Me voy a la cama."),
    ("I want to sleep.", "Quiero dormir."),
    ("I play a game.", "Juego un juego.")
]

vocab_size = 256  # 총 아스키 코드 개수

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.n_layers = 2
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=self.n_layers, dropout=0.2)
        self.decoder = nn.GRU(hidden_size, hidden_size, num_layers=self.n_layers, dropout=0.2)
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.project = nn.Linear(hidden_size * 2, vocab_size)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, inputs, targets):
        batch_size = 1
        seq_len = inputs.size(0)
        
        # 인코더
        initial_state = self._init_state(batch_size)
        embedding = self.embedding(inputs).unsqueeze(1)
        encoder_output, encoder_state = self.encoder(embedding, initial_state)
        
        # 디코더
        decoder_state = encoder_state
        decoder_input = torch.LongTensor([ord('V')]).to(inputs.device)
        
        outputs = []
        for i in range(targets.size(0)):
            decoder_input = self.embedding(decoder_input).unsqueeze(1)
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            
            # 어텐션 계산
            decoder_output_reshaped = decoder_output.squeeze(1)
            encoder_output_reshaped = encoder_output.squeeze(1)
            
            attention_scores = torch.matmul(decoder_output_reshaped, encoder_output_reshaped.transpose(0, 1))
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            context = torch.matmul(attention_weights, encoder_output_reshaped)
            context = context.unsqueeze(1)
            
            combined = torch.cat([decoder_output, context], dim=-1)
            projection = self.project(combined)
            
            outputs.append(projection)
            decoder_input = torch.LongTensor([targets[i]]).to(inputs.device)
        
        outputs = torch.stack(outputs).squeeze()
        return outputs
    
    def translate(self, input_text, max_length=20):
        input_seq = torch.LongTensor(list(map(ord, input_text)))
        print(f"입력 시퀀스: {input_seq}")
        
        # 인코더
        initial_state = self._init_state(1)
        embedding = self.embedding(input_seq).unsqueeze(1)
        encoder_output, encoder_state = self.encoder(embedding, initial_state)
        
        # 디코더
        decoder_state = encoder_state
        decoder_input = torch.LongTensor([ord('V')])
        
        translated_text = []
        last_char = None
        repeat_count = 0
        
        for i in range(max_length):
            decoder_input = self.embedding(decoder_input).unsqueeze(1)
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            
            # 어텐션 계산
            decoder_output_reshaped = decoder_output.squeeze(1)
            encoder_output_reshaped = encoder_output.squeeze(1)
            
            attention_scores = torch.matmul(decoder_output_reshaped, encoder_output_reshaped.transpose(0, 1))
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            context = torch.matmul(attention_weights, encoder_output_reshaped)
            context = context.unsqueeze(1)
            
            combined = torch.cat([decoder_output, context], dim=-1)
            projection = self.project(combined)
            
            # 소프트맥스 적용 및 예측
            probs = F.softmax(projection.squeeze(0), dim=-1)
            values, indices = torch.topk(probs, k=1)
            next_char = indices.item()
            
            print(f"Step {i}:")
            print(f"  Projection shape: {projection.size()}")
            print(f"  Values: {values.detach().numpy()}")
            print(f"  Indices: {indices.detach().numpy()}")
            print(f"  Predicted char: {chr(next_char)} (ASCII: {next_char})")
            
            # 종료 조건 체크
            if next_char < 32 or next_char > 126:  # 유효하지 않은 ASCII 문자
                break
                
            if next_char == last_char:
                repeat_count += 1
                if repeat_count > 2:  # 같은 문자가 3번 이상 반복되면 종료
                    break
            else:
                repeat_count = 0
                last_char = next_char
            
            # 문장 종료 조건 체크
            if len(translated_text) > 0 and next_char == ord('.'):
                break
                
            translated_text.append(chr(next_char))
            decoder_input = torch.LongTensor([next_char])
        
        result = ''.join(translated_text)
        print(f"최종 번역 결과: {result}")
        return result
    
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_size).zero_()

# 모델 초기화
seq2seq = Seq2Seq(vocab_size, 64)  # 은닉층 크기 증가
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(seq2seq.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# 학습
log = []
for epoch in range(2000):  # 에포크 수 증가
    total_loss = 0
    for x_text, y_text in training_pairs:
        x = torch.LongTensor(list(map(ord, x_text)))
        y = torch.LongTensor(list(map(ord, y_text)))
        
        prediction = seq2seq(x, y)
        loss = criterion(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(seq2seq.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(training_pairs)
    scheduler.step(avg_loss)
    log.append(avg_loss)
    
    if epoch % 100 == 0:
        print(f"\nEpoch: {epoch}, Average Loss: {avg_loss}")
        # 각 학습 예제에 대한 예측 출력
        for x_text, y_text in training_pairs:
            x = torch.LongTensor(list(map(ord, x_text)))
            y = torch.LongTensor(list(map(ord, y_text)))
            prediction = seq2seq(x, y)
            _, top1 = prediction.data.topk(1, 1)
            print(f"Input: {x_text}")
            print(f"Prediction: {[chr(c) for c in top1.squeeze().numpy().tolist()]}")

# 손실 그래프 출력
plt.plot(log)
plt.ylabel('cross entropy loss')
plt.show()

# 번역 테스트
print("\n번역 테스트:")
test_texts = ["I go to bed", "I want to sleep", "I play a game"]
for text in test_texts:
    print(f"\n입력: {text}")
    translated_text = seq2seq.translate(text)
    print(f"번역: {translated_text}")

