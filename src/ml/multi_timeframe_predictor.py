# src/ml/multi_timeframe_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MultiTimeframePredictor(nn.Module):
    """
    다중 타임프레임 신호를 입력으로 받아 미회귀 확률과 예상 수익률을 예측하는 딥러닝 모델.
    CNN, 양방향 LSTM, Multi-head Attention을 결합하여 타임프레임 간의 복잡한 관계를 학습합니다.
    """
    def __init__(self, num_timeframes: int = 15, input_features: int = 5, embedding_dim: int = 16):
        """
        모델 초기화

        Args:
            num_timeframes (int): 입력으로 사용할 타임프레임의 수 (기본값: 15)
            input_features (int): 각 타임프레임별 특징의 수 (예: 신호유무, 신호강도 등)
            embedding_dim (int): 타임프레임 ID 임베딩 차원
        """
        super(MultiTimeframePredictor, self).__init__()
        
        # 1. 타임프레임별 임베딩 레이어
        self.timeframe_embedding = nn.Embedding(num_timeframes, embedding_dim)
        
        # 총 입력 채널 수: 원본 특징 + 임베딩
        total_features = input_features + embedding_dim
        
        # 2. CNN 1D for local pattern recognition
        self.conv1d = nn.Conv1d(in_channels=total_features, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(64)
        
        # 3. LSTM for sequential pattern modeling
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.2)
        
        # 4. Attention mechanism for identifying important timeframes
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True) # 128 * 2 (bidirectional)
        
        # 5. 최종 예측을 위한 분류기 및 회귀기
        self.fc_layer = nn.Linear(256, 128)
        self.reversion_classifier = nn.Linear(128, 2)  # [회귀 확률, 미회귀 확률]
        self.return_regressor = nn.Linear(128, 1)      # [예상 수익률]
        
    def forward(self, x: torch.Tensor, timeframe_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        순전파 로직

        Args:
            x (torch.Tensor): 입력 특징 텐서 (batch, num_timeframes, input_features)
            timeframe_ids (torch.Tensor): 타임프레임 ID 텐서 (batch, num_timeframes)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - 미회귀 확률 (batch, 2)
                - 예상 수익률 (batch, 1)
                - 어텐션 가중치 (batch, num_timeframes, num_timeframes)
        """
        # 타임프레임 임베딩
        tf_embeddings = self.timeframe_embedding(timeframe_ids)
        
        # 원본 특징과 임베딩 결합
        x = torch.cat([x, tf_embeddings], dim=2)

        # CNN: (batch, features, timeframes) 형태로 변환하여 적용
        x_conv = x.transpose(1, 2)
        x_conv = self.batch_norm(self.relu(self.conv1d(x_conv)))
        x_conv = x_conv.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x_conv)
        
        # Attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Attention 결과의 평균 풀링
        pooled_out = torch.mean(attn_out, dim=1)
        
        # 최종 예측 레이어
        fc_out = self.relu(self.fc_layer(pooled_out))
        
        reversion_prob = F.softmax(self.reversion_classifier(fc_out), dim=1)
        expected_return = self.return_regressor(fc_out)
        
        return reversion_prob, expected_return, attn_weights

if __name__ == '__main__':
    # 모델 테스트
    batch_size = 32
    num_timeframes = 15
    input_features = 5 # 예: [신호유무, 신호강도, 합의강도, 구조적강도, 신호일관성]

    model = MultiTimeframePredictor(num_timeframes=num_timeframes, input_features=input_features)
    
    # 더미 데이터 생성
    dummy_input = torch.randn(batch_size, num_timeframes, input_features)
    dummy_timeframe_ids = torch.arange(0, num_timeframes).unsqueeze(0).repeat(batch_size, 1)

    # 모델 순전파 테스트
    prob, ret, weights = model(dummy_input, dummy_timeframe_ids)

    print("--- Model Test ---")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Timeframe ID shape: {dummy_timeframe_ids.shape}")
    print(f"Output probability shape: {prob.shape}")
    print(f"Output return shape: {ret.shape}")
    print(f"Attention weights shape: {weights.shape}")

    # 모델 구조 출력
    print("\n--- Model Architecture ---")
    print(model) 