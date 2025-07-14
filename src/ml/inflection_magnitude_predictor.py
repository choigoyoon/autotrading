# src/ml/inflection_magnitude_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class InflectionMagnitudePredictor(nn.Module):
    """
    변곡점의 반등 강도를 예측하는 딥러닝 모델.

    이 모델은 변곡점 이전의 '컨텍스트(Context)'와 변곡 '순간(Moment)'의 특징을
    별도로 분석하고 융합하여, 반등의 강도, 지속기간, 최대수익률을 예측합니다.

    - Context Transformer: 과거 N일간의 시장 상황(하락 지속 기간, 거래량 패턴 등)을 분석합니다.
    - Moment Analyzer: 변곡점 순간의 특징(다중 타임프레임 합의, 다이버전스 강도 등)을 분석합니다.
    - Cross-Attention Fusion: 컨텍스트 정보와 순간 정보를 융합하여 시너지를 창출합니다.
    - Multi-task Output: 3가지 목표(강도, 지속기간, 수익률)를 동시에 예측합니다.
    """
    def __init__(self, 
                 context_seq_len: int = 30,
                 context_feature_dim: int = 10, 
                 moment_feature_dim: int = 15,
                 d_model: int = 128, 
                 n_head: int = 8, 
                 num_encoder_layers: int = 3,
                 dim_feedforward: int = 512,
                 num_magnitude_classes: int = 4,
                 num_duration_classes: int = 4,
                 dropout: float = 0.1):
        """
        모델 아키텍처를 초기화합니다.

        Args:
            context_seq_len (int): 컨텍스트 시퀀스의 길이 (예: 30일).
            context_feature_dim (int): 컨텍스트 데이터의 특징 차원.
            moment_feature_dim (int): 변곡점 순간 데이터의 특징 차원.
            d_model (int): 트랜스포머 및 주요 레이어의 기본 차원.
            n_head (int): 어텐션 헤드의 수.
            num_encoder_layers (int): 트랜스포머 인코더 레이어의 수.
            dim_feedforward (int): 트랜스포머 피드포워드 네트워크의 차원.
            num_magnitude_classes (int): 반등 강도 분류 클래스 수 (약함/보통/강함/폭발적).
            num_duration_classes (int): 반등 지속기간 분류 클래스 수 (1일/1주/1달/3달+).
            dropout (float): 드롭아웃 비율.
        """
        super().__init__()
        self.d_model = d_model

        # 1. 입력 프로젝션 레이어
        self.context_projection = nn.Linear(context_feature_dim, d_model)
        self.moment_projection = nn.Linear(moment_feature_dim, d_model)
        
        # Positional Encoding for Context Transformer
        self.positional_encoding = nn.Parameter(torch.zeros(1, context_seq_len, d_model))

        # 2. Context Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_head, dim_feedforward, dropout, batch_first=True
        )
        self.context_transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # 3. Moment Analyzer (단순 MLP)
        self.moment_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 4. Cross-Attention Fusion
        self.cross_attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)

        # 5. Multi-task Output 예측 헤드
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.magnitude_classifier = nn.Linear(d_model * 2, num_magnitude_classes)
        self.duration_classifier = nn.Linear(d_model * 2, num_duration_classes)
        self.return_regressor = nn.Linear(d_model * 2, 1)

    def forward(self, context_sequence: torch.Tensor, moment_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        모델의 순전파 로직을 정의합니다.

        Args:
            context_sequence (torch.Tensor): (batch, context_seq_len, context_feature_dim)
                                              변곡점 이전 N일간의 시장 상황 데이터.
            moment_features (torch.Tensor): (batch, moment_feature_dim)
                                            변곡점 순간의 특징 데이터.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - magnitude_logits (torch.Tensor): 반등 강도 예측 (batch, num_magnitude_classes)
                - duration_logits (torch.Tensor): 지속 기간 예측 (batch, num_duration_classes)
                - predicted_return (torch.Tensor): 최대 수익률 예측 (batch, 1)
        """
        # 1. 입력 프로젝션 및 인코딩
        context_proj = self.context_projection(context_sequence) + self.positional_encoding
        moment_proj = self.moment_projection(moment_features)

        # 2. Context 정보 처리
        context_embedding = self.context_transformer(context_proj) # (batch, seq_len, d_model)

        # 3. Moment 정보 처리
        moment_embedding = self.moment_analyzer(moment_proj) # (batch, d_model)
        
        # 4. Cross-Attention을 이용한 정보 융합
        # Moment가 Query, Context가 Key/Value가 되어 관련성 높은 과거 정보를 추출
        query = moment_embedding.unsqueeze(1) # (batch, 1, d_model)
        key = context_embedding
        value = context_embedding
        
        fused_embedding, _ = self.cross_attention(query, key, value) # (batch, 1, d_model)
        fused_embedding = fused_embedding.squeeze(1) # (batch, d_model)

        # 5. 최종 예측
        output_features = self.output_layer(fused_embedding)
        
        magnitude_logits = self.magnitude_classifier(output_features)
        duration_logits = self.duration_classifier(output_features)
        predicted_return = self.return_regressor(output_features)

        return magnitude_logits, duration_logits, predicted_return

if __name__ == '__main__':
    # --- 모델 테스트 ---
    # 하이퍼파라미터 정의
    batch_size = 4
    context_len = 30  # 과거 30일
    context_features = 10 # 하락기간, 거래량 패턴, 변동성 등
    moment_features = 15  # 다중 타임프레임 합의, 다이버전스 강도 등
    
    # 모델 인스턴스 생성
    model = InflectionMagnitudePredictor(
        context_seq_len=context_len,
        context_feature_dim=context_features,
        moment_feature_dim=moment_features
    )
    
    # 더미 데이터 생성
    dummy_context = torch.randn(batch_size, context_len, context_features)
    dummy_moment = torch.randn(batch_size, moment_features)
    
    # 모델 순전파 테스트
    mag_logits, dur_logits, pred_return = model(dummy_context, dummy_moment)
    
    print("--- Inflection Magnitude Predictor Test ---")
    print(f"Model Architecture:\n{model}")
    print("\n--- Input Shapes ---")
    print(f"Context Sequence: {dummy_context.shape}")
    print(f"Moment Features:  {dummy_moment.shape}")
    print("\n--- Output Shapes ---")
    print(f"Magnitude Logits: {mag_logits.shape}")
    print(f"Duration Logits:  {dur_logits.shape}")
    print(f"Predicted Return: {pred_return.shape}")

    # --- 예측값 확인 ---
    mag_probs = F.softmax(mag_logits, dim=1)
    dur_probs = F.softmax(dur_logits, dim=1)
    
    print("\n--- Sample Predictions (for one item in batch) ---")
    print(f"Predicted Magnitude Probabilities: {mag_probs[0].detach().numpy()}")
    print(f"Predicted Magnitude Class: {torch.argmax(mag_probs[0]).item()}")
    print(f"Predicted Duration Probabilities:  {dur_probs[0].detach().numpy()}")
    print(f"Predicted Duration Class:  {torch.argmax(dur_probs[0]).item()}")
    print(f"Predicted Max Return: {pred_return[0].item():.4f}") 