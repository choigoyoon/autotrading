import torch
import torch.nn as nn
import math
from pathlib import Path

class PositionalEncoding(nn.Module):
    """
    Transformer 모델을 위한 위치 인코딩을 구현합니다.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TradingTransformer(nn.Module):
    """
    트레이딩 시그널 예측을 위한 Transformer 기반 모델.

    입력: (batch_size, seq_len, input_dim)
    출력:
      - signal_out: (batch_size, num_classes) - 매수/매도/관망 분류
      - return_out: (batch_size, 1) - 수익률 예측
      - confidence_out: (batch_size, 1) - 예측 신뢰도
    """
    def __init__(self, input_dim: int = 50, d_model: int = 256, nhead: int = 8, 
                 num_encoder_layers: int = 6, dim_feedforward: int = 2048, 
                 num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        # Multi-task heads
        self.signal_head = nn.Linear(d_model, num_classes)
        self.return_head = nn.Linear(d_model, 1)
        self.confidence_head = nn.Linear(d_model, 1)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # src shape: (batch_size, seq_len, input_dim)
        
        # 1. Input embedding
        embedded_src = self.input_embedding(src) * math.sqrt(self.d_model)
        
        # 2. Positional encoding
        pos_encoded_src = self.pos_encoder(embedded_src.permute(1, 0, 2)).permute(1, 0, 2)
        
        # 3. Transformer encoder
        # The output will be (batch_size, seq_len, d_model)
        memory = self.transformer_encoder(pos_encoded_src)
        
        # 4. Aggregate sequence information
        # Use the output of the last time step for prediction
        aggregated_output = memory[:, -1, :]
        
        # 5. Multi-task outputs
        signal_out = self.signal_head(aggregated_output)
        return_out = self.return_head(aggregated_output)
        confidence_out = torch.sigmoid(self.confidence_head(aggregated_output)) # Confidence score between 0 and 1
        
        return signal_out, return_out, confidence_out

class MultiTaskLoss(nn.Module):
    """
    분류와 회귀를 위한 멀티태스크 손실 함수.
    """
    def __init__(self, signal_weight: float = 1.0, return_weight: float = 1.0):
        super().__init__()
        self.signal_weight = signal_weight
        self.return_weight = return_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions: tuple, targets: tuple) -> torch.Tensor:
        signal_pred, return_pred, _ = predictions
        signal_target, return_target = targets

        # Signal (Classification) Loss
        signal_loss = self.cross_entropy_loss(signal_pred, signal_target)
        
        # Return (Regression) Loss
        return_loss = self.mse_loss(return_pred.squeeze(), return_target)

        total_loss = (self.signal_weight * signal_loss) + (self.return_weight * return_loss)
        return total_loss

def model_summary(model: nn.Module):
    """
    모델의 파라미터 수와 구조를 출력합니다.
    """
    print("--- 모델 요약 ---")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"총 파라미터: {total_params:,}")
    print(f"학습 가능한 파라미터: {trainable_params:,}")
    print("----------------")

def save_model(model: nn.Module, path: str, epoch: int):
    """모델의 state_dict를 저장합니다."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, save_path)
    print(f"모델이 {save_path}에 저장되었습니다.")

def load_model(model: nn.Module, path: str):
    """저장된 state_dict를 모델에 로드합니다."""
    load_path = Path(path)
    if not load_path.exists():
        print(f"경고: 모델 파일 {load_path}을(를) 찾을 수 없습니다.")
        return None
    
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"모델을 {load_path}에서 로드했습니다 (에포크: {checkpoint['epoch']}).")
    return checkpoint['epoch']


if __name__ == '__main__':
    # --- 테스트 설정 ---
    batch_size = 32
    seq_len = 240
    input_dim_test = 50 # 요구사항에 명시된 피처 수
    d_model_test = 256
    num_classes_test = 3

    # --- 모델 생성 및 요약 ---
    model = TradingTransformer(
        input_dim=input_dim_test, 
        d_model=d_model_test, 
        num_classes=num_classes_test
    )
    model_summary(model)

    # --- 더미 데이터로 Forward Pass 테스트 ---
    dummy_input = torch.randn(batch_size, seq_len, input_dim_test)
    print(f"\n입력 데이터 형태: {dummy_input.shape}")

    signal, ret, conf = model(dummy_input)

    print("\n--- 출력 데이터 형태 ---")
    print(f"Signal output shape: {signal.shape}")
    print(f"Return output shape: {ret.shape}")
    print(f"Confidence output shape: {conf.shape}")

    # --- 손실 함수 테스트 ---
    loss_fn = MultiTaskLoss(signal_weight=0.7, return_weight=0.3)
    dummy_signal_target = torch.randint(0, num_classes_test, (batch_size,))
    dummy_return_target = torch.randn(batch_size)
    
    loss = loss_fn((signal, ret, conf), (dummy_signal_target, dummy_return_target))
    print(f"\n계산된 멀티태스크 손실: {loss.item():.4f}")

    # --- 모델 저장/로드 테스트 ---
    model_path = "models/test_transformer.pth"
    save_model(model, model_path, epoch=10)
    
    new_model = TradingTransformer(
        input_dim=input_dim_test,
        d_model=d_model_test, 
        num_classes=num_classes_test
    )
    load_model(new_model, model_path)
    print("\n모델 아키텍처 스크립트 테스트 완료.") 