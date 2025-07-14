import random
import sys
from pathlib import Path
from typing import TypedDict

from typing_extensions import override

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root: Path = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- TypedDicts for static analysis ---

class PTData(TypedDict):
    """ .pt 파일에 저장된 데이터 구조 """
    features: torch.Tensor
    label: torch.Tensor

class SequenceLabel(TypedDict):
    """ 라벨 추출 결과를 위한 구조 """
    file: str
    original_label: int
    converted_label: int

class Prediction(TypedDict):
    """ AI 모델 예측 결과를 위한 구조 """
    file: str
    original_label: int
    true_label: int
    predicted_label: int
    prob_sell: float
    prob_buy: float
    correct: int

class ModelConfig(TypedDict, total=False):
    """ 모델 설정 일부 """
    input_dim: int
    d_model: int
    nhead: int
    num_encoder_layers: int
    dim_feedforward: int
    dropout: float

class Checkpoint(TypedDict):
    """ 모델 체크포인트 구조 """
    config: ModelConfig
    model_state_dict: dict[str, torch.Tensor]
    val_acc: float


def setup_cuda_optimization() -> bool:
    """CUDA 최적화 설정"""
    if torch.cuda.is_available():
        print(f"🔥 CUDA 사용 가능: {torch.cuda.get_device_name()}")
        print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB") # type: ignore
        torch.backends.cudnn.benchmark = True
        return True
    else:
        print("❌ CUDA 사용 불가능 - CPU 모드")
        return False

class ExactTradingTransformer(nn.Module):
    """학습된 모델과 정확히 동일한 구조"""
    
    input_projection: nn.Linear
    positional_encoding: nn.Parameter
    transformer_encoder: nn.TransformerEncoder
    signal_head: nn.Sequential
    dropout: nn.Dropout

    def __init__(self, input_dim: int = 8, d_model: int = 64, nhead: int = 4, 
                 num_encoder_layers: int = 2, dim_feedforward: int = 256, 
                 num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(500, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.signal_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
        
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        
        # 'x'를 재할당하는 대신, 'h' (hidden) 변수를 사용하여 데이터 흐름을 명확히 합니다.
        h = self.input_projection(x)
        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0)
        h = h + pos_enc
        h = self.transformer_encoder(h)
        h = h.mean(dim=1)
        h = self.dropout(h)
        signal_pred = self.signal_head(h)
        return signal_pred

def diagnose_labeling_vs_ai() -> None:
    """🔍 라벨링 vs AI 예측 진단 도구"""
    cuda_available = setup_cuda_optimization()
    device = torch.device("cuda" if cuda_available else "cpu")
    
    model_path = project_root / "models" / "pytorch26_transformer_v1.pth"
    sequences_dir = project_root / "data" / "sequences_macd" / "test"
    output_dir = project_root / "results" / "diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    
    # 1. 원본 라벨링 데이터 분석
    print("\n📊 Step 1: 원본 라벨링 데이터 분석")
    merged_labels_path = project_root / "data" / "processed" / "btc_usdt_kst" / "labeled" / "merged_all_labels.parquet"
    if not merged_labels_path.exists():
        print(f"❌ 원본 라벨링 파일 없음: {merged_labels_path}")
        return
        
    merged_labels: pd.DataFrame = pd.read_parquet(merged_labels_path)
    print(f"✅ 원본 라벨링 데이터 로드: {len(merged_labels):,}행")
    original_label_dist = merged_labels['label'].value_counts().sort_index() # type: ignore
    for label, count in original_label_dist.items():
        print(f"   Label {label}: {count:,} ({count / len(merged_labels) * 100:.2f}%)")
    
    # 2. 시퀀스 데이터 라벨 분포 체크
    print(f"\n📁 Step 2: 시퀀스 데이터 라벨 분포 체크")
    available_files: list[Path] = list(sequences_dir.glob("*.pt"))
    if not available_files:
        print(f"❌ 시퀀스 데이터 없음: {sequences_dir}")
        return

    sample_size = min(1000, len(available_files))
    # np.random.choice는 타입 정보를 잃게 하므로, random.sample을 사용합니다.
    sample_files = random.sample(available_files, sample_size)
    sequence_labels: list[SequenceLabel] = []
    
    for file_path in tqdm(sample_files, desc="라벨 추출"):
        try:
            # .get() 대신 직접 접근하고, isinstance 체크를 제거합니다.
            # TypedDict에 따라 'label' 키는 항상 존재해야 합니다.
            loaded_sequence_data: PTData = torch.load(file_path, map_location='cpu') # type: ignore
            original_label_tensor = loaded_sequence_data['label']
            original_label = int(original_label_tensor.item())
            
            sequence_labels.append({
                'file': file_path.name,
                'original_label': original_label,
                'converted_label': 1 if original_label == 1 else 0
            })
        except Exception as e:
            print(f"⚠️ 파일 로드 실패: {file_path.name}, 오류: {e}")

    sequence_df = pd.DataFrame(sequence_labels)
    if not sequence_df.empty:
        print(f"\n🔄 시퀀스 변환 후 라벨 분포:")
        print(sequence_df['converted_label'].value_counts(normalize=True).sort_index()) # type: ignore
    
    # 3. AI 모델 예측 분석
    print(f"\n🤖 Step 3: AI 모델 예측 분석")
    if not model_path.exists():
        print(f"❌ 모델 파일 없음: {model_path}")
        return

    checkpoint: Checkpoint = torch.load(model_path, map_location=device) # type: ignore
    model_config_dict = checkpoint.get('config', {})
    
    model = ExactTradingTransformer(
        input_dim=model_config_dict.get('input_dim', 8),
        d_model=model_config_dict.get('d_model', 64)
    )
    
    model_state = checkpoint.get('model_state_dict', {})
    filtered_state = {k: v for k, v in model_state.items() if not k.startswith(('return_head', 'confidence_head'))}
    _ = model.load_state_dict(filtered_state, strict=False)
    _ = model.to(device)
    _ = model.eval()
    
    print(f"✅ AI 모델 로드 완료 (학습 정확도: {checkpoint.get('val_acc', 0):.4f})")
    
    sample_predictions: list[Prediction] = []
    test_files_paths = random.sample(available_files, min(500, len(available_files)))
    
    with torch.no_grad():
        for file_path in tqdm(test_files_paths, desc="AI 예측"):
            try:
                loaded_test_data: PTData = torch.load(file_path, map_location='cpu') # type: ignore
                features: torch.Tensor = loaded_test_data['features'].unsqueeze(0).to(device)
                original_label = int(loaded_test_data['label'].item())
                true_label = 1 if original_label == 1 else 0
                
                signal_pred = model(features)
                probabilities = torch.softmax(signal_pred, dim=1).cpu().numpy()[0]
                predicted_label = int(torch.argmax(signal_pred, dim=1).item())
                
                sample_predictions.append({
                    'file': file_path.name, 'original_label': original_label, 'true_label': true_label,
                    'predicted_label': predicted_label, 'prob_sell': probabilities[0],
                    'prob_buy': probabilities[1], 'correct': int(true_label == predicted_label)
                })
            except Exception as e:
                print(f"⚠️ 예측 실패: {file_path.name}, 오류: {e}")

    if not sample_predictions:
        print("❌ 예측 결과 없음")
        return
        
    pred_df = pd.DataFrame(sample_predictions)
    accuracy = pred_df['correct'].mean()
    print(f"\n📈 AI 예측 정확도 (샘플): {accuracy:.2%}")
    
    # 결과 저장
    output_path = output_dir / "ai_vs_labeling_diagnosis.parquet"
    pred_df.to_parquet(output_path, index=False)
    print(f"\n💾 진단 결과 저장 완료: {output_path}")

if __name__ == "__main__":
    diagnose_labeling_vs_ai()
