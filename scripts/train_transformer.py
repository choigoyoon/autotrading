import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
from tqdm import tqdm

# 🔥 PyTorch 2.6 호환성 설정
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def setup_cuda_optimization():
    """CUDA 환경 최적화"""
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name()}")
        print(f"   CUDA: {torch.version.cuda}, 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB") # type: ignore
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        return True
    return False

# === 🔥 개선된 데이터셋 (라벨 불균형 해결) ===
class BalancedTradingDataset(torch.utils.data.Dataset):
    """라벨 균형 맞춘 트레이딩 데이터셋"""
    
    def __init__(self, data_dir, max_samples=None, balance_ratio=0.3):
        self.data_dir = Path(data_dir)
        self.balance_ratio = balance_ratio  # 매수/매도 신호 비율
        
        # 파일 리스트 생성 및 라벨별 분류
        self.buy_files = []
        self.sell_files = []
        self.hold_files = []
        
        if self.data_dir.exists():
            print(f"📂 데이터 디렉토리 스캔: {self.data_dir}")
            for file in self.data_dir.iterdir():
                if file.suffix == '.pt':
                    try:
                        # 빠른 라벨 확인 (메타데이터만)
                        data = torch.load(file, map_location='cpu', weights_only=False)
                        label = data.get('label', 0)
                        
                        if label == 1:  # 매수
                            self.buy_files.append(str(file))
                        elif label == -1:  # 매도
                            self.sell_files.append(str(file))
                        else:  # 관망
                            self.hold_files.append(str(file))
                            
                    except Exception:
                        continue
        
        print(f"📊 라벨 분포 - 매수: {len(self.buy_files)}, 매도: {len(self.sell_files)}, 관망: {len(self.hold_files)}")
        
        # 🔥 라벨 균형 맞추기
        self.balanced_files = self._balance_dataset()
        
        # 샘플 제한
        if max_samples and len(self.balanced_files) > max_samples:
            self.balanced_files = self.balanced_files[:max_samples]
        
        print(f"✅ 균형 맞춘 데이터: {len(self.balanced_files)}개")
    
    def _balance_dataset(self):
        """데이터셋 균형 맞추기"""
        # 매수/매도 신호 개수 결정
        signal_count = min(len(self.buy_files), len(self.sell_files))
        target_signal_count = int(signal_count * 2)  # 매수 + 매도
        
        # 관망 신호 개수 (전체의 60-70% 정도)
        target_hold_count = int(target_signal_count * 2)
        
        balanced_files = []
        
        # 매수 신호 추가
        if self.buy_files:
            count = min(target_signal_count // 2, len(self.buy_files))
            balanced_files.extend(self.buy_files[:count])
        
        # 매도 신호 추가
        if self.sell_files:
            count = min(target_signal_count // 2, len(self.sell_files))
            balanced_files.extend(self.sell_files[:count])
        
        # 관망 신호 추가
        if self.hold_files:
            count = min(target_hold_count, len(self.hold_files))
            balanced_files.extend(self.hold_files[:count])
        
        return balanced_files
    
    def __len__(self):
        return len(self.balanced_files)
    
    def __getitem__(self, idx):
        try:
            data = torch.load(self.balanced_files[idx], map_location='cpu', weights_only=False)
            features = data['features']
            label = data['label']
            
            # 🔥 3클래스 라벨링 (매도/관망/매수)
            if label == -1:      # 매도
                label_tensor = torch.tensor(0, dtype=torch.long)
            elif label == 0:     # 관망
                label_tensor = torch.tensor(1, dtype=torch.long)
            else:                # 매수 (label == 1)
                label_tensor = torch.tensor(2, dtype=torch.long)
            
            return features, label_tensor
        
        except Exception:
            # 에러 시 관망 라벨로 반환
            return torch.randn(240, 8), torch.tensor(1, dtype=torch.long)

# === 🚀 개선된 트랜스포머 (신호 감지 특화) ===
class TradingSignalTransformer(nn.Module):
    """트레이딩 신호 감지 특화 트랜스포머"""
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_encoder_layers=4, 
                 dim_feedforward=512, num_classes=3, dropout=0.1):
        super().__init__()
        
        # 입력 전처리
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 위치 인코딩
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model) * 0.1)
        
        # 트랜스포머 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # GELU 활성화 함수
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # 🔥 다중 풀링 전략
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 🔥 신호 감지 헤드 (더 깊은 네트워크)
        self.signal_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # avg + max pooling
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 🔥 신뢰도 헤드 추가
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # 0-1 신뢰도
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 입력 투영
        x = self.input_projection(x)
        
        # 위치 인코딩
        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0)
        x = x + pos_enc
        
        # 트랜스포머
        x = self.transformer_encoder(x)  # [batch, seq, d_model]
        
        # 🔥 다중 풀링
        x_transposed = x.transpose(1, 2)  # [batch, d_model, seq]
        avg_pooled = self.global_pool(x_transposed).squeeze(-1)  # [batch, d_model]
        max_pooled = self.max_pool(x_transposed).squeeze(-1)     # [batch, d_model]
        
        # 풀링 결합
        combined = torch.cat([avg_pooled, max_pooled], dim=1)  # [batch, d_model*2]
        combined = self.dropout(combined)
        
        # 출력
        signal_pred = self.signal_head(combined)
        confidence = self.confidence_head(combined)
        
        return signal_pred, confidence

# === 🔥 개선된 훈련 함수 ===
def run_training_epoch(model, loader, optimizer, device, epoch_num, total_epochs):
    """훈련 에포크"""
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    
    # 🔥 가중 손실 함수 (클래스 불균형 해결)
    class_weights = torch.tensor([2.0, 1.0, 2.0]).to(device)  # [매도, 관망, 매수]
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    confidence_loss_fn = nn.MSELoss()
    
    progress_bar = tqdm(loader, desc=f"Train Epoch {epoch_num}/{total_epochs}")
    
    for features, signal_target in progress_bar:
        features = features.to(device, non_blocking=True)
        signal_target = signal_target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass
        signal_pred, confidence = model(features)
        
        # 🔥 복합 손실 함수
        signal_loss = loss_fn(signal_pred, signal_target)
        
        # 신뢰도 목표 (정답 예측시 높은 신뢰도)
        preds = torch.argmax(signal_pred, dim=1)
        confidence_target = (preds == signal_target).float().unsqueeze(1)
        confidence_loss = confidence_loss_fn(confidence, confidence_target)
        
        total_loss_value = signal_loss + 0.1 * confidence_loss
        
        # Backward pass
        total_loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 통계
        total_loss += total_loss_value.item() * features.size(0)
        total_correct += (preds == signal_target).sum().item()
        total_samples += features.size(0)
        
        # 🔥 매수/매도 신호 개수 추적
        buy_signals = (preds == 2).sum().item()
        sell_signals = (preds == 0).sum().item()
        
        progress_bar.set_postfix(
            loss=f"{total_loss/total_samples:.4f}",
            acc=f"{total_correct/total_samples:.3f}",
            buy=buy_signals,
            sell=sell_signals
        )
    
    return total_loss / total_samples, total_correct / total_samples

def run_validation_epoch(model, loader, device, epoch_num, total_epochs):
    """검증 에포크"""
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_confidences = []
    signal_counts = [0, 0, 0]  # [매도, 관망, 매수]
    
    class_weights = torch.tensor([2.0, 1.0, 2.0]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    progress_bar = tqdm(loader, desc=f"Val Epoch {epoch_num}/{total_epochs}")
    
    with torch.no_grad():
        for features, signal_target in progress_bar:
            features = features.to(device, non_blocking=True)
            signal_target = signal_target.to(device, non_blocking=True)
            
            signal_pred, confidence = model(features)
            loss = loss_fn(signal_pred, signal_target)
            
            total_loss += loss.item() * features.size(0)
            
            preds = torch.argmax(signal_pred, dim=1)
            total_correct += (preds == signal_target).sum().item()
            total_samples += features.size(0)
            
            # 신호 카운트
            for i in range(3):
                signal_counts[i] += int((preds == i).sum().item())
            
            all_confidences.extend(confidence.cpu().numpy())
            
            progress_bar.set_postfix(
                loss=f"{total_loss/total_samples:.4f}",
                acc=f"{total_correct/total_samples:.3f}",
                conf=f"{np.mean(all_confidences):.3f}"
            )
    
    # 🔥 상세 통계 출력
    print(f"   📊 신호 분포 - 매도: {signal_counts[0]}, 관망: {signal_counts[1]}, 매수: {signal_counts[2]}")
    print(f"   📊 평균 신뢰도: {np.mean(all_confidences):.3f}")
    
    return total_loss / total_samples, total_correct / total_samples

# === 🔥 설정 클래스 개선 ===
class ImprovedConfig:
    MODEL_DIR = project_root / "models"
    MODEL_NAME = "improved_trading_transformer_v2.pth"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 20  # 더 많은 에포크
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-4
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0
    
    # 더 큰 모델 (성능 우선)
    D_MODEL = 128
    N_HEAD = 8
    NUM_ENCODER_LAYERS = 4
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1
    NUM_CLASSES = 3  # 매도/관망/매수

def create_improved_dataloaders(batch_size=32, num_workers=4):
    """개선된 데이터로더"""
    from torch.utils.data import DataLoader
    
    sequence_dir = project_root / 'data' / 'sequences_macd'
    dataloaders = {}
    
    # 🔥 더 많은 샘플 사용
    max_samples = {
        'train': 20000,   # 늘림
        'val': 5000,      
        'test': 2000      
    }
    
    for split in ['train', 'val', 'test']:
        data_path = sequence_dir / split
        if data_path.exists():
            dataset = BalancedTradingDataset(
                data_path, 
                max_samples=max_samples[split]
            )
            
            dataloaders[split] = DataLoader(
                dataset, 
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
                persistent_workers=(num_workers > 0)
            )
            print(f"✅ {split} 로더: {len(dataset)}개 샘플")
    
    return dataloaders

def main():
    """개선된 메인 함수"""
    
    setup_cuda_optimization()
    config = ImprovedConfig()
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🚀 개선된 트레이딩 트랜스포머 학습!")
    print(f"Device: {config.DEVICE}, Batch: {config.BATCH_SIZE}, Epochs: {config.EPOCHS}")
    
    # 🔥 개선된 데이터로더
    print("\n📊 균형 맞춘 데이터로더 생성...")
    try:
        dataloaders = create_improved_dataloaders(
            batch_size=config.BATCH_SIZE, 
            num_workers=config.NUM_WORKERS
        )
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return
    
    # 입력 차원 확인
    print("\n🔍 입력 차원 확인...")
    try:
        sample_files = list(Path(project_root / 'data' / 'sequences_macd' / 'train').glob('*.pt'))
        if sample_files:
            sample_data = torch.load(sample_files[0], weights_only=False)
            input_dim = sample_data['features'].shape[1]
            print(f"✅ 입력 차원: {input_dim}")
        else:
            raise FileNotFoundError("샘플 파일 없음")
    except Exception as e:
        print(f"❌ 차원 확인 실패: {e}")
        input_dim = 8
    
    # 🔥 개선된 모델
    print("\n🤖 개선된 모델 초기화...")
    model = TradingSignalTransformer(
        input_dim=input_dim,
        d_model=config.D_MODEL,
        nhead=config.N_HEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 모델: {total_params:,}개 파라미터")
    
    # 🔥 개선된 옵티마이저 (학습률 스케줄러 추가)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    # 학습 루프
    print("\n🚀 개선된 학습 시작!")
    best_val_acc = 0.0
    start_time = time.time()
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(config.EPOCHS):
        # 훈련
        train_loss, train_acc = run_training_epoch(
            model, dataloaders['train'], optimizer, config.DEVICE, epoch + 1, config.EPOCHS
        )
        
        # 검증
        val_loss, val_acc = run_validation_epoch(
            model, dataloaders['val'], config.DEVICE, epoch + 1, config.EPOCHS
        )
        
        # 스케줄러 업데이트
        scheduler.step()
        
        # 기록
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print(f"  Train: Loss {train_loss:.4f}, Acc {train_acc:.4f}")
        print(f"  Val:   Loss {val_loss:.4f}, Acc {val_acc:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'config': {
                    'input_dim': input_dim,
                    'd_model': config.D_MODEL,
                    'num_classes': config.NUM_CLASSES
                }
            }, config.MODEL_DIR / config.MODEL_NAME)
            print(f"  ✅ 최고 성능 모델 저장! (정확도: {val_acc:.4f})")
    
    # 🔥 학습 곡선 시각화
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss History')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy History')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot([scheduler.get_last_lr()[0]] * len(train_losses))
    plt.title('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(config.MODEL_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    total_time = time.time() - start_time
    print(f"\n🎉 개선된 학습 완료!")
    print(f"⏰ 시간: {total_time/60:.1f}분")
    print(f"🎯 최고 검증 정확도: {best_val_acc:.4f}")
    print(f"📁 모델: {config.MODEL_DIR / config.MODEL_NAME}")

if __name__ == '__main__':
    main()
