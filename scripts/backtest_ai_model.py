import torch
import pandas as pd
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

def setup_cuda_optimization():
    """CUDA 최적화 설정"""
    if torch.cuda.is_available():
        print(f"🔥 CUDA 사용 가능: {torch.cuda.get_device_name()}")
        print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        
        return True
    else:
        print("❌ CUDA 사용 불가능 - CPU 모드")
        return False

# === 🔥 정확한 모델 구조 (학습된 모델과 동일) ===
class ExactTradingTransformer(torch.nn.Module):
    """학습된 모델과 정확히 동일한 구조"""
    
    def __init__(self, input_dim=8, d_model=64, nhead=4, num_encoder_layers=2, 
                 dim_feedforward=256, num_classes=2, dropout=0.1):
        super().__init__()
        
        self.input_projection = torch.nn.Linear(input_dim, d_model)
        self.positional_encoding = torch.nn.Parameter(torch.randn(500, d_model))
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 🔥 학습된 모델과 정확히 동일한 출력 헤드
        self.signal_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Positional encoding
        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0)
        x = x + pos_enc
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Pooling
        x = x.mean(dim=1)
        x = self.dropout(x)
        
        # 🔥 신호 예측만 (학습된 모델과 동일)
        signal_pred = self.signal_head(x)
        
        return signal_pred

def diagnose_labeling_vs_ai():
    """
    🔍 라벨링 vs AI 예측 진단 도구
    """
    # --- CUDA 설정 ---
    cuda_available = setup_cuda_optimization()
    device = torch.device("cuda" if cuda_available else "cpu")
    
    # --- 경로 설정 ---
    model_path = project_root / "models" / "pytorch26_transformer_v1.pth"
    sequences_dir = project_root / "data" / "sequences_macd" / "test"
    output_dir = project_root / "results" / "diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 라벨링 vs AI 예측 진단 시작!")
    print("=" * 60)
    
    # --- 1. 원본 라벨링 데이터 분석 ---
    print("\n📊 Step 1: 원본 라벨링 데이터 분석")
    try:
        merged_labels_path = project_root / "data" / "processed" / "btc_usdt_kst" / "labeled" / "merged_all_labels.parquet"
        
        if merged_labels_path.exists():
            merged_labels = pd.read_parquet(merged_labels_path)
            print(f"✅ 원본 라벨링 데이터 로드: {len(merged_labels):,}행")
            
            # 원본 라벨 분포
            original_label_dist = merged_labels['label'].value_counts().sort_index()
            print(f"\n📈 원본 라벨 분포:")
            for label, count in original_label_dist.items():
                pct = count / len(merged_labels) * 100
                label_name = {-1: "매도", 0: "관망", 1: "매수"}.get(label, f"기타({label})")
                print(f"   {label_name}({label}): {count:,}개 ({pct:.2f}%)")
        else:
            print(f"❌ 원본 라벨링 파일 없음: {merged_labels_path}")
            return
            
    except Exception as e:
        print(f"❌ 원본 라벨링 데이터 분석 실패: {e}")
        return
    
    # --- 2. 시퀀스 데이터 라벨 분포 체크 ---
    print(f"\n📁 Step 2: 시퀀스 데이터 라벨 분포 체크")
    try:
        # 시퀀스 파일들 스캔
        available_files = list(sequences_dir.glob("*.pt"))
        if not available_files:
            print(f"❌ 시퀀스 데이터 없음: {sequences_dir}")
            return
        
        print(f"✅ 시퀀스 파일 발견: {len(available_files):,}개")
        
        # 샘플링해서 라벨 분포 확인 (1000개 샘플)
        sample_size = min(1000, len(available_files))
        sample_files = np.random.choice(available_files, sample_size, replace=False)
        
        sequence_labels = []
        print(f"📊 {sample_size}개 파일에서 라벨 샘플링 중...")
        
        for file in tqdm(sample_files, desc="라벨 추출"):
            try:
                data = torch.load(file, map_location='cpu', weights_only=False)
                original_label = data['label']
                
                # 변환 과정 시뮬레이션
                if original_label == -1:
                    converted_label = 0  # 매도 → 0
                elif original_label == 1:
                    converted_label = 1  # 매수 → 1
                else:
                    converted_label = 0  # 관망 → 0
                
                sequence_labels.append({
                    'file': file.name,
                    'original_label': int(original_label),
                    'converted_label': converted_label
                })
                
            except Exception:
                print(f"⚠️ 파일 로드 실패: {file.name}")
        
        # 시퀀스 라벨 분포 분석
        sequence_df = pd.DataFrame(sequence_labels)
        
        print(f"\n📈 시퀀스 원본 라벨 분포 (변환 전):")
        orig_dist = sequence_df['original_label'].value_counts().sort_index()
        for label, count in orig_dist.items():
            pct = count / len(sequence_df) * 100
            label_name = {-1: "매도", 0: "관망", 1: "매수"}.get(label, f"기타({label})")
            print(f"   {label_name}({label}): {count}개 ({pct:.2f}%)")
        
        print(f"\n🔄 시퀀스 변환 후 라벨 분포:")
        conv_dist = sequence_df['converted_label'].value_counts().sort_index()
        for label, count in conv_dist.items():
            pct = count / len(sequence_df) * 100
            label_name = {0: "관망/매도", 1: "매수"}.get(label, f"기타({label})")
            print(f"   {label_name}({label}): {count}개 ({pct:.2f}%)")
        
        # 🚨 핵심 문제 진단
        buy_ratio_original = len(merged_labels[merged_labels['label'] == 1]) / len(merged_labels) * 100
        buy_ratio_sequence = len(sequence_df[sequence_df['converted_label'] == 1]) / len(sequence_df) * 100
        
        print(f"\n🚨 핵심 문제 진단:")
        print(f"   원본 매수 라벨 비율: {buy_ratio_original:.2f}%")
        print(f"   시퀀스 매수 라벨 비율: {buy_ratio_sequence:.2f}%")
        
        if buy_ratio_sequence < 1.0:
            print(f"   ❌ 시퀀스에서 매수 라벨이 심각하게 부족함!")
        elif buy_ratio_sequence < buy_ratio_original * 0.5:
            print(f"   ⚠️ 시퀀스에서 매수 라벨이 원본 대비 많이 감소함")
        else:
            print(f"   ✅ 시퀀스 매수 라벨 비율 정상")
            
    except Exception as e:
        print(f"❌ 시퀀스 데이터 분석 실패: {e}")
        return
    
    # --- 3. AI 모델 예측 분석 ---
    print(f"\n🤖 Step 3: AI 모델 예측 분석")
    try:
        # 모델 로드
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model_config = checkpoint.get('config', {})
        
        model = ExactTradingTransformer(
            input_dim=model_config.get('input_dim', 8),
            d_model=model_config.get('d_model', 64)
        )
        
        # 모델 가중치 로드
        model_state = checkpoint['model_state_dict']
        filtered_state = {k: v for k, v in model_state.items() 
                         if not (k.startswith('return_head') or k.startswith('confidence_head'))}
        
        model.load_state_dict(filtered_state, strict=False)
        model.to(device)
        model.eval()
        
        print(f"✅ AI 모델 로드 완료 (학습 정확도: {checkpoint.get('val_acc', 0):.4f})")
        
        # 샘플 예측 수행
        sample_predictions = []
        test_files = np.random.choice(available_files, min(500, len(available_files)), replace=False)
        
        print(f"🔮 {len(test_files)}개 샘플로 AI 예측 테스트...")
        
        with torch.no_grad():
            for file in tqdm(test_files, desc="AI 예측"):
                try:
                    data = torch.load(file, map_location='cpu', weights_only=False)
                    features = data['features'].unsqueeze(0).to(device)  # 배치 차원 추가
                    original_label = int(data['label'])
                    
                    # 라벨 변환
                    true_label = 1 if original_label == 1 else 0
                    
                    # AI 예측
                    signal_pred = model(features)
                    probabilities = torch.softmax(signal_pred, dim=1)[0].cpu().numpy()
                    predicted_label = int(torch.argmax(signal_pred, dim=1).cpu().numpy()[0])
                    
                    sample_predictions.append({
                        'file': file.name,
                        'original_label': original_label,
                        'true_label': true_label,
                        'predicted_label': predicted_label,
                        'prob_sell': probabilities[0],
                        'prob_buy': probabilities[1],
                        'correct': int(true_label == predicted_label)
                    })
                    
                except Exception:
                    continue
        
        # AI 예측 결과 분석
        pred_df = pd.DataFrame(sample_predictions)
        
        print(f"\n🎯 AI 예측 결과 분석:")
        print(f"   총 예측 샘플: {len(pred_df)}개")
        print(f"   전체 정확도: {pred_df['correct'].mean():.4f} ({pred_df['correct'].mean()*100:.2f}%)")
        
        # 예측 라벨 분포
        ai_pred_dist = pred_df['predicted_label'].value_counts().sort_index()
        print(f"\n🤖 AI 예측 라벨 분포:")
        for label, count in ai_pred_dist.items():
            pct = count / len(pred_df) * 100
            label_name = {0: "관망/매도", 1: "매수"}.get(label, f"기타({label})")
            print(f"   {label_name}({label}): {count}개 ({pct:.2f}%)")
        
        # 실제 라벨 분포
        true_dist = pred_df['true_label'].value_counts().sort_index()
        print(f"\n📊 실제 라벨 분포:")
        for label, count in true_dist.items():
            pct = count / len(pred_df) * 100
            label_name = {0: "관망/매도", 1: "매수"}.get(label, f"기타({label})")
            print(f"   {label_name}({label}): {count}개 ({pct:.2f}%)")
        
        # 매수 확률 분포
        buy_prob_stats = pred_df['prob_buy'].describe()
        print(f"\n📈 매수 확률 통계:")
        print(f"   평균: {buy_prob_stats['mean']:.4f}")
        print(f"   중간값: {buy_prob_stats['50%']:.4f}")
        print(f"   최대값: {buy_prob_stats['max']:.4f}")
        print(f"   표준편차: {buy_prob_stats['std']:.4f}")
        
        # 다양한 임계값에서의 매수 신호
        print(f"\n🎚️ 임계값별 매수 신호 수:")
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            buy_signals = (pred_df['prob_buy'] > threshold).sum()
            pct = buy_signals / len(pred_df) * 100
            print(f"   임계값 {threshold}: {buy_signals}개 ({pct:.2f}%)")
        
        # 결과 저장
        pred_df.to_csv(output_dir / "ai_prediction_analysis.csv", index=False)
        sequence_df.to_csv(output_dir / "sequence_label_analysis.csv", index=False)
        
    except Exception as e:
        print(f"❌ AI 모델 분석 실패: {e}")
        return
    
    # --- 4. 최종 진단 결과 ---
    print(f"\n" + "="*60)
    print(f"🔍 최종 진단 결과")
    print(f"="*60)
    
    # 매수 라벨 추적
    original_buy_count = len(merged_labels[merged_labels['label'] == 1])
    sequence_buy_count = len(sequence_df[sequence_df['converted_label'] == 1])
    ai_buy_predictions = len(pred_df[pred_df['predicted_label'] == 1])
    ai_high_prob_buy = len(pred_df[pred_df['prob_buy'] > 0.5])
    
    print(f"\n📈 매수 라벨 추적:")
    print(f"   1️⃣ 원본 라벨링: {original_buy_count:,}개 매수 라벨")
    print(f"   2️⃣ 시퀀스 변환: {sequence_buy_count}개 매수 라벨 (샘플 기준)")
    print(f"   3️⃣ AI 예측: {ai_buy_predictions}개 매수 예측")
    print(f"   4️⃣ AI 고확률: {ai_high_prob_buy}개 (확률 50% 이상)")
    
    # 문제 진단
    problems = []
    if sequence_buy_count == 0:
        problems.append("🚨 시퀀스 데이터에 매수 라벨이 없음 - 시퀀스 생성 과정 문제")
    if ai_buy_predictions == 0:
        problems.append("🚨 AI가 매수 예측을 전혀 하지 않음 - 모델 학습 문제")
    if pred_df['prob_buy'].max() < 0.7:
        problems.append("⚠️ AI 매수 확률이 전반적으로 낮음 - 모델 보수적 성향")
    
    if problems:
        print(f"\n❌ 발견된 문제들:")
        for i, problem in enumerate(problems, 1):
            print(f"   {i}. {problem}")
    else:
        print(f"\n✅ 특별한 문제가 발견되지 않았습니다.")
    
    # 해결책 제안
    print(f"\n💡 해결책 제안:")
    if sequence_buy_count == 0:
        print(f"   1. generate_sequences.py에서 매수 라벨 샘플링 비율 조정")
        print(f"   2. 전체 데이터에서 매수 라벨 영역 우선 선택")
    
    if ai_buy_predictions == 0:
        print(f"   3. 모델 재학습시 클래스 가중치 적용")
        print(f"   4. 임계값을 0.3-0.4로 낮춰서 매수 신호 증가")
    
    print(f"\n📁 상세 결과는 다음 파일에서 확인:")
    print(f"   - {output_dir / 'ai_prediction_analysis.csv'}")
    print(f"   - {output_dir / 'sequence_label_analysis.csv'}")

if __name__ == "__main__":
    diagnose_labeling_vs_ai()
