import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import time
import ccxt  # 실시간 데이터 수집을 위해 ccxt 라이브러리 설치 필요 (pip install ccxt)
import warnings
import sys # sys 모듈 임포트
import os
from ta.trend import macd_diff

# 프로젝트 루트 설정 (이 파일의 위치에 따라 조정 필요)
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ml.trading_transformer import TradingTransformer
from tools.backtest_ai_model import BacktestConfig

# --- 설정 ---
class LiveConfig(BacktestConfig):
    # 모델 및 스케일러 경로
    MODEL_DIR = project_root / "models"
    MODEL_NAME = "trading_transformer_v1.pth"
    SCALER_PATH = project_root / "data" / "sequences" / "scaler.joblib"

    # 모델 하이퍼파라미터 (train_transformer.py와 동일해야 함)
    INPUT_DIM = 8  # open, high, low, close, volume, macd, macdsignal, macdhist
    D_MODEL = 128
    N_HEAD = 8
    NUM_ENCODER_LAYERS = 4
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1
    NUM_CLASSES = 2  # 0: SELL, 1: BUY

    # 예측 설정
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEQUENCE_LENGTH = 240  # 240분 데이터
    CONFIDENCE_THRESHOLD = 0.80  # 신뢰도 80% 이상인 예측만 출력
    SYMBOL = 'BTC/USDT'
    TIMEFRAME = '1m'

    # 실시간 데이터 소스 (시뮬레이션용)
    DATA_SOURCE_PATH = project_root / 'data' / 'processed' / 'btc_usdt_kst' / 'resampled_ohlcv' / '1min.parquet'
    # 데이터 업데이트 간격(초)
    REFRESH_INTERVAL_SECONDS = 60

# --- 함수 구현 ---

def load_trained_model() -> TradingTransformer:
    """학습된 PyTorch 모델과 Scaler를 로드합니다."""
    print("--- 1. 모델 및 스케일러 로딩 ---")
    
    # 모델 인스턴스화
    model = TradingTransformer(
        input_dim=LiveConfig.INPUT_DIM,
        d_model=LiveConfig.D_MODEL,
        nhead=LiveConfig.N_HEAD,
        num_encoder_layers=LiveConfig.NUM_ENCODER_LAYERS,
        dim_feedforward=LiveConfig.DIM_FEEDFORWARD,
        num_classes=LiveConfig.NUM_CLASSES,
        dropout=LiveConfig.DROPOUT
    ).to(LiveConfig.DEVICE)

    # 모델 가중치 로드
    model_path = LiveConfig.MODEL_DIR / LiveConfig.MODEL_NAME
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=LiveConfig.DEVICE))
    model.eval()  # 평가 모드로 설정
    print(f"✅ 모델 로드 완료: {model_path}")

    return model

def get_latest_sequence(scaler) -> pd.DataFrame:
    """거래소에서 최신 시퀀스 데이터를 가져와 전처리합니다."""
    print(f"\n--- 2. 최신 {LiveConfig.SEQUENCE_LENGTH}분 데이터 수집 ({LiveConfig.SYMBOL}) ---")
    
    try:
        exchange = ccxt.binance()  # 다른 거래소로 변경 가능
        ohlcv = exchange.fetch_ohlcv(LiveConfig.SYMBOL, LiveConfig.TIMEFRAME, limit=LiveConfig.SEQUENCE_LENGTH + 50) # 여유분 포함
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # 최신 240개 데이터만 선택
        df = df.iloc[-LiveConfig.SEQUENCE_LENGTH:]
        
        if len(df) < LiveConfig.SEQUENCE_LENGTH:
            raise ValueError(f"데이터가 충분하지 않습니다. {len(df)}/{LiveConfig.SEQUENCE_LENGTH}개 수집됨.")

        print(f"✅ 최신 데이터 수집 완료 (시작: {df.index[0]}, 끝: {df.index[-1]})")
        
        # 피처 엔지니어링 (MACD)
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macdsignal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macdhist'] = df['macd'] - df['macdsignal']
        
        # 정규화
        features_to_use = ['open', 'high', 'low', 'close', 'volume', 'macd', 'macdsignal', 'macdhist']
        df_features = df[features_to_use]
        
        sequence_scaled = scaler.transform(df_features)
        
        return sequence_scaled, df.iloc[-1] # 마지막 행 데이터(MACD 비교용) 반환

    except Exception as e:
        print(f"❌ 데이터 수집 실패: {e}")
        return None, None

def predict_signals(model: TradingTransformer, sequence: np.ndarray) -> tuple:
    """AI 모델로 시그널을 예측합니다."""
    print("\n--- 3. AI 시그널 예측 수행 ---")
    
    with torch.no_grad():
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(LiveConfig.DEVICE)
        
        signal_logits, _, confidence = model(sequence_tensor)
        
        # 예측 결과 처리
        probabilities = F.softmax(signal_logits, dim=1)
        confidence.item()
        
        predicted_class = torch.argmax(probabilities).item()
        prediction_confidence = probabilities[0, predicted_class].item()
        
        signal = "BUY" if predicted_class == 1 else "SELL"
        
        print(f"✅ 예측 완료: {signal} (신뢰도: {prediction_confidence:.2%})")
        return signal, prediction_confidence

def compare_with_macd(ai_signal: str, last_row: pd.Series) -> str:
    """AI 시그널과 MACD 시그널을 비교합니다."""
    macd_hist = last_row['macdhist']
    macd_signal = "BUY" if macd_hist > 0 else "SELL"
    
    agreement = "✅ (동일 시그널)" if ai_signal == macd_signal else "❌ (시그널 불일치)"
    print(f"   - MACD 시그널: {macd_signal} (히스토그램: {macd_hist:.4f})")
    return agreement

def display_final_signal(signal: str, confidence: float, macd_agreement: str):
    """최종 예측 결과를 형식에 맞춰 출력합니다."""
    print("\n---------------- 최종 예측 결과 ----------------")
    if confidence < LiveConfig.CONFIDENCE_THRESHOLD:
        print(f"📉 AI 예측 보류: 신뢰도 낮음 ({confidence:.2%})")
        print(f"   (필터링 기준: {LiveConfig.CONFIDENCE_THRESHOLD:.0%})")
    else:
        print(f"🤖 AI 예측: {signal} (신뢰도: {confidence:.2%})")
        # 예상 수익률/리스크는 현재 모델에서 실제 값을 예측하지 않으므로 임시 값으로 표시
        print(f"📈 예상 수익률: N/A")
        print(f"⚠️ 예상 리스크: N/A")
        print(f"🎯 MACD 합의: {macd_agreement}")
    print("------------------------------------------------")

def load_latest_model_and_scaler(config: LiveConfig) -> tuple[TradingTransformer, object, torch.device]:
    """학습된 모델과 스케일러를 로드합니다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TradingTransformer(
        input_dim=config.INPUT_DIM, d_model=config.D_MODEL, nhead=config.N_HEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS, dim_feedforward=config.DIM_FEEDFORWARD,
        num_classes=config.NUM_CLASSES
    ).to(device)
    
    model_path = config.MODEL_DIR / config.MODEL_NAME
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    scaler = joblib.load(config.SCALER_PATH)
    
    print("✅ 모델 및 스케일러 로드 완료")
    print(f"   - 모델: {model_path}")
    print(f"   - 스케일러: {config.SCALER_PATH}")
    print(f"   - 실행 디바이스: {device}")
    
    return model, scaler, device

def get_current_data(config: LiveConfig, iteration: int) -> pd.DataFrame:
    """
    최신 시퀀스 데이터를 가져옵니다. 
    실제 환경에서는 API 호출로 대체되어야 합니다.
    여기서는 저장된 parquet 파일의 데이터를 순차적으로 사용하여 시뮬레이션합니다.
    """
    df = pd.read_parquet(config.DATA_SOURCE_PATH)
    
    # 시뮬레이션을 위해 데이터셋의 뒷부분을 순차적으로 사용
    # 60000은 테스트 데이터셋의 대략적인 크기입니다.
    start_index = len(df) - 60000 + (iteration * 1) # 1분씩 이동
    end_index = start_index + config.SEQUENCE_LENGTH
    
    if end_index > len(df):
        print("⚠️ 데이터셋의 끝에 도달했습니다. 시뮬레이션을 처음부터 다시 시작합니다.")
        start_index = len(df) - 60000
        end_index = start_index + config.SEQUENCE_LENGTH
        iteration = 0

    return df.iloc[start_index:end_index], iteration

def get_features_from_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV 데이터로부터 모델 입력에 사용할 피처를 생성합니다.
    (generate_sequences.py의 로직과 일치해야 함)
    """
    df['return'] = df['close'].pct_change().fillna(0)
    df['volume_change'] = df['volume'].pct_change().fillna(0)
    # MACD 히스토그램을 피처로 사용
    df['macd_hist'] = macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9).fillna(0)
    
    # 사용할 피처 선택
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'return', 'volume_change', 'macd_hist']
    return df[feature_columns]

def predict_with_confidence(model: TradingTransformer, scaler: object, sequence: pd.DataFrame, device: torch.device) -> tuple[int, float]:
    """AI 모델을 사용하여 다음 스텝을 예측하고 신뢰도를 반환합니다."""
    scaled_features = scaler.transform(sequence)
    features_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        signal_logits, _, _ = model(features_tensor)
        probs = torch.softmax(signal_logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
    
    # 모델 출력 1: 매수, 0: 매도
    signal = 1 if predicted_class.item() == 1 else -1
    return signal, confidence.item()

def generate_hybrid_signal(ai_signal: int, ai_confidence: float, macd_signal: int) -> tuple[str, str, str]:
    """AI와 MACD 신호를 종합하여 최종 투자 의견과 권장 포지션을 생성합니다."""
    ai_text = f"BUY (신뢰도: {ai_confidence:.1%})" if ai_signal == 1 else f"SELL (신뢰도: {ai_confidence:.1%})"
    macd_text = "BUY" if macd_signal == 1 else "SELL"
    
    hybrid_text = ""
    position_text = ""
    
    if ai_signal == macd_signal:
        if ai_confidence >= 0.85:
            hybrid_text = f"강력한 {ai_text.split(' ')[0]} 신호!"
            position_text = "40% (적극)"
        elif ai_confidence >= 0.70:
            hybrid_text = f"일치된 {ai_text.split(' ')[0]} 신호"
            position_text = "20% (중간)"
        else:
            hybrid_text = f"약한 {ai_text.split(' ')[0]} 신호"
            position_text = "10% (소극)"
    else:
        hybrid_text = "신호 충돌, 관망 권장"
        position_text = "0% (관망)"
        
    return ai_text, macd_text, hybrid_text, position_text

def monitor_real_time(config: LiveConfig):
    """실시간 모니터링 루프를 실행합니다."""
    model, scaler, device = load_latest_model_and_scaler(config)
    iteration = 0
    
    try:
        while True:
            # 1. 최신 데이터 수집 (시뮬레이션)
            current_sequence_df, iteration = get_current_data(config, iteration)
            if current_sequence_df.empty:
                time.sleep(config.REFRESH_INTERVAL_SECONDS)
                continue

            # 2. 피처 생성
            features_df = get_features_from_data(current_sequence_df)

            # 3. AI 예측 수행
            ai_signal, ai_confidence = predict_with_confidence(model, scaler, features_df, device)
            
            # 4. MACD 신호 계산 (가장 마지막 데이터 기준)
            last_macd_hist = features_df['macd_hist'].iloc[-1]
            macd_signal = 1 if last_macd_hist > 0 else -1
            
            # 5. 하이브리드 신호 생성
            ai_text, macd_text, hybrid_text, position_text = generate_hybrid_signal(ai_signal, ai_confidence, macd_signal)
            
            # 6. 결과 출력
            os.system('cls' if os.name == 'nt' else 'clear')
            current_time = current_sequence_df.index[-1]
            current_price = current_sequence_df['close'].iloc[-1]
            
            print("="*25)
            print("  실시간 트레이딩 신호")
            print("="*25)
            print(f"⏰ 시간: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"💰 현재가: {current_price:,.2f} USDT")
            print("-" * 25)
            print(f"🤖 AI 예측\t: {ai_text}")
            print(f"📊 MACD 신호\t: {macd_text}")
            print(f"🎯 하이브리드\t: {hybrid_text}")
            print(f"💪 권장 포지션\t: {position_text}")
            print("="*25)
            print(f"(다음 업데이트까지 {config.REFRESH_INTERVAL_SECONDS}초 대기...)")
            
            iteration += 1
            time.sleep(config.REFRESH_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n실시간 모니터링을 종료합니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")

def main():
    """메인 실행 함수"""
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    try:
        live_config = LiveConfig()
        monitor_real_time(live_config)
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    # ccxt 설치 안내
    try:
        import ccxt
    except ImportError:
        print("실시간 데이터 수집을 위해 'ccxt' 라이브러리가 필요합니다.")
        print("터미널에 'pip install ccxt'를 입력하여 설치해주세요.")
        exit()
        
    main() 