import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import time
import ccxt
import warnings
import sys
import os
from ta.trend import macd_diff
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict, Any, List
import numpy.typing as npt
from torch import nn

# --- 프로젝트 설정 ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.ml.trading_transformer import TradingTransformer

warnings.filterwarnings('ignore', category=UserWarning)

# --- 설정 ---
class LiveConfig:
    """라이브 예측을 위한 설정 클래스"""
    # 경로
    MODEL_DIR: Path = project_root / "models"
    DATA_SOURCE_PATH: Path = project_root / 'data' / 'processed' / 'btc_usdt_kst' / 'resampled_ohlcv' / '1min.parquet'
    
    # 모델 및 스케일러
    MODEL_NAME: str = "trading_transformer_v1.pth"
    SCALER_NAME: str = "scaler.joblib"
    
    @property
    def MODEL_PATH(self) -> Path:
        return self.MODEL_DIR / self.MODEL_NAME
    
    @property
    def SCALER_PATH(self) -> Path:
        # 스케일러는 보통 시퀀스 데이터와 함께 저장됨
        return project_root / "data" / "sequences" / self.SCALER_NAME

    # 모델 하이퍼파라미터 (학습 시점과 동일해야 함)
    INPUT_DIM: int = 8
    D_MODEL: int = 128
    N_HEAD: int = 8
    NUM_ENCODER_LAYERS: int = 4
    DIM_FEEDFORWARD: int = 512
    DROPOUT: float = 0.1
    NUM_CLASSES: int = 2  # 0: SELL, 1: BUY

    # 예측 설정
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEQUENCE_LENGTH: int = 240
    CONFIDENCE_THRESHOLD: float = 0.80
    SYMBOL: str = 'BTC/USDT'
    TIMEFRAME: str = '1m'
    REFRESH_INTERVAL_SECONDS: int = 60

# --- AI 예측기 클래스 ---
class AIPredictor:
    """AI 모델 로딩, 데이터 전처리 및 예측을 담당하는 클래스"""
    def __init__(self, config: LiveConfig):
        self.config = config
        self.model = self._load_model()
        self.scaler = self._load_scaler()

    def _load_model(self) -> TradingTransformer:
        """학습된 PyTorch 모델을 로드합니다."""
        model = TradingTransformer(
            input_dim=self.config.INPUT_DIM,
            d_model=self.config.D_MODEL,
            nhead=self.config.N_HEAD,
            num_encoder_layers=self.config.NUM_ENCODER_LAYERS,
            dim_feedforward=self.config.DIM_FEEDFORWARD,
            num_classes=self.config.NUM_CLASSES,
            dropout=self.config.DROPOUT
        ).to(self.config.DEVICE)

        if not self.config.MODEL_PATH.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.config.MODEL_PATH}")
        
        model.load_state_dict(torch.load(self.config.MODEL_PATH, map_location=self.config.DEVICE))
        model.eval()
        print(f"✅ 모델 로드 완료: {self.config.MODEL_PATH}")
        return model

    def _load_scaler(self) -> StandardScaler:
        """학습 시 사용된 StandardScaler를 로드합니다."""
        if not self.config.SCALER_PATH.exists():
            raise FileNotFoundError(f"스케일러 파일을 찾을 수 없습니다: {self.config.SCALER_PATH}")
        scaler: StandardScaler = joblib.load(self.config.SCALER_PATH)
        print(f"✅ 스케일러 로드 완료: {self.config.SCALER_PATH}")
        return scaler

    def get_latest_sequence_from_exchange(self) -> Optional[pd.DataFrame]:
        """거래소에서 최신 시퀀스 데이터를 가져옵니다."""
        try:
            exchange = ccxt.binance()
            ohlcv = exchange.fetch_ohlcv(self.config.SYMBOL, self.config.TIMEFRAME, limit=self.config.SEQUENCE_LENGTH + 50)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            df = df.iloc[-self.config.SEQUENCE_LENGTH:]
            
            if len(df) < self.config.SEQUENCE_LENGTH:
                print(f"⚠️ 데이터 부족: {len(df)}/{self.config.SEQUENCE_LENGTH}개 수집됨.")
                return None
            return df
        except Exception as e:
            print(f"❌ 데이터 수집 실패: {e}")
            return None

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """모델 입력에 사용할 피처를 생성합니다."""
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macdsignal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macdhist'] = df['macd'] - df['macdsignal']
        
        features_to_use = ['open', 'high', 'low', 'close', 'volume', 'macd', 'macdsignal', 'macdhist']
        return df[features_to_use]

    def predict(self, sequence_features: pd.DataFrame) -> Tuple[str, float]:
        """AI 모델로 시그널을 예측합니다."""
        sequence_scaled = self.scaler.transform(sequence_features)
        
        with torch.no_grad():
            sequence_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0).to(self.config.DEVICE)
            signal_logits, _, _ = self.model(sequence_tensor)
            probabilities = F.softmax(signal_logits, dim=1)
            
            prediction_confidence, predicted_class = torch.max(probabilities, dim=1)
            
            signal = "BUY" if predicted_class.item() == 1 else "SELL"
            return signal, prediction_confidence.item()

# --- 실시간 모니터링 클래스 ---
class RealTimeMonitor:
    """실시간 데이터 수집 및 예측 실행 루프를 관리"""
    def __init__(self, config: LiveConfig, predictor: AIPredictor):
        self.config = config
        self.predictor = predictor
    
    def run(self) -> None:
        """실시간 예측 루프를 시작합니다."""
        print("\n" + "="*20 + " 실시간 AI 예측 모니터링 시작 " + "="*20)
        print(f"종목: {self.config.SYMBOL} | 타임프레임: {self.config.TIMEFRAME}")
        print(f"업데이트 간격: {self.config.REFRESH_INTERVAL_SECONDS}초")
        print("="*60)
        
        while True:
            print(f"\n[{pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d %H:%M:%S')}] 최신 데이터로 예측 수행...")
            
            # 1. 데이터 수집
            latest_data = self.predictor.get_latest_sequence_from_exchange()
            if latest_data is None:
                time.sleep(self.config.REFRESH_INTERVAL_SECONDS)
                continue
            
            # 2. 피처 생성
            features = self.predictor.create_features(latest_data)
            
            # 3. AI 예측
            ai_signal, confidence = self.predictor.predict(features)
            
            # 4. MACD 비교 및 결과 출력
            self.display_results(ai_signal, confidence, features.iloc[-1])

            time.sleep(self.config.REFRESH_INTERVAL_SECONDS)

    def display_results(self, ai_signal: str, confidence: float, last_row: pd.Series) -> None:
        """최종 예측 결과를 형식에 맞춰 출력합니다."""
        macd_hist = last_row['macdhist']
        macd_signal = "BUY" if macd_hist > 0 else "SELL"
        agreement = "✅ 동일" if ai_signal == macd_signal else "❌ 불일치"

        print("\n" + "─"*15 + " 최종 예측 결과 " + "─"*15)
        if confidence < self.config.CONFIDENCE_THRESHOLD:
            print(f"📉 [보류] AI 예측 신뢰도 낮음: {confidence:.2%}")
            print(f"   (필터링 기준: > {self.config.CONFIDENCE_THRESHOLD:.0%})")
        else:
            signal_color = "\033[92m" if ai_signal == "BUY" else "\033[91m"
            reset_color = "\033[0m"
            print(f"🤖 AI 예측: {signal_color}{ai_signal}{reset_color} (신뢰도: {confidence:.2%})")
            print(f"🎯 MACD 비교: {macd_signal} (히스토그램: {macd_hist:.4f}) -> {agreement}")
        print("─"*47)

# --- 메인 실행 로직 ---
def main() -> None:
    """메인 실행 함수"""
    try:
        config = LiveConfig()
        predictor = AIPredictor(config)
        monitor = RealTimeMonitor(config, predictor)
        monitor.run()
    except FileNotFoundError as e:
        print(f"\n❌ 초기화 실패: 필수 파일을 찾을 수 없습니다.")
        print(f"   오류: {e}")
        print("   프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}")
        print("   프로그램을 종료합니다.")

if __name__ == "__main__":
    # 실제 실시간 모니터링을 실행하려면 아래 주석을 해제하세요.
    # main()
    
    # --- 테스트용 단일 실행 ---
    def test_single_run():
        print("--- 테스트용 단일 실행 모드 ---")
        try:
            config = LiveConfig()
            predictor = AIPredictor(config)
            
            # 시뮬레이션 데이터 사용
            df_source = pd.read_parquet(config.DATA_SOURCE_PATH)
            test_data = df_source.iloc[-config.SEQUENCE_LENGTH:] # 마지막 240개 데이터 사용
            
            print(f"테스트 데이터 시간: {test_data.index[0]} ~ {test_data.index[-1]}")

            features = predictor.create_features(test_data)
            ai_signal, confidence = predictor.predict(features)
            
            monitor = RealTimeMonitor(config, predictor)
            monitor.display_results(ai_signal, confidence, features.iloc[-1])
            
        except FileNotFoundError as e:
            print(f"❌ 테스트 실패: {e}")
        except Exception as e:
            print(f"❌ 테스트 중 오류 발생: {e}")
            
    test_single_run() 