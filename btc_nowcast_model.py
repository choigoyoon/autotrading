"""
BTC 나우캐스트 모델
LSTM 기반 시계열 예측 모델로 다음 15분 가격을 예측합니다.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os


class BTCNowcastModel:
    """BTC 나우캐스트 LSTM 모델"""

    def __init__(self, sequence_length=24, prediction_horizon=1):
        """
        Args:
            sequence_length: 입력 시퀀스 길이 (24 = 6시간, 15분 단위)
            prediction_horizon: 예측 미래 시점 (1 = 다음 15분)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None

    def prepare_sequences(self, df, feature_columns, target_column='close'):
        """
        시계열 데이터를 LSTM 입력 형태로 변환

        Args:
            df: 특징이 포함된 DataFrame
            feature_columns: 특징 컬럼 리스트
            target_column: 예측 대상 컬럼

        Returns:
            X, y (numpy arrays)
        """
        self.feature_columns = feature_columns

        # NaN 제거
        df = df.dropna()

        # 특징과 타겟 분리
        features = df[feature_columns].values
        target = df[target_column].values

        # 스케일링
        features_scaled = self.scaler.fit_transform(features)

        X, y = [], []

        for i in range(len(features_scaled) - self.sequence_length - self.prediction_horizon + 1):
            X.append(features_scaled[i:i + self.sequence_length])
            # 다음 15분 가격 변화율 예측
            current_price = target[i + self.sequence_length - 1]
            future_price = target[i + self.sequence_length + self.prediction_horizon - 1]
            price_change = (future_price - current_price) / current_price
            y.append(price_change)

        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """
        LSTM 모델 구축

        Args:
            input_shape: (sequence_length, num_features)
        """
        model = keras.Sequential([
            # LSTM 레이어 1
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.BatchNormalization(),

            # LSTM 레이어 2
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.BatchNormalization(),

            # LSTM 레이어 3
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.BatchNormalization(),

            # Dense 레이어
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),

            # 출력 레이어 (가격 변화율)
            layers.Dense(1, activation='linear')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        return model

    def train(self, df, feature_columns, epochs=50, batch_size=32, validation_split=0.2):
        """
        모델 훈련

        Args:
            df: 학습 데이터
            feature_columns: 특징 컬럼 리스트
            epochs: 훈련 에폭 수
            batch_size: 배치 크기
            validation_split: 검증 데이터 비율

        Returns:
            훈련 히스토리
        """
        print("시퀀스 준비 중...")
        X, y = self.prepare_sequences(df, feature_columns)

        print(f"데이터 shape: X={X.shape}, y={y.shape}")

        # 모델 빌드
        if self.model is None:
            print("모델 빌드 중...")
            self.build_model(input_shape=(self.sequence_length, len(feature_columns)))

        print(f"\n모델 구조:")
        self.model.summary()

        # 콜백 설정
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]

        # 훈련
        print(f"\n모델 훈련 시작...")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def predict(self, df, feature_columns):
        """
        예측 수행

        Args:
            df: 예측할 데이터 (최소 sequence_length 개의 행 필요)
            feature_columns: 특징 컬럼 리스트

        Returns:
            예측된 가격 변화율
        """
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")

        # 마지막 sequence_length 개의 데이터 사용
        df_recent = df.tail(self.sequence_length).copy()

        # NaN 확인
        if df_recent[feature_columns].isna().any().any():
            raise ValueError("입력 데이터에 NaN이 포함되어 있습니다.")

        # 특징 추출 및 스케일링
        features = df_recent[feature_columns].values
        features_scaled = self.scaler.transform(features)

        # 시퀀스 생성 (배치 차원 추가)
        X = features_scaled.reshape(1, self.sequence_length, len(feature_columns))

        # 예측
        prediction = self.model.predict(X, verbose=0)

        return prediction[0][0]

    def predict_price(self, df, feature_columns):
        """
        실제 가격 예측

        Args:
            df: 예측할 데이터
            feature_columns: 특징 컬럼 리스트

        Returns:
            예측된 가격
        """
        current_price = df['close'].iloc[-1]
        price_change = self.predict(df, feature_columns)
        predicted_price = current_price * (1 + price_change)

        return predicted_price, price_change

    def save(self, model_path='models/btc_nowcast'):
        """모델 저장"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # 모델 저장
        self.model.save(f"{model_path}_model.keras")

        # 스케일러 저장
        joblib.dump(self.scaler, f"{model_path}_scaler.pkl")

        # 메타데이터 저장
        metadata = {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'feature_columns': self.feature_columns
        }
        joblib.dump(metadata, f"{model_path}_metadata.pkl")

        print(f"모델 저장 완료: {model_path}")

    def load(self, model_path='models/btc_nowcast'):
        """모델 로드"""
        # 모델 로드
        self.model = keras.models.load_model(f"{model_path}_model.keras")

        # 스케일러 로드
        self.scaler = joblib.load(f"{model_path}_scaler.pkl")

        # 메타데이터 로드
        metadata = joblib.load(f"{model_path}_metadata.pkl")
        self.sequence_length = metadata['sequence_length']
        self.prediction_horizon = metadata['prediction_horizon']
        self.feature_columns = metadata['feature_columns']

        print(f"모델 로드 완료: {model_path}")

    def evaluate(self, df, feature_columns):
        """
        모델 평가

        Args:
            df: 평가 데이터
            feature_columns: 특징 컬럼 리스트

        Returns:
            평가 메트릭
        """
        X, y = self.prepare_sequences(df, feature_columns)
        results = self.model.evaluate(X, y, verbose=0)

        metrics = {
            'loss': results[0],
            'mae': results[1]
        }

        return metrics


def backtest(model, df, feature_columns, initial_balance=10000):
    """
    간단한 백테스팅

    Args:
        model: 훈련된 모델
        df: 백테스트 데이터
        feature_columns: 특징 컬럼 리스트
        initial_balance: 초기 자본

    Returns:
        백테스트 결과
    """
    balance = initial_balance
    position = 0  # 0: 노포지션, 1: 롱 포지션
    trades = []
    balances = [initial_balance]

    # 시퀀스 길이 이후부터 시작
    for i in range(model.sequence_length, len(df) - 1):
        # 현재까지의 데이터로 예측
        current_df = df.iloc[:i+1]

        try:
            predicted_price, price_change = model.predict_price(current_df, feature_columns)
            current_price = df['close'].iloc[i]
            next_price = df['close'].iloc[i+1]

            # 간단한 전략: 상승 예측 시 매수, 하락 예측 시 매도
            if price_change > 0.001 and position == 0:  # 0.1% 이상 상승 예측 시 매수
                position = 1
                entry_price = next_price
                trades.append({
                    'type': 'BUY',
                    'price': entry_price,
                    'time': df['datetime'].iloc[i+1] if 'datetime' in df.columns else i+1
                })

            elif price_change < -0.001 and position == 1:  # 0.1% 이상 하락 예측 시 매도
                position = 0
                exit_price = next_price
                pnl = (exit_price - entry_price) / entry_price
                balance *= (1 + pnl)
                balances.append(balance)

                trades.append({
                    'type': 'SELL',
                    'price': exit_price,
                    'pnl': pnl,
                    'balance': balance,
                    'time': df['datetime'].iloc[i+1] if 'datetime' in df.columns else i+1
                })

        except Exception as e:
            continue

    # 최종 수익률 계산
    final_return = (balance - initial_balance) / initial_balance

    results = {
        'initial_balance': initial_balance,
        'final_balance': balance,
        'total_return': final_return,
        'num_trades': len([t for t in trades if t['type'] == 'SELL']),
        'trades': trades,
        'balance_history': balances
    }

    return results


if __name__ == "__main__":
    print("BTC 나우캐스트 모델 모듈")
    print("실제 사용 예시는 train_nowcast.py를 참조하세요.")
