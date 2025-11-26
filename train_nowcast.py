"""
BTC 나우캐스트 모델 훈련 스크립트
수집된 데이터를 사용하여 LSTM 모델을 훈련합니다.
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from technical_indicators import calculate_indicators, TechnicalIndicators
from btc_nowcast_model import BTCNowcastModel, backtest


def load_latest_data(symbol='BTC_USDT_USDT', data_dir='data'):
    """
    최신 데이터 파일 로드

    Args:
        symbol: 심볼 이름
        data_dir: 데이터 디렉토리

    Returns:
        DataFrame
    """
    pattern = os.path.join(data_dir, f"{symbol}_*.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"{symbol} 데이터 파일을 찾을 수 없습니다.")

    # 가장 최근 파일 선택
    latest_file = max(files, key=os.path.getmtime)
    print(f"데이터 로드: {latest_file}")

    df = pd.read_csv(latest_file)

    # datetime 컬럼이 있으면 파싱
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"데이터 shape: {df.shape}")
    print(f"기간: {df['datetime'].min()} ~ {df['datetime'].max()}")

    return df


def prepare_data(df, test_size=0.2):
    """
    데이터 준비 (지표 추가 및 분할)

    Args:
        df: 원본 데이터
        test_size: 테스트 데이터 비율

    Returns:
        train_df, test_df
    """
    print("\n기술적 지표 계산 중...")
    df_with_indicators = calculate_indicators(df)

    # NaN 제거
    df_clean = df_with_indicators.dropna()
    print(f"NaN 제거 후 shape: {df_clean.shape}")

    # 시간순 분할 (과거 데이터로 학습, 최근 데이터로 테스트)
    split_idx = int(len(df_clean) * (1 - test_size))

    train_df = df_clean.iloc[:split_idx].copy()
    test_df = df_clean.iloc[split_idx:].copy()

    print(f"\n학습 데이터: {len(train_df)} rows")
    print(f"테스트 데이터: {len(test_df)} rows")

    return train_df, test_df


def plot_training_history(history):
    """훈련 히스토리 시각화"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Training and Validation MAE')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    print("\n훈련 히스토리 저장: models/training_history.png")
    plt.close()


def plot_backtest_results(results):
    """백테스트 결과 시각화"""
    plt.figure(figsize=(12, 6))

    plt.plot(results['balance_history'])
    plt.axhline(y=results['initial_balance'], color='r', linestyle='--', label='Initial Balance')
    plt.xlabel('Trade Number')
    plt.ylabel('Balance ($)')
    plt.title(f'Backtest Results - Total Return: {results["total_return"]*100:.2f}%')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('models/backtest_results.png', dpi=300, bbox_inches='tight')
    print("백테스트 결과 저장: models/backtest_results.png")
    plt.close()


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("BTC 나우캐스트 모델 훈련")
    print("=" * 60)

    # 1. 데이터 로드
    try:
        df = load_latest_data('BTC_USDT_USDT')
    except FileNotFoundError as e:
        print(f"\n에러: {e}")
        print("먼저 bybit_collector_ccxt.py를 실행하여 데이터를 수집하세요.")
        return

    # 2. 데이터 준비
    train_df, test_df = prepare_data(df, test_size=0.2)

    # 3. 특징 컬럼 선택
    ti = TechnicalIndicators(train_df)
    feature_columns = ti.get_feature_columns()
    print(f"\n사용할 특징 수: {len(feature_columns)}")

    # 4. 모델 생성 및 훈련
    print("\n" + "=" * 60)
    print("모델 훈련 시작")
    print("=" * 60)

    model = BTCNowcastModel(
        sequence_length=24,  # 6시간 (15분 * 24)
        prediction_horizon=1  # 다음 15분
    )

    history = model.train(
        df=train_df,
        feature_columns=feature_columns,
        epochs=100,
        batch_size=64,
        validation_split=0.2
    )

    # 5. 훈련 히스토리 시각화
    plot_training_history(history)

    # 6. 테스트 데이터 평가
    print("\n" + "=" * 60)
    print("모델 평가")
    print("=" * 60)

    metrics = model.evaluate(test_df, feature_columns)
    print(f"테스트 Loss: {metrics['loss']:.6f}")
    print(f"테스트 MAE: {metrics['mae']:.6f}")

    # 7. 백테스트
    print("\n" + "=" * 60)
    print("백테스트 실행")
    print("=" * 60)

    backtest_results = backtest(
        model=model,
        df=test_df,
        feature_columns=feature_columns,
        initial_balance=10000
    )

    print(f"\n백테스트 결과:")
    print(f"  초기 자본: ${backtest_results['initial_balance']:,.2f}")
    print(f"  최종 자본: ${backtest_results['final_balance']:,.2f}")
    print(f"  총 수익률: {backtest_results['total_return']*100:.2f}%")
    print(f"  거래 횟수: {backtest_results['num_trades']}")

    # 백테스트 결과 시각화
    if len(backtest_results['balance_history']) > 1:
        plot_backtest_results(backtest_results)

    # 8. 모델 저장
    print("\n" + "=" * 60)
    print("모델 저장")
    print("=" * 60)

    model.save('models/btc_nowcast')

    # 9. 예측 예시
    print("\n" + "=" * 60)
    print("예측 예시 (최근 데이터)")
    print("=" * 60)

    current_price = test_df['close'].iloc[-1]
    predicted_price, price_change = model.predict_price(test_df, feature_columns)

    print(f"현재 가격: ${current_price:,.2f}")
    print(f"예측 가격 (다음 15분): ${predicted_price:,.2f}")
    print(f"예상 변화율: {price_change*100:.2f}%")

    print("\n" + "=" * 60)
    print("훈련 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
