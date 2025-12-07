"""
Phase 1: H/L 라벨링
MACD 히스토그램 기반 H/L 라벨 생성
"""

import pandas as pd
import numpy as np


def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    MACD 계산

    Args:
        df: OHLCV DataFrame
        fast: 빠른 EMA 기간
        slow: 느린 EMA 기간
        signal: 시그널 EMA 기간

    Returns:
        DataFrame with MACD columns
    """
    df = df.copy()

    # EMA 계산
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

    # MACD
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    return df


def generate_hl_labels(df):
    """
    H/L 라벨 생성

    규칙:
    - H: 히스토그램 양수→음수 전환 시, 이전 양수 구간의 price high 최고점
    - L: 히스토그램 음수→양수 전환 시, 이전 음수 구간의 price low 최저점

    Args:
        df: MACD 계산된 DataFrame

    Returns:
        DataFrame with label columns
    """
    df = df.copy()

    # 라벨 컬럼 초기화
    df['label'] = None
    df['label_price'] = np.nan
    df['label_macd'] = np.nan
    df['label_idx'] = np.nan

    # 히스토그램 부호 변화 탐지
    hist = df['macd_hist'].values

    # 양수/음수 구간 추적
    current_sign = None  # 1: 양수, -1: 음수
    segment_start = 0

    for i in range(len(df)):
        if pd.isna(hist[i]):
            continue

        sign = 1 if hist[i] >= 0 else -1

        # 부호 변화 감지
        if current_sign is not None and sign != current_sign:
            # 이전 구간 분석
            segment_df = df.iloc[segment_start:i]

            if current_sign == 1:  # 양수→음수: H 라벨
                # 이전 양수 구간의 high 최고점 찾기
                max_idx = segment_df['high'].idxmax()
                max_price = segment_df.loc[max_idx, 'high']
                max_macd = segment_df.loc[max_idx, 'macd_hist']

                # 전환 시점(i)에 라벨 확정
                df.loc[df.index[i], 'label'] = 'H'
                df.loc[df.index[i], 'label_price'] = max_price
                df.loc[df.index[i], 'label_macd'] = max_macd
                df.loc[df.index[i], 'label_idx'] = max_idx

            elif current_sign == -1:  # 음수→양수: L 라벨
                # 이전 음수 구간의 low 최저점 찾기
                min_idx = segment_df['low'].idxmin()
                min_price = segment_df.loc[min_idx, 'low']
                min_macd = segment_df.loc[min_idx, 'macd_hist']

                # 전환 시점(i)에 라벨 확정
                df.loc[df.index[i], 'label'] = 'L'
                df.loc[df.index[i], 'label_price'] = min_price
                df.loc[df.index[i], 'label_macd'] = min_macd
                df.loc[df.index[i], 'label_idx'] = min_idx

            # 새 구간 시작
            segment_start = i

        current_sign = sign

    return df


def get_labeled_points(df):
    """
    라벨링된 포인트만 추출

    Args:
        df: 라벨링된 DataFrame

    Returns:
        DataFrame with only labeled points
    """
    labeled = df[df['label'].notna()].copy()
    return labeled


if __name__ == "__main__":
    import glob

    # 최신 BTC 데이터 로드
    data_files = glob.glob("data/BTC_USDT_USDT_15m_*.csv")

    if not data_files:
        print("BTC 데이터 파일을 찾을 수 없습니다.")
    else:
        latest_file = max(data_files, key=lambda x: x.split('_')[-1])

        print(f"데이터 로드: {latest_file}")
        df = pd.read_csv(latest_file)

        # datetime 컬럼 변환
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.rename(columns={'timestamp': 'datetime'}, inplace=True)

        print(f"데이터 크기: {len(df)} 캔들")
        print(f"기간: {df['datetime'].min()} ~ {df['datetime'].max()}\n")

        # MACD 계산
        print("MACD 계산 중...")
        df = calculate_macd(df)

        # H/L 라벨링
        print("H/L 라벨링 중...")
        df = generate_hl_labels(df)

        # 결과 출력
        labeled = get_labeled_points(df)

        print(f"\n총 라벨: {len(labeled)}개")
        print(f"  H: {len(labeled[labeled['label'] == 'H'])}개")
        print(f"  L: {len(labeled[labeled['label'] == 'L'])}개")

        # 샘플 출력
        print("\n최근 10개 라벨:")
        print(labeled[['datetime', 'label', 'label_price', 'label_macd']].tail(10).to_string(index=False))

        # 저장
        output_path = "output_phase1_labeled.csv"
        df.to_csv(output_path, index=False)
        print(f"\n결과 저장: {output_path}")
