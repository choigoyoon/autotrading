"""
Phase 3: 다이버전스 탐지
H/L 라벨에서 가격 vs MACD 다이버전스 탐지
"""

import pandas as pd
import numpy as np
from phase1_hl_labeling import calculate_macd, generate_hl_labels, get_labeled_points


def detect_divergence(df, lookback=5):
    """
    가격 vs MACD 다이버전스 탐지

    다이버전스 유형:
    1. Bearish (H 기준): 가격↑ MACD↓ → 숏 신호
    2. Bullish (L 기준): 가격↓ MACD↑ → 롱 신호
    3. Hidden Bearish (H): 가격↓ MACD↑ → 하락 지속
    4. Hidden Bullish (L): 가격↑ MACD↓ → 상승 지속

    Args:
        df: 라벨링된 DataFrame
        lookback: 다이버전스 체크 범위 (최근 N개 라벨)

    Returns:
        DataFrame with divergence columns
    """
    df = df.copy()

    # 다이버전스 컬럼 초기화
    df['divergence_type'] = None
    df['divergence_strength'] = 0
    df['divergence_gap'] = 0.0

    labeled = get_labeled_points(df)

    # H 라벨 다이버전스 (베어리시 & 히든 베어리시)
    h_labels = labeled[labeled['label'] == 'H'].reset_index()

    for i in range(1, len(h_labels)):
        current_idx = h_labels.loc[i, 'index']
        current_price = h_labels.loc[i, 'label_price']
        current_macd = h_labels.loc[i, 'label_macd']

        # 최근 lookback 범위 내에서 이전 H와 비교
        start = max(0, i - lookback)

        for j in range(start, i):
            prev_price = h_labels.loc[j, 'label_price']
            prev_macd = h_labels.loc[j, 'label_macd']

            # 가격 및 MACD 변화
            price_change = (current_price - prev_price) / prev_price * 100
            macd_change = current_macd - prev_macd

            # Bearish Divergence: 가격↑ MACD↓
            if price_change > 0 and macd_change < 0:
                strength = i - j + 1  # 연속 개수
                gap = abs(price_change) + abs(macd_change / prev_macd * 100)

                # 기존 값보다 강한 경우만 업데이트
                if df.loc[current_idx, 'divergence_type'] is None or \
                   df.loc[current_idx, 'divergence_strength'] < strength:
                    df.loc[current_idx, 'divergence_type'] = 'bearish'
                    df.loc[current_idx, 'divergence_strength'] = strength
                    df.loc[current_idx, 'divergence_gap'] = gap

            # Hidden Bearish: 가격↓ MACD↑
            elif price_change < 0 and macd_change > 0:
                strength = i - j + 1
                gap = abs(price_change) + abs(macd_change / prev_macd * 100)

                if df.loc[current_idx, 'divergence_type'] is None or \
                   df.loc[current_idx, 'divergence_strength'] < strength:
                    df.loc[current_idx, 'divergence_type'] = 'hidden_bearish'
                    df.loc[current_idx, 'divergence_strength'] = strength
                    df.loc[current_idx, 'divergence_gap'] = gap

    # L 라벨 다이버전스 (불리시 & 히든 불리시)
    l_labels = labeled[labeled['label'] == 'L'].reset_index()

    for i in range(1, len(l_labels)):
        current_idx = l_labels.loc[i, 'index']
        current_price = l_labels.loc[i, 'label_price']
        current_macd = l_labels.loc[i, 'label_macd']

        # 최근 lookback 범위 내에서 이전 L과 비교
        start = max(0, i - lookback)

        for j in range(start, i):
            prev_price = l_labels.loc[j, 'label_price']
            prev_macd = l_labels.loc[j, 'label_macd']

            # 가격 및 MACD 변화
            price_change = (current_price - prev_price) / prev_price * 100
            macd_change = current_macd - prev_macd

            # Bullish Divergence: 가격↓ MACD↑
            if price_change < 0 and macd_change > 0:
                strength = i - j + 1
                gap = abs(price_change) + abs(macd_change / prev_macd * 100)

                if df.loc[current_idx, 'divergence_type'] is None or \
                   df.loc[current_idx, 'divergence_strength'] < strength:
                    df.loc[current_idx, 'divergence_type'] = 'bullish'
                    df.loc[current_idx, 'divergence_strength'] = strength
                    df.loc[current_idx, 'divergence_gap'] = gap

            # Hidden Bullish: 가격↑ MACD↓
            elif price_change > 0 and macd_change < 0:
                strength = i - j + 1
                gap = abs(price_change) + abs(macd_change / prev_macd * 100)

                if df.loc[current_idx, 'divergence_type'] is None or \
                   df.loc[current_idx, 'divergence_strength'] < strength:
                    df.loc[current_idx, 'divergence_type'] = 'hidden_bullish'
                    df.loc[current_idx, 'divergence_strength'] = strength
                    df.loc[current_idx, 'divergence_gap'] = gap

    return df


if __name__ == "__main__":
    import glob

    # Phase 1 결과 로드
    if glob.glob("output_phase1_labeled.csv"):
        print("Phase 1 결과 로드 중...")
        df = pd.read_csv("output_phase1_labeled.csv")
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        # Phase 1 실행
        print("Phase 1 실행 중...")
        data_files = glob.glob("data/BTC_USDT_USDT_15m_*.csv")
        latest_file = max(data_files, key=lambda x: x.split('_')[-1])

        df = pd.read_csv(latest_file)
        df['datetime'] = pd.to_datetime(df['datetime'])

        df = calculate_macd(df)
        df = generate_hl_labels(df)

    print(f"데이터 크기: {len(df)} 캔들\n")

    # 다이버전스 탐지
    print("다이버전스 탐지 중...")
    df = detect_divergence(df, lookback=5)

    # 결과 출력
    divergence_count = df['divergence_type'].notna().sum()

    print(f"\n총 다이버전스: {divergence_count}개")

    if divergence_count > 0:
        div_types = df['divergence_type'].value_counts()
        print("\n다이버전스 유형별:")
        for div_type, count in div_types.items():
            print(f"  {div_type}: {count}개")

        # 통계
        divergences = df[df['divergence_type'].notna()]
        print(f"\n다이버전스 통계:")
        print(f"  평균 강도: {divergences['divergence_strength'].mean():.1f}")
        print(f"  평균 갭: {divergences['divergence_gap'].mean():.1f}%")

        # 샘플 출력
        print("\n최근 10개 다이버전스:")
        recent_div = divergences[['datetime', 'label', 'divergence_type', 'divergence_strength', 'divergence_gap']].tail(10)
        print(recent_div.to_string(index=False))

    # 저장
    output_path = "output_phase3_divergence.csv"
    df.to_csv(output_path, index=False)
    print(f"\n결과 저장: {output_path}")
