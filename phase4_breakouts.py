"""
Phase 4: 돌파 이벤트 탐지
추세선 돌파 및 H/L 크로스 이벤트 탐지
"""

import pandas as pd
import numpy as np
from phase1_hl_labeling import calculate_macd, generate_hl_labels, get_labeled_points
from phase2_trendlines import generate_trendlines, get_trendline_price


def detect_trendline_breakouts(df, trendlines_df, volume_window=20):
    """
    추세선 돌파 탐지

    Args:
        df: 원본 DataFrame
        trendlines_df: 추세선 DataFrame
        volume_window: 평균 거래량 계산 윈도우

    Returns:
        breakouts list
    """
    breakouts = []

    # 평균 거래량 계산
    df['avg_volume'] = df['volume'].rolling(volume_window).mean()

    for idx, trendline in trendlines_df.iterrows():
        start_idx = trendline['start_idx']
        end_idx = trendline['end_idx']
        tline_type = trendline['type']

        # 추세선 이후 데이터에서 돌파 탐지 (end_idx 이후 100 캔들까지)
        search_end = min(end_idx + 100, len(df) - 1)

        for i in range(end_idx, search_end):
            # 추세선 가격 계산
            trendline_price = get_trendline_price(trendline, i)

            current_close = df.iloc[i]['close']
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']

            # 하락추세선 돌파 (상승 돌파)
            if tline_type == 'down' and current_close > trendline_price:
                # 이전 봉이 추세선 아래였는지 확인
                if i > 0:
                    prev_close = df.iloc[i - 1]['close']
                    prev_trendline = get_trendline_price(trendline, i - 1)

                    if prev_close <= prev_trendline:  # 이전 봉은 아래
                        candle_size = (current_high - current_low) / current_close * 100
                        volume_ratio = df.iloc[i]['volume'] / df.iloc[i]['avg_volume'] if df.iloc[i]['avg_volume'] > 0 else 1

                        breakouts.append({
                            'type': 'trendline_up',
                            'break_idx': i,
                            'break_price': current_close,
                            'reference_price': trendline_price,
                            'trendline_idx': idx,
                            'candle_size': candle_size,
                            'volume_ratio': volume_ratio,
                            'support_confirmed': False  # 나중에 확인
                        })

                        break  # 이 추세선에서는 첫 돌파만

            # 상승추세선 돌파 (하락 돌파)
            elif tline_type == 'up' and current_close < trendline_price:
                if i > 0:
                    prev_close = df.iloc[i - 1]['close']
                    prev_trendline = get_trendline_price(trendline, i - 1)

                    if prev_close >= prev_trendline:  # 이전 봉은 위
                        candle_size = (current_high - current_low) / current_close * 100
                        volume_ratio = df.iloc[i]['volume'] / df.iloc[i]['avg_volume'] if df.iloc[i]['avg_volume'] > 0 else 1

                        breakouts.append({
                            'type': 'trendline_down',
                            'break_idx': i,
                            'break_price': current_close,
                            'reference_price': trendline_price,
                            'trendline_idx': idx,
                            'candle_size': candle_size,
                            'volume_ratio': volume_ratio,
                            'support_confirmed': False
                        })

                        break

    return breakouts


def detect_hl_crosses(df, lookback=10):
    """
    H/L 크로스 탐지 (저항→지지, 지지→저항 전환)

    Args:
        df: 라벨링된 DataFrame
        lookback: 되돌림 확인 범위

    Returns:
        breakouts list
    """
    breakouts = []
    labeled = get_labeled_points(df)

    # H 라벨 크로스 (저항 돌파 → 지지 전환)
    h_labels = labeled[labeled['label'] == 'H']

    for h_idx, h_row in h_labels.iterrows():
        h_price = h_row['label_price']

        # H 이후 데이터에서 돌파 탐지 (최대 100 캔들)
        search_start = h_idx + 1
        search_end = min(h_idx + 100, len(df))

        for i in range(search_start, search_end):
            current_close = df.iloc[i]['close']
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']

            # H 돌파 (close > H price)
            if current_close > h_price:
                # 이전 봉이 H 아래였는지 확인
                if i > 0 and df.iloc[i - 1]['close'] <= h_price:
                    candle_size = (current_high - current_low) / current_close * 100
                    volume_ratio = df.iloc[i]['volume'] / df.iloc[i - lookback:i]['volume'].mean() if i >= lookback else 1

                    # 되돌림에서 지지 확인 (돌파 후 lookback 캔들 내)
                    support_confirmed = False
                    retest_end = min(i + lookback, len(df))

                    for j in range(i + 1, retest_end):
                        if df.iloc[j]['low'] <= h_price * 1.005 and df.iloc[j]['low'] > h_price * 0.995:
                            # H 근처 되돌림 발생
                            if df.iloc[j]['close'] > h_price:
                                support_confirmed = True
                                break

                    breakouts.append({
                        'type': 'hl_cross_long',
                        'break_idx': i,
                        'break_price': current_close,
                        'reference_price': h_price,
                        'trendline_idx': -1,
                        'candle_size': candle_size,
                        'volume_ratio': volume_ratio,
                        'support_confirmed': support_confirmed
                    })

                    break

    # L 라벨 크로스 (지지 이탈 → 저항 전환)
    l_labels = labeled[labeled['label'] == 'L']

    for l_idx, l_row in l_labels.iterrows():
        l_price = l_row['label_price']

        search_start = l_idx + 1
        search_end = min(l_idx + 100, len(df))

        for i in range(search_start, search_end):
            current_close = df.iloc[i]['close']
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']

            # L 이탈 (close < L price)
            if current_close < l_price:
                if i > 0 and df.iloc[i - 1]['close'] >= l_price:
                    candle_size = (current_high - current_low) / current_close * 100
                    volume_ratio = df.iloc[i]['volume'] / df.iloc[i - lookback:i]['volume'].mean() if i >= lookback else 1

                    # 되돌림에서 저항 확인
                    support_confirmed = False  # 여기서는 resistance confirmed 의미
                    retest_end = min(i + lookback, len(df))

                    for j in range(i + 1, retest_end):
                        if df.iloc[j]['high'] >= l_price * 0.995 and df.iloc[j]['high'] < l_price * 1.005:
                            if df.iloc[j]['close'] < l_price:
                                support_confirmed = True
                                break

                    breakouts.append({
                        'type': 'hl_cross_short',
                        'break_idx': i,
                        'break_price': current_close,
                        'reference_price': l_price,
                        'trendline_idx': -1,
                        'candle_size': candle_size,
                        'volume_ratio': volume_ratio,
                        'support_confirmed': support_confirmed
                    })

                    break

    return breakouts


def detect_breakouts(df, trendlines_df, lookback=10):
    """
    모든 돌파 이벤트 통합 탐지

    Args:
        df: 원본 DataFrame
        trendlines_df: 추세선 DataFrame
        lookback: 되돌림 확인 범위

    Returns:
        breakouts DataFrame
    """
    print("  추세선 돌파 탐지 중...")
    trendline_breakouts = detect_trendline_breakouts(df, trendlines_df)

    print("  H/L 크로스 탐지 중...")
    hl_breakouts = detect_hl_crosses(df, lookback)

    # 통합
    all_breakouts = trendline_breakouts + hl_breakouts

    breakouts_df = pd.DataFrame(all_breakouts)

    if len(breakouts_df) > 0:
        # break_idx 기준 정렬
        breakouts_df = breakouts_df.sort_values('break_idx').reset_index(drop=True)

    return breakouts_df


if __name__ == "__main__":
    import glob

    # Phase 1, 2 결과 로드
    if glob.glob("output_phase1_labeled.csv") and glob.glob("output_phase2_trendlines.csv"):
        print("Phase 1, 2 결과 로드 중...")
        df = pd.read_csv("output_phase1_labeled.csv")
        df['datetime'] = pd.to_datetime(df['datetime'])

        trendlines_df = pd.read_csv("output_phase2_trendlines.csv")
    else:
        print("Phase 1, 2 실행 필요")
        exit(1)

    print(f"데이터 크기: {len(df)} 캔들")
    print(f"추세선: {len(trendlines_df)}개\n")

    # 돌파 탐지
    print("돌파 이벤트 탐지 중...")
    breakouts_df = detect_breakouts(df, trendlines_df, lookback=10)

    print(f"\n총 돌파: {len(breakouts_df)}개")

    if len(breakouts_df) > 0:
        breakout_types = breakouts_df['type'].value_counts()
        print("\n돌파 유형별:")
        for b_type, count in breakout_types.items():
            print(f"  {b_type}: {count}개")

        # 통계
        print(f"\n돌파 통계:")
        print(f"  평균 캔들 크기: {breakouts_df['candle_size'].mean():.2f}%")
        print(f"  평균 거래량 비율: {breakouts_df['volume_ratio'].mean():.2f}x")
        print(f"  지지/저항 확인율: {breakouts_df['support_confirmed'].sum() / len(breakouts_df) * 100:.1f}%")

        # 샘플 출력
        print("\n최근 10개 돌파:")
        print(breakouts_df[['break_idx', 'type', 'candle_size', 'volume_ratio', 'support_confirmed']].tail(10).to_string(index=False))

    # 저장
    output_path = "output_phase4_breakouts.csv"
    breakouts_df.to_csv(output_path, index=False)
    print(f"\n결과 저장: {output_path}")
