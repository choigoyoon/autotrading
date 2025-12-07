"""
Phase 2: 추세선 생성
H/L 라벨에서 추세선 자동 생성
"""

import pandas as pd
import numpy as np
from phase1_hl_labeling import calculate_macd, generate_hl_labels, get_labeled_points


def generate_trendlines(df, min_touches=2):
    """
    H/L 라벨에서 추세선 생성

    규칙:
    - 하락추세선: 연속 H 라벨에서 price 하락 (H1 > H2 > H3...)
    - 상승추세선: 연속 L 라벨에서 price 상승 (L1 < L2 < L3...)

    Args:
        df: 라벨링된 DataFrame
        min_touches: 최소 터치 횟수 (기본 2)

    Returns:
        trendlines DataFrame
    """
    labeled = get_labeled_points(df)

    trendlines = []

    # H 라벨 추세선 (하락추세)
    h_labels = labeled[labeled['label'] == 'H'].copy()

    for i in range(len(h_labels) - min_touches + 1):
        # 현재 H부터 시작하는 하락 시퀀스 찾기
        sequence = [i]
        current_price = h_labels.iloc[i]['label_price']

        for j in range(i + 1, len(h_labels)):
            next_price = h_labels.iloc[j]['label_price']

            # 하락하면 추가
            if next_price < current_price:
                sequence.append(j)
                current_price = next_price
            # 상승하면 시퀀스 종료
            elif next_price > current_price * 1.01:  # 1% 이상 상승 시 종료
                break

        # min_touches 이상이면 추세선 생성
        if len(sequence) >= min_touches:
            start_idx = h_labels.index[sequence[0]]
            end_idx = h_labels.index[sequence[-1]]

            start_price = h_labels.iloc[sequence[0]]['label_price']
            end_price = h_labels.iloc[sequence[-1]]['label_price']

            # 기울기 계산 (price change per bar)
            duration = end_idx - start_idx
            slope = (end_price - start_price) / duration if duration > 0 else 0

            trendlines.append({
                'type': 'down',
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_price': start_price,
                'end_price': end_price,
                'slope': slope,
                'touch_count': len(sequence),
                'duration': duration
            })

    # L 라벨 추세선 (상승추세)
    l_labels = labeled[labeled['label'] == 'L'].copy()

    for i in range(len(l_labels) - min_touches + 1):
        # 현재 L부터 시작하는 상승 시퀀스 찾기
        sequence = [i]
        current_price = l_labels.iloc[i]['label_price']

        for j in range(i + 1, len(l_labels)):
            next_price = l_labels.iloc[j]['label_price']

            # 상승하면 추가
            if next_price > current_price:
                sequence.append(j)
                current_price = next_price
            # 하락하면 시퀀스 종료
            elif next_price < current_price * 0.99:  # 1% 이상 하락 시 종료
                break

        # min_touches 이상이면 추세선 생성
        if len(sequence) >= min_touches:
            start_idx = l_labels.index[sequence[0]]
            end_idx = l_labels.index[sequence[-1]]

            start_price = l_labels.iloc[sequence[0]]['label_price']
            end_price = l_labels.iloc[sequence[-1]]['label_price']

            # 기울기 계산
            duration = end_idx - start_idx
            slope = (end_price - start_price) / duration if duration > 0 else 0

            trendlines.append({
                'type': 'up',
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_price': start_price,
                'end_price': end_price,
                'slope': slope,
                'touch_count': len(sequence),
                'duration': duration
            })

    trendlines_df = pd.DataFrame(trendlines)

    # start_idx 기준 정렬
    if len(trendlines_df) > 0:
        trendlines_df = trendlines_df.sort_values('start_idx').reset_index(drop=True)

    return trendlines_df


def get_trendline_price(trendline, current_idx):
    """
    특정 시점의 추세선 가격 계산 (무한 연장)

    Args:
        trendline: trendline row (Series)
        current_idx: 현재 인덱스

    Returns:
        추세선 가격
    """
    start_idx = trendline['start_idx']
    start_price = trendline['start_price']
    slope = trendline['slope']

    # 추세선 가격 = start_price + slope * (current_idx - start_idx)
    price = start_price + slope * (current_idx - start_idx)

    return price


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

    # 추세선 생성
    print("추세선 생성 중...")
    trendlines_df = generate_trendlines(df, min_touches=2)

    print(f"\n총 추세선: {len(trendlines_df)}개")
    print(f"  하락추세선: {len(trendlines_df[trendlines_df['type'] == 'down'])}개")
    print(f"  상승추세선: {len(trendlines_df[trendlines_df['type'] == 'up'])}개")

    # 통계
    if len(trendlines_df) > 0:
        print(f"\n추세선 통계:")
        print(f"  평균 터치 횟수: {trendlines_df['touch_count'].mean():.1f}")
        print(f"  평균 지속 기간: {trendlines_df['duration'].mean():.0f} 캔들")
        print(f"  최대 터치 횟수: {trendlines_df['touch_count'].max()}")
        print(f"  최대 지속 기간: {trendlines_df['duration'].max()} 캔들")

        # 샘플 출력 (최근 10개)
        print("\n최근 10개 추세선:")
        print(trendlines_df[['type', 'start_idx', 'touch_count', 'duration', 'slope']].tail(10).to_string(index=False))

    # 저장
    output_path = "output_phase2_trendlines.csv"
    trendlines_df.to_csv(output_path, index=False)
    print(f"\n결과 저장: {output_path}")
