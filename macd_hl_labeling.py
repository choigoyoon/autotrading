"""
MACD 기반 High/Low 라벨링 시스템
MACD 히스토그램의 부호 전환 지점을 기반으로 H/L을 탐지합니다.
"""

import pandas as pd
import numpy as np


class MACDHighLowLabeler:
    """MACD 히스토그램 기반 H/L 라벨링"""

    def __init__(self, fast=12, slow=26, signal=9):
        """
        Args:
            fast: MACD 빠른 EMA 기간
            slow: MACD 느린 EMA 기간
            signal: MACD 시그널 기간
        """
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def calculate_macd(self, df):
        """
        MACD 계산 (pandas로 직접 구현)

        Args:
            df: OHLCV DataFrame

        Returns:
            MACD 지표가 추가된 DataFrame
        """
        df = df.copy()

        # EMA 계산
        ema_fast = df['close'].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.slow, adjust=False).mean()

        # MACD 라인
        df['macd'] = ema_fast - ema_slow

        # 시그널 라인
        df['macd_signal'] = df['macd'].ewm(span=self.signal, adjust=False).mean()

        # 히스토그램
        df['macd_hist'] = df['macd'] - df['macd_signal']

        return df

    def find_histogram_crossovers(self, df):
        """
        히스토그램 부호 전환 지점 탐지

        Args:
            df: MACD가 계산된 DataFrame

        Returns:
            전환 지점 리스트 [{index, type, from_sign, to_sign}, ...]
        """
        if 'macd_hist' not in df.columns:
            df = self.calculate_macd(df)

        crossovers = []

        # NaN이 아닌 값만 사용
        hist = df['macd_hist'].dropna()

        for i in range(1, len(hist)):
            prev_sign = np.sign(hist.iloc[i-1])
            curr_sign = np.sign(hist.iloc[i])

            # 부호 전환 감지
            if prev_sign != curr_sign and prev_sign != 0 and curr_sign != 0:
                crossover_type = 'pos_to_neg' if prev_sign > 0 else 'neg_to_pos'

                crossovers.append({
                    'index': hist.index[i],
                    'type': crossover_type,
                    'from_sign': prev_sign,
                    'to_sign': curr_sign,
                    'timestamp': df.loc[hist.index[i], 'datetime'] if 'datetime' in df.columns else hist.index[i]
                })

        return crossovers

    def label_highs_lows(self, df):
        """
        H/L 라벨링

        로직:
        - 양수→음수 전환: 직전 양수 구간의 high 최댓값 = H
        - 음수→양수 전환: 직전 음수 구간의 low 최솟값 = L

        Args:
            df: OHLCV DataFrame

        Returns:
            H/L 리스트 [{timestamp, type, price, index, histogram}, ...]
        """
        df = self.calculate_macd(df)
        crossovers = self.find_histogram_crossovers(df)

        highs_lows = []

        for i, cross in enumerate(crossovers):
            # 이전 크로스오버 인덱스
            if i == 0:
                # 첫 크로스오버는 데이터 시작부터
                prev_idx = 0
            else:
                prev_idx = df.index.get_loc(crossovers[i-1]['index']) + 1

            curr_idx = df.index.get_loc(cross['index'])

            # 구간 데이터
            segment = df.iloc[prev_idx:curr_idx]

            if len(segment) == 0:
                continue

            # 양수→음수: High
            if cross['type'] == 'pos_to_neg':
                high_idx = segment['high'].idxmax()
                high_price = segment.loc[high_idx, 'high']

                highs_lows.append({
                    'timestamp': df.loc[high_idx, 'datetime'] if 'datetime' in df.columns else high_idx,
                    'type': 'H',
                    'price': high_price,
                    'index': high_idx,
                    'histogram': df.loc[cross['index'], 'macd_hist'],
                    'crossover_timestamp': cross['timestamp']
                })

            # 음수→양수: Low
            elif cross['type'] == 'neg_to_pos':
                low_idx = segment['low'].idxmin()
                low_price = segment.loc[low_idx, 'low']

                highs_lows.append({
                    'timestamp': df.loc[low_idx, 'datetime'] if 'datetime' in df.columns else low_idx,
                    'type': 'L',
                    'price': low_price,
                    'index': low_idx,
                    'histogram': df.loc[cross['index'], 'macd_hist'],
                    'crossover_timestamp': cross['timestamp']
                })

        return pd.DataFrame(highs_lows)

    def analyze_hl_statistics(self, hl_df):
        """
        H/L 통계 분석

        Args:
            hl_df: label_highs_lows()의 결과 DataFrame

        Returns:
            통계 딕셔너리
        """
        if len(hl_df) == 0:
            return None

        stats = {
            'total_count': len(hl_df),
            'high_count': len(hl_df[hl_df['type'] == 'H']),
            'low_count': len(hl_df[hl_df['type'] == 'L']),
            'price_range': {
                'min': hl_df['price'].min(),
                'max': hl_df['price'].max(),
                'mean': hl_df['price'].mean(),
                'std': hl_df['price'].std()
            }
        }

        # H-L 간격 계산
        if 'timestamp' in hl_df.columns:
            intervals = []
            for i in range(1, len(hl_df)):
                interval = hl_df.iloc[i]['timestamp'] - hl_df.iloc[i-1]['timestamp']
                intervals.append(interval.total_seconds() / 3600)  # 시간 단위

            if intervals:
                stats['avg_interval_hours'] = np.mean(intervals)
                stats['median_interval_hours'] = np.median(intervals)
                stats['min_interval_hours'] = np.min(intervals)
                stats['max_interval_hours'] = np.max(intervals)

        return stats


def resample_to_timeframe(df, timeframe):
    """
    15분봉 데이터를 다른 타임프레임으로 리샘플링

    Args:
        df: 15분봉 DataFrame (datetime 인덱스)
        timeframe: 타임프레임 ('1H', '4H', '1D', '3D', '1W')

    Returns:
        리샘플링된 DataFrame
    """
    df = df.copy()

    # datetime을 인덱스로 설정
    if 'datetime' in df.columns and df.index.name != 'datetime':
        df.set_index('datetime', inplace=True)

    # 리샘플링 규칙
    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # datetime 컬럼 복원
    resampled.reset_index(inplace=True)

    return resampled


def process_all_timeframes(df_15m):
    """
    모든 타임프레임에서 H/L 라벨링 수행

    Args:
        df_15m: 15분봉 DataFrame

    Returns:
        {timeframe: {df, hl_df, stats}, ...}
    """
    timeframes = ['15T', '1H', '4H', '1D', '3D', '1W']
    results = {}

    labeler = MACDHighLowLabeler()

    for tf in timeframes:
        print(f"\n처리 중: {tf}")

        # 리샘플링
        if tf == '15T':
            df_resampled = df_15m.copy()
        else:
            df_resampled = resample_to_timeframe(df_15m, tf)

        print(f"  데이터 shape: {df_resampled.shape}")

        # H/L 라벨링
        hl_df = labeler.label_highs_lows(df_resampled)
        print(f"  H/L 개수: {len(hl_df)}")

        # 통계
        stats = labeler.analyze_hl_statistics(hl_df)

        if stats:
            print(f"  H: {stats['high_count']}, L: {stats['low_count']}")
            if 'avg_interval_hours' in stats:
                print(f"  평균 간격: {stats['avg_interval_hours']:.1f}시간")

        results[tf] = {
            'df': df_resampled,
            'hl_df': hl_df,
            'stats': stats
        }

    return results


if __name__ == "__main__":
    import os
    import glob

    print("=" * 60)
    print("MACD H/L 라벨링 시스템 테스트")
    print("=" * 60)

    # 데이터 로드
    data_dir = "data"
    pattern = os.path.join(data_dir, "BTC_USDT_USDT_15m_*.csv")
    files = glob.glob(pattern)

    if not files:
        print("\nBTC 데이터 파일이 없습니다.")
        print("먼저 bybit_collector_ccxt.py를 실행하여 데이터를 수집하세요.")
    else:
        latest_file = max(files, key=os.path.getmtime)
        print(f"\n데이터 로드: {latest_file}")

        df = pd.read_csv(latest_file)
        df['datetime'] = pd.to_datetime(df['datetime'])

        print(f"원본 데이터: {df.shape}")
        print(f"기간: {df['datetime'].min()} ~ {df['datetime'].max()}")

        # 모든 타임프레임 처리
        results = process_all_timeframes(df)

        # 결과 요약
        print("\n" + "=" * 60)
        print("타임프레임별 H/L 요약")
        print("=" * 60)

        for tf, result in results.items():
            stats = result['stats']
            if stats:
                print(f"\n{tf}:")
                print(f"  총 H/L: {stats['total_count']}")
                print(f"  H: {stats['high_count']}, L: {stats['low_count']}")
                if 'avg_interval_hours' in stats:
                    print(f"  평균 간격: {stats['avg_interval_hours']:.1f}시간 ({stats['avg_interval_hours']/24:.1f}일)")
