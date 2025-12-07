"""
L값 근접도 분석 시스템
- L값 확정 전 진입 타이밍 포착
- 체크박스 근접도 기반 예측
- 나우캐스트 준수
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class LProximityAnalyzer:
    """L값 근접도 분석기"""

    def __init__(self, df):
        """
        Args:
            df: OHLCV + MACD 데이터프레임
        """
        self.df = df.copy()
        self.l_values = []
        self.checkboxes_history = {}

        print("=" * 60)
        print("L값 근접도 분석 시스템")
        print("=" * 60)

    # ═══════════════════════════════════════════════════════
    # 1. L값 수집
    # ═══════════════════════════════════════════════════════

    def collect_l_values(self):
        """확정된 L값 수집 (MACD 0-cross)"""

        print("\n[1단계] L값 수집 중...")

        for i in range(1, len(self.df)):
            prev_hist = self.df.iloc[i-1]['macd_hist']
            curr_hist = self.df.iloc[i]['macd_hist']

            # L값 확정 (음수 → 양수)
            if prev_hist < 0 and curr_hist >= 0:
                self.l_values.append({
                    'idx': i,
                    'datetime': self.df.iloc[i]['datetime'],
                    'price': self.df.iloc[i]['low'],
                    'close': self.df.iloc[i]['close']
                })

        print(f"  총 L값: {len(self.l_values)}개")
        print(f"  기간: {self.l_values[0]['datetime']} ~ {self.l_values[-1]['datetime']}")

        return self.l_values

    # ═══════════════════════════════════════════════════════
    # 2. 지표 계산
    # ═══════════════════════════════════════════════════════

    def calculate_rsi(self, window=14):
        """RSI 계산"""
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_bollinger_bands(self, window=20, num_std=2):
        """볼린저 밴드 계산"""
        sma = self.df['close'].rolling(window=window).mean()
        std = self.df['close'].rolling(window=window).std()

        upper = sma + (std * num_std)
        lower = sma - (std * num_std)

        return upper, sma, lower

    def detect_fvg(self, idx):
        """FVG (Fair Value Gap) 감지"""
        if idx < 2:
            return False, 0.0

        # FVG 조건: candle[i-2].high < candle[i].low
        candle_before = self.df.iloc[idx - 2]
        candle_current = self.df.iloc[idx]

        if candle_before['high'] < candle_current['low']:
            gap_size = (candle_current['low'] - candle_before['high']) / candle_before['high'] * 100
            return True, gap_size

        return False, 0.0

    def detect_liquidity_sweep(self, idx, lookback=10):
        """Liquidity Sweep 감지 (저점 돌파 후 즉시 반등)"""
        if idx < lookback + 1:
            return False

        # 최근 lookback 기간 최저가
        recent_low = self.df.iloc[idx - lookback:idx]['low'].min()
        current_low = self.df.iloc[idx]['low']
        current_close = self.df.iloc[idx]['close']

        # 저점 돌파 (현재 저가 < 최근 최저가)
        # AND 즉시 반등 (종가 > 최근 최저가)
        if current_low < recent_low and current_close > recent_low:
            return True

        return False

    def count_support_levels(self, idx, lookback=20):
        """중간 지지 개수 (하락 중 반등 횟수)"""
        if idx < lookback:
            return 0

        recent_data = self.df.iloc[idx - lookback:idx]

        # 하락 추세에서 반등 횟수
        support_count = 0
        for i in range(1, len(recent_data)):
            prev_close = recent_data.iloc[i-1]['close']
            curr_close = recent_data.iloc[i]['close']

            # 반등 = 이전 하락 후 상승
            if curr_close > prev_close:
                support_count += 1

        return support_count

    # ═══════════════════════════════════════════════════════
    # 3. 체크박스 근접도 계산
    # ═══════════════════════════════════════════════════════

    def calculate_proximity(self, value, target_range, reverse=False):
        """
        범위 기반 근접도 계산

        Args:
            value: 현재 값
            target_range: (min, max) 목표 범위
            reverse: True면 max에 가까울수록 100%

        Returns:
            근접도 (0~100%)
        """
        min_val, max_val = target_range

        # 범위 밖
        if value < min_val or value > max_val:
            return 0.0

        # 범위 내
        if reverse:
            # max에 가까울수록 좋음
            proximity = (value - min_val) / (max_val - min_val) * 100
        else:
            # min에 가까울수록 좋음
            proximity = (max_val - value) / (max_val - min_val) * 100

        return proximity

    def calculate_checkboxes(self, idx):
        """특정 시점의 모든 체크박스 근접도 계산"""

        if idx < 50:  # 최소 데이터 필요
            return None

        candle = self.df.iloc[idx]
        checkboxes = {}

        # RSI
        rsi = self.calculate_rsi()
        rsi_val = rsi.iloc[idx] if idx < len(rsi) else 50
        checkboxes['rsi_proximity'] = self.calculate_proximity(rsi_val, (20, 40))

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands()
        if idx < len(bb_lower):
            price = candle['close']
            bb_low = bb_lower.iloc[idx]
            bb_mid = bb_middle.iloc[idx]

            if not np.isnan(bb_low) and not np.isnan(bb_mid) and bb_mid > bb_low:
                bb_proximity = (price - bb_low) / (bb_mid - bb_low) * 100
                checkboxes['bb_proximity'] = max(0, min(100, 100 - bb_proximity))  # 하단에 가까울수록 100
            else:
                checkboxes['bb_proximity'] = 0
        else:
            checkboxes['bb_proximity'] = 0

        # MACD 히스토그램 0 접근도
        macd_hist = candle['macd_hist']
        recent_hist_max = self.df.iloc[max(0, idx-20):idx]['macd_hist'].abs().max()

        if recent_hist_max > 0:
            macd_0_proximity = (1 - abs(macd_hist) / recent_hist_max) * 100
            checkboxes['macd_0_proximity'] = max(0, macd_0_proximity)
        else:
            checkboxes['macd_0_proximity'] = 0

        # FVG
        fvg_exists, fvg_size = self.detect_fvg(idx)
        checkboxes['fvg_exists'] = fvg_exists
        checkboxes['fvg_size'] = fvg_size

        # Liquidity Sweep
        checkboxes['liquidity_sweep'] = self.detect_liquidity_sweep(idx)

        # 중간 지지 개수 (적을수록 좋음)
        support_count = self.count_support_levels(idx)
        checkboxes['support_count'] = support_count
        checkboxes['clean_drop'] = (support_count == 0)  # 깔끔한 하락

        # 꼬리 비율
        body = abs(candle['close'] - candle['open'])
        lower_tail = min(candle['open'], candle['close']) - candle['low']
        upper_tail = candle['high'] - max(candle['open'], candle['close'])

        checkboxes['lower_tail_ratio'] = lower_tail / body if body > 0 else 0
        checkboxes['has_long_lower_tail'] = (lower_tail > body * 2)  # 아래 꼬리 > 몸통*2

        # 평균 근접도 (주요 지표만)
        main_proximities = [
            checkboxes['rsi_proximity'],
            checkboxes['bb_proximity'],
            checkboxes['macd_0_proximity']
        ]
        checkboxes['avg_proximity'] = np.mean(main_proximities)

        return checkboxes

    # ═══════════════════════════════════════════════════════
    # 4. L값 전후 분석
    # ═══════════════════════════════════════════════════════

    def analyze_l_context(self, l_idx, window=20):
        """L값 기준 전후 분석"""

        context = {
            'l_idx': l_idx,
            'before': [],
            'at_l': None,
            'after': [],
            'proximity_trend': [],
            'success': None
        }

        # 전: L값 -window ~ L값 -1
        for i in range(max(50, l_idx - window), l_idx):
            cb = self.calculate_checkboxes(i)
            if cb:
                context['before'].append(cb)
                context['proximity_trend'].append(cb['avg_proximity'])

        # 중: L값
        context['at_l'] = self.calculate_checkboxes(l_idx)
        if context['at_l']:
            context['proximity_trend'].append(context['at_l']['avg_proximity'])

        # 후: L값 +1 ~ L값 +window
        for i in range(l_idx + 1, min(len(self.df), l_idx + window + 1)):
            cb = self.calculate_checkboxes(i)
            if cb:
                context['after'].append(cb)

        # 성공 여부 (L값 이후 상승했는가?)
        if l_idx + window < len(self.df):
            l_price = self.df.iloc[l_idx]['close']
            future_max = self.df.iloc[l_idx:l_idx + window]['high'].max()
            gain = (future_max - l_price) / l_price * 100

            context['success'] = (gain >= 2.0)  # 2% 이상 상승 = 성공
            context['gain'] = gain

        return context

    # ═══════════════════════════════════════════════════════
    # 5. 패턴 학습
    # ═══════════════════════════════════════════════════════

    def find_effective_patterns(self, all_contexts):
        """효과적인 체크박스 조합 찾기"""

        print("\n[패턴 분석] 성공 vs 실패 L값 특징 비교...")

        success_contexts = [c for c in all_contexts if c['success'] == True]
        failure_contexts = [c for c in all_contexts if c['success'] == False]

        print(f"  성공 L값: {len(success_contexts)}개")
        print(f"  실패 L값: {len(failure_contexts)}개")

        # 성공 L값 특징
        success_features = {
            'fvg_rate': 0,
            'liquidity_sweep_rate': 0,
            'clean_drop_rate': 0,
            'long_tail_rate': 0,
            'avg_rsi_proximity': [],
            'avg_bb_proximity': [],
            'avg_macd_proximity': [],
            'avg_overall_proximity': []
        }

        for ctx in success_contexts:
            at_l = ctx['at_l']
            if at_l:
                success_features['fvg_rate'] += (1 if at_l['fvg_exists'] else 0)
                success_features['liquidity_sweep_rate'] += (1 if at_l['liquidity_sweep'] else 0)
                success_features['clean_drop_rate'] += (1 if at_l['clean_drop'] else 0)
                success_features['long_tail_rate'] += (1 if at_l['has_long_lower_tail'] else 0)
                success_features['avg_rsi_proximity'].append(at_l['rsi_proximity'])
                success_features['avg_bb_proximity'].append(at_l['bb_proximity'])
                success_features['avg_macd_proximity'].append(at_l['macd_0_proximity'])
                success_features['avg_overall_proximity'].append(at_l['avg_proximity'])

        n_success = len(success_contexts)
        if n_success > 0:
            success_features['fvg_rate'] = success_features['fvg_rate'] / n_success * 100
            success_features['liquidity_sweep_rate'] = success_features['liquidity_sweep_rate'] / n_success * 100
            success_features['clean_drop_rate'] = success_features['clean_drop_rate'] / n_success * 100
            success_features['long_tail_rate'] = success_features['long_tail_rate'] / n_success * 100
            success_features['avg_rsi_proximity'] = np.mean(success_features['avg_rsi_proximity'])
            success_features['avg_bb_proximity'] = np.mean(success_features['avg_bb_proximity'])
            success_features['avg_macd_proximity'] = np.mean(success_features['avg_macd_proximity'])
            success_features['avg_overall_proximity'] = np.mean(success_features['avg_overall_proximity'])

        # 실패 L값 특징 (동일 로직)
        failure_features = {
            'fvg_rate': 0,
            'liquidity_sweep_rate': 0,
            'clean_drop_rate': 0,
            'long_tail_rate': 0,
            'avg_rsi_proximity': [],
            'avg_bb_proximity': [],
            'avg_macd_proximity': [],
            'avg_overall_proximity': []
        }

        for ctx in failure_contexts:
            at_l = ctx['at_l']
            if at_l:
                failure_features['fvg_rate'] += (1 if at_l['fvg_exists'] else 0)
                failure_features['liquidity_sweep_rate'] += (1 if at_l['liquidity_sweep'] else 0)
                failure_features['clean_drop_rate'] += (1 if at_l['clean_drop'] else 0)
                failure_features['long_tail_rate'] += (1 if at_l['has_long_lower_tail'] else 0)
                failure_features['avg_rsi_proximity'].append(at_l['rsi_proximity'])
                failure_features['avg_bb_proximity'].append(at_l['bb_proximity'])
                failure_features['avg_macd_proximity'].append(at_l['macd_0_proximity'])
                failure_features['avg_overall_proximity'].append(at_l['avg_proximity'])

        n_failure = len(failure_contexts)
        if n_failure > 0:
            failure_features['fvg_rate'] = failure_features['fvg_rate'] / n_failure * 100
            failure_features['liquidity_sweep_rate'] = failure_features['liquidity_sweep_rate'] / n_failure * 100
            failure_features['clean_drop_rate'] = failure_features['clean_drop_rate'] / n_failure * 100
            failure_features['long_tail_rate'] = failure_features['long_tail_rate'] / n_failure * 100
            failure_features['avg_rsi_proximity'] = np.mean(failure_features['avg_rsi_proximity'])
            failure_features['avg_bb_proximity'] = np.mean(failure_features['avg_bb_proximity'])
            failure_features['avg_macd_proximity'] = np.mean(failure_features['avg_macd_proximity'])
            failure_features['avg_overall_proximity'] = np.mean(failure_features['avg_overall_proximity'])

        return success_features, failure_features

    # ═══════════════════════════════════════════════════════
    # 6. 백테스트
    # ═══════════════════════════════════════════════════════

    def backtest(self, proximity_threshold=80, tp_pct=2.0, sl_pct=2.0):
        """
        L값 근접도 기반 백테스트

        진입 조건:
        - 평균 근접도 >= threshold
        - AND (FVG 발생 OR Liquidity Sweep OR 깔끔한 하락)

        청산:
        - TP 2% / SL 2%
        """
        print(f"\n[백테스트] 근접도 임계값 {proximity_threshold}%...")

        trades = []
        last_trade_idx = -999

        for i in range(50, len(self.df) - 20):  # 최소 데이터 확보

            # 간격 체크
            if i - last_trade_idx < 10:
                continue

            # 체크박스 계산
            cb = self.calculate_checkboxes(i)
            if not cb:
                continue

            # 진입 조건
            if cb['avg_proximity'] >= proximity_threshold:
                # 확인 신호
                if cb['fvg_exists'] or cb['liquidity_sweep'] or cb['clean_drop']:

                    # 진입!
                    entry_idx = i + 1  # 다음 봉
                    if entry_idx >= len(self.df):
                        break

                    entry_price = self.df.iloc[entry_idx]['open']
                    entry_time = self.df.iloc[entry_idx]['datetime']

                    tp_price = entry_price * (1 + tp_pct / 100)
                    sl_price = entry_price * (1 - sl_pct / 100)

                    # 청산 시뮬레이션
                    exit_idx = None
                    exit_price = None
                    exit_type = None

                    for j in range(entry_idx, min(entry_idx + 50, len(self.df))):
                        candle = self.df.iloc[j]

                        # SL
                        if candle['low'] <= sl_price:
                            exit_idx = j
                            exit_price = sl_price
                            exit_type = 'SL'
                            break

                        # TP
                        if candle['high'] >= tp_price:
                            exit_idx = j
                            exit_price = tp_price
                            exit_type = 'TP'
                            break

                    # 타임아웃
                    if exit_idx is None:
                        exit_idx = min(entry_idx + 50, len(self.df) - 1)
                        exit_price = self.df.iloc[exit_idx]['close']
                        exit_type = 'TIMEOUT'

                    # PnL 계산
                    pnl = (exit_price - entry_price) / entry_price * 100

                    # L값과의 거리 계산
                    distance_from_l = None
                    for l in self.l_values:
                        if l['idx'] >= entry_idx and l['idx'] <= entry_idx + 20:
                            distance_from_l = (entry_price - l['price']) / l['price'] * 100
                            break

                    trades.append({
                        'entry_idx': entry_idx,
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_type': exit_type,
                        'proximity': cb['avg_proximity'],
                        'fvg': cb['fvg_exists'],
                        'liquidity_sweep': cb['liquidity_sweep'],
                        'clean_drop': cb['clean_drop'],
                        'distance_from_l': distance_from_l
                    })

                    last_trade_idx = entry_idx

        # 통계
        if len(trades) == 0:
            print("  거래 없음")
            return None

        df_trades = pd.DataFrame(trades)

        win_trades = df_trades[df_trades['pnl'] > 0]
        lose_trades = df_trades[df_trades['pnl'] <= 0]

        win_rate = len(win_trades) / len(df_trades) * 100
        avg_pnl = df_trades['pnl'].mean()
        avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
        avg_loss = lose_trades['pnl'].mean() if len(lose_trades) > 0 else 0

        # L값 거리 (None 제외)
        distance_values = df_trades['distance_from_l'].dropna()
        avg_distance = distance_values.mean() if len(distance_values) > 0 else 0

        print(f"  총 거래: {len(df_trades)}개")
        print(f"  승률: {win_rate:.1f}%")
        print(f"  평균 PnL: {avg_pnl:+.2f}%")
        print(f"  평균 승: {avg_win:+.2f}%")
        print(f"  평균 패: {avg_loss:+.2f}%")
        print(f"  L값 대비 진입가: {avg_distance:+.2f}%")

        return df_trades

    # ═══════════════════════════════════════════════════════
    # 7. 메인 실행
    # ═══════════════════════════════════════════════════════

    def run(self):
        """전체 분석 실행"""

        # 1. L값 수집
        self.collect_l_values()

        # 2. 지표 추가
        print("\n[2단계] 지표 계산 중...")
        self.df['rsi'] = self.calculate_rsi()
        print("  RSI, MACD, BB 계산 완료")

        # 3. L값 전후 분석
        print(f"\n[3단계] L값 전후 분석 중... ({len(self.l_values)}개)")
        all_contexts = []

        for i, l_val in enumerate(self.l_values):
            if i % 100 == 0:
                print(f"  진행: {i}/{len(self.l_values)}")

            ctx = self.analyze_l_context(l_val['idx'])
            all_contexts.append(ctx)

        # 4. 패턴 학습
        success_feat, failure_feat = self.find_effective_patterns(all_contexts)

        print("\n" + "=" * 60)
        print("성공 L값 특징 (2% 이상 상승)")
        print("=" * 60)
        print(f"  FVG 발생률: {success_feat['fvg_rate']:.1f}%")
        print(f"  Liquidity Sweep: {success_feat['liquidity_sweep_rate']:.1f}%")
        print(f"  깔끔한 하락: {success_feat['clean_drop_rate']:.1f}%")
        print(f"  긴 아래꼬리: {success_feat['long_tail_rate']:.1f}%")
        print(f"  평균 RSI 근접도: {success_feat['avg_rsi_proximity']:.1f}%")
        print(f"  평균 BB 근접도: {success_feat['avg_bb_proximity']:.1f}%")
        print(f"  평균 MACD 근접도: {success_feat['avg_macd_proximity']:.1f}%")
        print(f"  평균 전체 근접도: {success_feat['avg_overall_proximity']:.1f}%")

        print("\n" + "=" * 60)
        print("실패 L값 특징 (2% 미만)")
        print("=" * 60)
        print(f"  FVG 발생률: {failure_feat['fvg_rate']:.1f}%")
        print(f"  Liquidity Sweep: {failure_feat['liquidity_sweep_rate']:.1f}%")
        print(f"  깔끔한 하락: {failure_feat['clean_drop_rate']:.1f}%")
        print(f"  긴 아래꼬리: {failure_feat['long_tail_rate']:.1f}%")
        print(f"  평균 RSI 근접도: {failure_feat['avg_rsi_proximity']:.1f}%")
        print(f"  평균 BB 근접도: {failure_feat['avg_bb_proximity']:.1f}%")
        print(f"  평균 MACD 근접도: {failure_feat['avg_macd_proximity']:.1f}%")
        print(f"  평균 전체 근접도: {failure_feat['avg_overall_proximity']:.1f}%")

        print("\n" + "=" * 60)
        print("차이점 (결정적 요소)")
        print("=" * 60)
        print(f"  FVG: +{success_feat['fvg_rate'] - failure_feat['fvg_rate']:.1f}%p")
        print(f"  Liquidity Sweep: +{success_feat['liquidity_sweep_rate'] - failure_feat['liquidity_sweep_rate']:.1f}%p")
        print(f"  깔끔한 하락: +{success_feat['clean_drop_rate'] - failure_feat['clean_drop_rate']:.1f}%p")
        print(f"  평균 근접도: +{success_feat['avg_overall_proximity'] - failure_feat['avg_overall_proximity']:.1f}%p")

        # 5. 백테스트
        print("\n" + "=" * 60)
        print("백테스트 (근접도 기반 진입)")
        print("=" * 60)

        for threshold in [70, 75, 80, 85]:
            print(f"\n--- 임계값 {threshold}% ---")
            self.backtest(proximity_threshold=threshold)


# ═══════════════════════════════════════════════════════════
# 메인 실행
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 데이터 로드
    print("데이터 로드 중...")
    df = pd.read_csv('output_phase1_labeled.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"데이터: {len(df):,}개 캔들")
    print(f"기간: {df['datetime'].min()} ~ {df['datetime'].max()}")

    # 분석 실행
    analyzer = LProximityAnalyzer(df)
    analyzer.run()

    print("\n✅ 분석 완료!")
