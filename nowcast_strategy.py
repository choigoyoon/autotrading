"""
나우캐스트 준수 전략 - 미래 데이터 참조 완전 제거

핵심 원칙:
1. 시점 T에서는 T-1까지의 확정된 데이터만 사용
2. H/L 라벨은 MACD 크로스 시점에 확정 (1봉 지연)
3. 추세선은 현재까지 확정된 H/L로만 생성
4. 돌파는 현재 봉의 close로 실시간 판단
"""

import pandas as pd
import numpy as np
from datetime import datetime
import ccxt
import time
import os


class NowcastStrategy:
    """나우캐스트 준수 전략 클래스"""
    
    def __init__(self, macd_fast=12, macd_slow=26, macd_signal=9):
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        
    def calculate_macd(self, df):
        """MACD 계산"""
        df = df.copy()
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        return df
    
    def generate_hl_labels_nowcast(self, df):
        """
        나우캐스트 준수 H/L 라벨링
        
        규칙:
        - MACD 히스토그램 부호 전환 시점에 라벨 확정
        - H: 양수→음수 전환 시, 직전 양수 구간의 high 최고점
        - L: 음수→양수 전환 시, 직전 음수 구간의 low 최저점
        - 라벨은 전환 시점(현재 봉)에 확정 → 미래 참조 없음
        """
        df = df.copy()
        df['label'] = None
        df['label_price'] = np.nan
        df['label_bar_idx'] = np.nan  # 실제 H/L 발생 봉
        
        hist = df['macd_hist'].values
        current_sign = None
        segment_start = 0
        
        for i in range(len(df)):
            if pd.isna(hist[i]):
                continue
                
            sign = 1 if hist[i] >= 0 else -1
            
            if current_sign is not None and sign != current_sign:
                # 이전 구간 분석 (과거 데이터만!)
                segment = df.iloc[segment_start:i]
                
                if len(segment) > 0:
                    if current_sign == 1:  # 양수→음수: H
                        max_idx = segment['high'].idxmax()
                        max_price = segment.loc[max_idx, 'high']
                        df.loc[i, 'label'] = 'H'
                        df.loc[i, 'label_price'] = max_price
                        df.loc[i, 'label_bar_idx'] = max_idx
                    else:  # 음수→양수: L
                        min_idx = segment['low'].idxmin()
                        min_price = segment.loc[min_idx, 'low']
                        df.loc[i, 'label'] = 'L'
                        df.loc[i, 'label_price'] = min_price
                        df.loc[i, 'label_bar_idx'] = min_idx
                
                segment_start = i
            
            current_sign = sign
        
        return df
    
    def get_active_trendlines_at(self, df, current_idx, min_touches=2, max_age=200):
        """
        현재 시점(current_idx)에서 유효한 추세선 계산
        
        핵심: current_idx 이전의 확정된 H/L만 사용
        
        Args:
            df: 라벨링된 DataFrame
            current_idx: 현재 봉 인덱스
            min_touches: 최소 터치 수
            max_age: 추세선 최대 유효 기간 (봉 수)
        
        Returns:
            list of trendlines
        """
        # current_idx 이전까지의 확정된 라벨만 사용
        past_df = df.iloc[:current_idx]
        
        h_labels = past_df[past_df['label'] == 'H'].copy()
        l_labels = past_df[past_df['label'] == 'L'].copy()
        
        trendlines = []
        
        # 하락 추세선 (H 연결) - 최근 것부터
        if len(h_labels) >= min_touches:
            # 최근 max_age 봉 내의 H만
            recent_h = h_labels[h_labels.index >= current_idx - max_age]
            
            if len(recent_h) >= min_touches:
                # 가장 최근 2개 H로 추세선
                h1_idx = recent_h.index[-2]
                h2_idx = recent_h.index[-1]
                h1_price = recent_h.loc[h1_idx, 'label_price']
                h2_price = recent_h.loc[h2_idx, 'label_price']
                
                # 하락 추세선 조건: H2 < H1
                if h2_price < h1_price:
                    slope = (h2_price - h1_price) / (h2_idx - h1_idx)
                    trendlines.append({
                        'type': 'down',
                        'start_idx': h1_idx,
                        'end_idx': h2_idx,
                        'start_price': h1_price,
                        'end_price': h2_price,
                        'slope': slope
                    })
        
        # 상승 추세선 (L 연결)
        if len(l_labels) >= min_touches:
            recent_l = l_labels[l_labels.index >= current_idx - max_age]
            
            if len(recent_l) >= min_touches:
                l1_idx = recent_l.index[-2]
                l2_idx = recent_l.index[-1]
                l1_price = recent_l.loc[l1_idx, 'label_price']
                l2_price = recent_l.loc[l2_idx, 'label_price']
                
                # 상승 추세선 조건: L2 > L1
                if l2_price > l1_price:
                    slope = (l2_price - l1_price) / (l2_idx - l1_idx)
                    trendlines.append({
                        'type': 'up',
                        'start_idx': l1_idx,
                        'end_idx': l2_idx,
                        'start_price': l1_price,
                        'end_price': l2_price,
                        'slope': slope
                    })
        
        return trendlines
    
    def get_trendline_price_at(self, trendline, idx):
        """특정 인덱스에서 추세선 가격 계산"""
        return trendline['start_price'] + trendline['slope'] * (idx - trendline['start_idx'])
    
    def detect_breakout_at(self, df, current_idx, trendlines):
        """
        현재 봉에서 돌파 감지
        
        조건:
        - 하락추세선 상향 돌파: 이전 봉 close < 추세선, 현재 봉 close > 추세선
        - 상승추세선 하향 돌파: 이전 봉 close > 추세선, 현재 봉 close < 추세선
        """
        if current_idx < 1 or len(trendlines) == 0:
            return None
        
        current_close = df.iloc[current_idx]['close']
        prev_close = df.iloc[current_idx - 1]['close']
        
        for tl in trendlines:
            tl_price_curr = self.get_trendline_price_at(tl, current_idx)
            tl_price_prev = self.get_trendline_price_at(tl, current_idx - 1)
            
            # 하락추세선 상향 돌파 (롱 신호)
            if tl['type'] == 'down':
                if prev_close <= tl_price_prev and current_close > tl_price_curr:
                    return {
                        'type': 'trendline_up',
                        'break_idx': current_idx,
                        'break_price': current_close,
                        'trendline_price': tl_price_curr,
                        'trendline': tl
                    }
            
            # 상승추세선 하향 돌파 (숏 신호)
            elif tl['type'] == 'up':
                if prev_close >= tl_price_prev and current_close < tl_price_curr:
                    return {
                        'type': 'trendline_down',
                        'break_idx': current_idx,
                        'break_price': current_close,
                        'trendline_price': tl_price_curr,
                        'trendline': tl
                    }
        
        return None
    
    def detect_fvg(self, df, idx):
        """
        FVG (Fair Value Gap) 감지
        Bullish FVG: candle[i-2].high < candle[i].low
        """
        if idx < 2:
            return False
        
        candle_2_high = df.iloc[idx - 2]['high']
        candle_0_low = df.iloc[idx]['low']
        
        return candle_0_low > candle_2_high
    
    def backtest_nowcast(self, df, tp_pct=2.0, sl_pct=2.0, min_interval=10, 
                         use_fvg_filter=False, use_dynamic_tp=False):
        """
        나우캐스트 준수 백테스트
        
        Args:
            df: OHLCV DataFrame
            tp_pct: Take Profit %
            sl_pct: Stop Loss %
            min_interval: 최소 거래 간격 (봉)
            use_fvg_filter: FVG 필터 사용 여부
            use_dynamic_tp: 동적 TP 사용 여부
        
        Returns:
            trades DataFrame, stats dict
        """
        print("MACD 계산 중...")
        df = self.calculate_macd(df)
        
        print("H/L 라벨링 중...")
        df = self.generate_hl_labels_nowcast(df)
        
        trades = []
        last_entry_idx = -min_interval - 1
        
        # 최소 MACD 웜업 기간
        start_idx = max(self.macd_slow + self.macd_signal, 50)
        
        print(f"백테스트 실행 중... (총 {len(df) - start_idx}봉)")
        
        for current_idx in range(start_idx, len(df) - 50):  # 50봉 여유
            # 간격 필터
            if current_idx - last_entry_idx < min_interval:
                continue
            
            # 현재 시점의 추세선 계산
            trendlines = self.get_active_trendlines_at(df, current_idx)
            
            if len(trendlines) == 0:
                continue
            
            # 돌파 감지
            breakout = self.detect_breakout_at(df, current_idx, trendlines)
            
            if breakout is None:
                continue
            
            # 롱만 (trendline_up)
            if breakout['type'] != 'trendline_up':
                continue
            
            # FVG 필터
            has_fvg = self.detect_fvg(df, current_idx)
            if use_fvg_filter and not has_fvg:
                continue
            
            # 동적 TP
            if use_dynamic_tp:
                actual_tp = 2.5 if has_fvg else 1.5
            else:
                actual_tp = tp_pct
            
            # 진입가 (다음 봉 시가로 진입 - 더 현실적)
            entry_idx = current_idx + 1
            entry_price = df.iloc[entry_idx]['open']
            
            # TP/SL 레벨
            tp_level = entry_price * (1 + actual_tp / 100)
            sl_level = entry_price * (1 - sl_pct / 100)
            
            # 결과 시뮬레이션 (최대 50봉)
            max_hold = min(entry_idx + 50, len(df))
            result = 'TIMEOUT'
            exit_price = df.iloc[max_hold - 1]['close']
            exit_idx = max_hold - 1
            
            for j in range(entry_idx + 1, max_hold):
                high = df.iloc[j]['high']
                low = df.iloc[j]['low']
                
                # TP 먼저 체크 (같은 봉에서 둘 다 터치 시)
                if high >= tp_level:
                    result = 'TP'
                    exit_price = tp_level
                    exit_idx = j
                    break
                elif low <= sl_level:
                    result = 'SL'
                    exit_price = sl_level
                    exit_idx = j
                    break
            
            # 수익 계산
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            
            trades.append({
                'entry_idx': entry_idx,
                'entry_time': df.iloc[entry_idx]['datetime'],
                'entry_price': entry_price,
                'exit_idx': exit_idx,
                'exit_price': exit_price,
                'tp_pct': actual_tp,
                'sl_pct': sl_pct,
                'has_fvg': has_fvg,
                'result': result,
                'pnl_pct': pnl_pct,
                'hold_bars': exit_idx - entry_idx
            })
            
            last_entry_idx = current_idx
        
        trades_df = pd.DataFrame(trades)
        
        # 통계
        if len(trades_df) > 0:
            stats = {
                'total_trades': len(trades_df),
                'win_rate': (trades_df['pnl_pct'] > 0).sum() / len(trades_df) * 100,
                'avg_pnl': trades_df['pnl_pct'].mean(),
                'total_pnl': trades_df['pnl_pct'].sum(),
                'max_win': trades_df['pnl_pct'].max(),
                'max_loss': trades_df['pnl_pct'].min(),
                'avg_hold_bars': trades_df['hold_bars'].mean(),
                'tp_count': (trades_df['result'] == 'TP').sum(),
                'sl_count': (trades_df['result'] == 'SL').sum(),
                'timeout_count': (trades_df['result'] == 'TIMEOUT').sum(),
            }
        else:
            stats = {'total_trades': 0}
        
        return trades_df, stats


def collect_data(days=365):
    """데이터 수집"""
    print("=" * 60)
    print("BTC 데이터 수집")
    print("=" * 60)
    
    exchange = ccxt.bybit({
        'enableRateLimit': True,
        'options': {'defaultType': 'linear'}
    })
    
    symbol = 'BTC/USDT:USDT'
    timeframe = '15m'
    
    # 시작 시간
    since = int((datetime.now() - pd.Timedelta(days=days)).timestamp() * 1000)
    
    all_ohlcv = []
    iteration = 0
    
    print(f"수집 기간: 최근 {days}일")
    
    while True:
        try:
            iteration += 1
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            
            if iteration % 10 == 0:
                print(f"  수집 중... {len(all_ohlcv)}개")
            
            if last_ts >= int(datetime.now().timestamp() * 1000) - 15 * 60 * 1000:
                break
            
            since = last_ts + 1
            time.sleep(0.1)
            
            if iteration >= 5000:
                break
                
        except Exception as e:
            print(f"에러: {e}")
            time.sleep(5)
            continue
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n수집 완료: {len(df)}개 봉")
    print(f"기간: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    
    return df


def run_parameter_optimization(df, strategy):
    """파라미터 최적화"""
    print("\n" + "=" * 60)
    print("파라미터 최적화")
    print("=" * 60)
    
    results = []
    
    # 파라미터 그리드
    tp_range = [1.0, 1.5, 2.0, 2.5, 3.0]
    sl_range = [1.0, 1.5, 2.0, 2.5, 3.0]
    interval_range = [5, 10, 15, 20]
    
    total_combos = len(tp_range) * len(sl_range) * len(interval_range)
    print(f"총 {total_combos}개 조합 테스트")
    
    combo_num = 0
    for tp in tp_range:
        for sl in sl_range:
            for interval in interval_range:
                combo_num += 1
                
                trades_df, stats = strategy.backtest_nowcast(
                    df.copy(), 
                    tp_pct=tp, 
                    sl_pct=sl, 
                    min_interval=interval,
                    use_fvg_filter=False,
                    use_dynamic_tp=False
                )
                
                if stats['total_trades'] > 0:
                    results.append({
                        'tp': tp,
                        'sl': sl,
                        'interval': interval,
                        'trades': stats['total_trades'],
                        'win_rate': stats['win_rate'],
                        'avg_pnl': stats['avg_pnl'],
                        'total_pnl': stats['total_pnl']
                    })
                
                if combo_num % 20 == 0:
                    print(f"  진행: {combo_num}/{total_combos}")
    
    results_df = pd.DataFrame(results)
    
    # 정렬 및 출력
    if len(results_df) > 0:
        results_df = results_df.sort_values('avg_pnl', ascending=False)
        
        print("\n상위 10개 결과:")
        print(results_df.head(10).to_string(index=False))
        
        # 최고 승률
        print("\n최고 승률 조합:")
        best_wr = results_df.sort_values('win_rate', ascending=False).head(5)
        print(best_wr.to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    print("=" * 60)
    print("나우캐스트 준수 전략 테스트")
    print("=" * 60)
    
    # 데이터 수집 (최근 2년)
    df = collect_data(days=730)
    
    # 전략 초기화
    strategy = NowcastStrategy()
    
    # 기본 백테스트
    print("\n" + "=" * 60)
    print("기본 백테스트 (TP:2%, SL:2%, 간격:10봉)")
    print("=" * 60)
    
    trades_df, stats = strategy.backtest_nowcast(
        df.copy(),
        tp_pct=2.0,
        sl_pct=2.0,
        min_interval=10,
        use_fvg_filter=False,
        use_dynamic_tp=False
    )
    
    print(f"\n결과:")
    print(f"  총 거래: {stats['total_trades']}회")
    print(f"  승률: {stats['win_rate']:.1f}%")
    print(f"  평균 수익: {stats['avg_pnl']:.3f}%")
    print(f"  총 수익: {stats['total_pnl']:.1f}%")
    print(f"  TP 횟수: {stats['tp_count']}")
    print(f"  SL 횟수: {stats['sl_count']}")
    print(f"  타임아웃: {stats['timeout_count']}")
    
    # FVG 필터 테스트
    print("\n" + "=" * 60)
    print("FVG 필터 적용")
    print("=" * 60)
    
    trades_fvg, stats_fvg = strategy.backtest_nowcast(
        df.copy(),
        tp_pct=2.0,
        sl_pct=2.0,
        min_interval=10,
        use_fvg_filter=True,
        use_dynamic_tp=False
    )
    
    print(f"\n결과:")
    print(f"  총 거래: {stats_fvg['total_trades']}회")
    print(f"  승률: {stats_fvg['win_rate']:.1f}%")
    print(f"  평균 수익: {stats_fvg['avg_pnl']:.3f}%")
    
    # 동적 TP 테스트
    print("\n" + "=" * 60)
    print("동적 TP 적용 (FVG 있음: 2.5%, 없음: 1.5%)")
    print("=" * 60)
    
    trades_dyn, stats_dyn = strategy.backtest_nowcast(
        df.copy(),
        tp_pct=2.0,
        sl_pct=2.0,
        min_interval=10,
        use_fvg_filter=False,
        use_dynamic_tp=True
    )
    
    print(f"\n결과:")
    print(f"  총 거래: {stats_dyn['total_trades']}회")
    print(f"  승률: {stats_dyn['win_rate']:.1f}%")
    print(f"  평균 수익: {stats_dyn['avg_pnl']:.3f}%")
    
    # 파라미터 최적화 (선택적)
    print("\n파라미터 최적화를 실행하시겠습니까? (시간 소요)")
    # results_df = run_parameter_optimization(df, strategy)
    
    # 결과 저장
    if len(trades_df) > 0:
        trades_df.to_csv('nowcast_trades.csv', index=False)
        print(f"\n거래 내역 저장: nowcast_trades.csv")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)
