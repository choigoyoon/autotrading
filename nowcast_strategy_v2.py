"""
나우캐스트 준수 전략 V2 - Binance 데이터 사용

핵심 원칙:
1. 시점 T에서는 T-1까지의 확정된 데이터만 사용
2. H/L 라벨은 MACD 크로스 시점에 확정 (1봉 지연)
3. 추세선은 현재까지 확정된 H/L로만 생성
4. 돌파는 현재 봉의 close로 실시간 판단
"""

import pandas as pd
import numpy as np
from datetime import datetime
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
        """
        df = df.copy()
        df['label'] = None
        df['label_price'] = np.nan
        df['label_bar_idx'] = np.nan
        
        hist = df['macd_hist'].values
        current_sign = None
        segment_start = 0
        
        for i in range(len(df)):
            if pd.isna(hist[i]):
                continue
                
            sign = 1 if hist[i] >= 0 else -1
            
            if current_sign is not None and sign != current_sign:
                segment = df.iloc[segment_start:i]
                
                if len(segment) > 0:
                    if current_sign == 1:
                        max_idx = segment['high'].idxmax()
                        max_price = segment.loc[max_idx, 'high']
                        df.loc[i, 'label'] = 'H'
                        df.loc[i, 'label_price'] = max_price
                        df.loc[i, 'label_bar_idx'] = max_idx
                    else:
                        min_idx = segment['low'].idxmin()
                        min_price = segment.loc[min_idx, 'low']
                        df.loc[i, 'label'] = 'L'
                        df.loc[i, 'label_price'] = min_price
                        df.loc[i, 'label_bar_idx'] = min_idx
                
                segment_start = i
            
            current_sign = sign
        
        return df
    
    def get_active_trendlines_at(self, df, current_idx, min_touches=2, max_age=200):
        """현재 시점에서 유효한 추세선 계산 - 미래 참조 없음"""
        past_df = df.iloc[:current_idx]
        
        h_labels = past_df[past_df['label'] == 'H'].copy()
        l_labels = past_df[past_df['label'] == 'L'].copy()
        
        trendlines = []
        
        # 하락 추세선 (H 연결)
        if len(h_labels) >= min_touches:
            recent_h = h_labels[h_labels.index >= current_idx - max_age]
            
            if len(recent_h) >= min_touches:
                h1_idx = recent_h.index[-2]
                h2_idx = recent_h.index[-1]
                h1_price = recent_h.loc[h1_idx, 'label_price']
                h2_price = recent_h.loc[h2_idx, 'label_price']
                
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
        """현재 봉에서 돌파 감지"""
        if current_idx < 1 or len(trendlines) == 0:
            return None
        
        current_close = df.iloc[current_idx]['close']
        prev_close = df.iloc[current_idx - 1]['close']
        
        for tl in trendlines:
            tl_price_curr = self.get_trendline_price_at(tl, current_idx)
            tl_price_prev = self.get_trendline_price_at(tl, current_idx - 1)
            
            if tl['type'] == 'down':
                if prev_close <= tl_price_prev and current_close > tl_price_curr:
                    return {
                        'type': 'trendline_up',
                        'break_idx': current_idx,
                        'break_price': current_close,
                        'trendline_price': tl_price_curr,
                        'trendline': tl
                    }
            
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
        """FVG 감지"""
        if idx < 2:
            return False
        candle_2_high = df.iloc[idx - 2]['high']
        candle_0_low = df.iloc[idx]['low']
        return candle_0_low > candle_2_high
    
    def backtest_nowcast(self, df, tp_pct=2.0, sl_pct=2.0, min_interval=10, 
                         use_fvg_filter=False, use_dynamic_tp=False,
                         long_only=True, verbose=True):
        """나우캐스트 준수 백테스트"""
        if verbose:
            print("MACD 계산 중...")
        df = self.calculate_macd(df)
        
        if verbose:
            print("H/L 라벨링 중...")
        df = self.generate_hl_labels_nowcast(df)
        
        trades = []
        last_entry_idx = -min_interval - 1
        
        start_idx = max(self.macd_slow + self.macd_signal, 50)
        
        if verbose:
            print(f"백테스트 실행 중... (총 {len(df) - start_idx}봉)")
        
        for current_idx in range(start_idx, len(df) - 50):
            if current_idx - last_entry_idx < min_interval:
                continue
            
            trendlines = self.get_active_trendlines_at(df, current_idx)
            
            if len(trendlines) == 0:
                continue
            
            breakout = self.detect_breakout_at(df, current_idx, trendlines)
            
            if breakout is None:
                continue
            
            # 롱 전용
            if long_only and breakout['type'] != 'trendline_up':
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
            
            # 진입 (다음 봉 시가)
            entry_idx = current_idx + 1
            entry_price = df.iloc[entry_idx]['open']
            
            direction = 'long' if breakout['type'] == 'trendline_up' else 'short'
            
            if direction == 'long':
                tp_level = entry_price * (1 + actual_tp / 100)
                sl_level = entry_price * (1 - sl_pct / 100)
            else:
                tp_level = entry_price * (1 - actual_tp / 100)
                sl_level = entry_price * (1 + sl_pct / 100)
            
            max_hold = min(entry_idx + 50, len(df))
            result = 'TIMEOUT'
            exit_price = df.iloc[max_hold - 1]['close']
            exit_idx = max_hold - 1
            
            for j in range(entry_idx + 1, max_hold):
                high = df.iloc[j]['high']
                low = df.iloc[j]['low']
                
                if direction == 'long':
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
                else:
                    if low <= tp_level:
                        result = 'TP'
                        exit_price = tp_level
                        exit_idx = j
                        break
                    elif high >= sl_level:
                        result = 'SL'
                        exit_price = sl_level
                        exit_idx = j
                        break
            
            if direction == 'long':
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100
            
            trades.append({
                'entry_idx': entry_idx,
                'entry_time': df.iloc[entry_idx]['datetime'] if 'datetime' in df.columns else entry_idx,
                'entry_price': entry_price,
                'direction': direction,
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


def collect_binance_data(symbol='BTCUSDT', interval='15m', days=365):
    """Binance에서 데이터 수집"""
    import requests
    
    print("=" * 60)
    print(f"Binance에서 {symbol} 데이터 수집")
    print("=" * 60)
    
    base_url = "https://api.binance.com/api/v3/klines"
    
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - pd.Timedelta(days=days)).timestamp() * 1000)
    
    all_data = []
    current_start = start_time
    
    print(f"수집 기간: 최근 {days}일")
    
    iteration = 0
    while current_start < end_time:
        iteration += 1
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'limit': 1000
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            data = response.json()
            
            if not data or len(data) == 0:
                break
            
            all_data.extend(data)
            current_start = data[-1][0] + 1
            
            if iteration % 10 == 0:
                print(f"  수집 중... {len(all_data)}개")
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"에러: {e}")
            time.sleep(1)
            continue
    
    if len(all_data) == 0:
        print("데이터 수집 실패")
        return None
    
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])
    df['volume'] = pd.to_numeric(df['volume'])
    
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n수집 완료: {len(df)}개 봉")
    print(f"기간: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    
    return df


def run_comprehensive_test(df, strategy):
    """종합 테스트 및 최적화"""
    print("\n" + "=" * 60)
    print("종합 테스트 및 최적화")
    print("=" * 60)
    
    results = []
    
    # 테스트 파라미터
    test_configs = [
        # 기본 설정들
        {'tp': 1.0, 'sl': 1.0, 'interval': 10, 'fvg': False, 'dyn': False, 'name': 'TP1_SL1'},
        {'tp': 1.5, 'sl': 1.5, 'interval': 10, 'fvg': False, 'dyn': False, 'name': 'TP1.5_SL1.5'},
        {'tp': 2.0, 'sl': 2.0, 'interval': 10, 'fvg': False, 'dyn': False, 'name': 'TP2_SL2'},
        {'tp': 2.5, 'sl': 2.5, 'interval': 10, 'fvg': False, 'dyn': False, 'name': 'TP2.5_SL2.5'},
        {'tp': 3.0, 'sl': 3.0, 'interval': 10, 'fvg': False, 'dyn': False, 'name': 'TP3_SL3'},
        
        # 넓은 SL 테스트 (기존 전략의 핵심 발견)
        {'tp': 1.0, 'sl': 2.0, 'interval': 10, 'fvg': False, 'dyn': False, 'name': 'TP1_SL2'},
        {'tp': 1.5, 'sl': 2.0, 'interval': 10, 'fvg': False, 'dyn': False, 'name': 'TP1.5_SL2'},
        {'tp': 1.0, 'sl': 3.0, 'interval': 10, 'fvg': False, 'dyn': False, 'name': 'TP1_SL3'},
        {'tp': 1.5, 'sl': 3.0, 'interval': 10, 'fvg': False, 'dyn': False, 'name': 'TP1.5_SL3'},
        
        # 간격 테스트
        {'tp': 2.0, 'sl': 2.0, 'interval': 5, 'fvg': False, 'dyn': False, 'name': 'INT5'},
        {'tp': 2.0, 'sl': 2.0, 'interval': 15, 'fvg': False, 'dyn': False, 'name': 'INT15'},
        {'tp': 2.0, 'sl': 2.0, 'interval': 20, 'fvg': False, 'dyn': False, 'name': 'INT20'},
        
        # FVG 필터
        {'tp': 2.0, 'sl': 2.0, 'interval': 10, 'fvg': True, 'dyn': False, 'name': 'FVG_FILTER'},
        {'tp': 2.5, 'sl': 2.0, 'interval': 10, 'fvg': True, 'dyn': False, 'name': 'FVG_TP2.5'},
        
        # 동적 TP
        {'tp': 2.0, 'sl': 2.0, 'interval': 10, 'fvg': False, 'dyn': True, 'name': 'DYN_TP'},
        {'tp': 2.0, 'sl': 2.0, 'interval': 10, 'fvg': True, 'dyn': True, 'name': 'FVG+DYN'},
    ]
    
    print(f"\n총 {len(test_configs)}개 설정 테스트")
    
    for i, config in enumerate(test_configs, 1):
        trades_df, stats = strategy.backtest_nowcast(
            df.copy(),
            tp_pct=config['tp'],
            sl_pct=config['sl'],
            min_interval=config['interval'],
            use_fvg_filter=config['fvg'],
            use_dynamic_tp=config['dyn'],
            verbose=False
        )
        
        if stats['total_trades'] > 0:
            results.append({
                'name': config['name'],
                'tp': config['tp'],
                'sl': config['sl'],
                'interval': config['interval'],
                'fvg': config['fvg'],
                'dyn': config['dyn'],
                'trades': stats['total_trades'],
                'win_rate': stats['win_rate'],
                'avg_pnl': stats['avg_pnl'],
                'total_pnl': stats['total_pnl'],
                'tp_count': stats['tp_count'],
                'sl_count': stats['sl_count']
            })
        
        print(f"  [{i}/{len(test_configs)}] {config['name']}: "
              f"거래={stats.get('total_trades', 0)}, "
              f"승률={stats.get('win_rate', 0):.1f}%, "
              f"평균={stats.get('avg_pnl', 0):.3f}%")
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("결과 요약 (평균 수익 순)")
    print("=" * 60)
    
    if len(results_df) > 0:
        results_df = results_df.sort_values('avg_pnl', ascending=False)
        print(results_df[['name', 'trades', 'win_rate', 'avg_pnl', 'total_pnl']].to_string(index=False))
        
        print("\n" + "=" * 60)
        print("최고 승률 설정")
        print("=" * 60)
        best_wr = results_df.sort_values('win_rate', ascending=False).head(5)
        print(best_wr[['name', 'trades', 'win_rate', 'avg_pnl']].to_string(index=False))
    
    return results_df


def run_extended_optimization(df, strategy):
    """확장 파라미터 최적화"""
    print("\n" + "=" * 60)
    print("확장 파라미터 최적화")
    print("=" * 60)
    
    results = []
    
    tp_range = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    sl_range = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    total = len(tp_range) * len(sl_range)
    print(f"총 {total}개 조합 테스트")
    
    count = 0
    for tp in tp_range:
        for sl in sl_range:
            count += 1
            
            trades_df, stats = strategy.backtest_nowcast(
                df.copy(),
                tp_pct=tp,
                sl_pct=sl,
                min_interval=10,
                use_fvg_filter=False,
                use_dynamic_tp=False,
                verbose=False
            )
            
            if stats['total_trades'] > 0:
                # 수수료 반영 (0.11% 왕복)
                net_pnl = stats['avg_pnl'] - 0.11
                
                results.append({
                    'tp': tp,
                    'sl': sl,
                    'trades': stats['total_trades'],
                    'win_rate': stats['win_rate'],
                    'avg_pnl': stats['avg_pnl'],
                    'net_pnl': net_pnl,
                    'total_pnl': stats['total_pnl']
                })
            
            if count % 10 == 0:
                print(f"  진행: {count}/{total}")
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        print("\n상위 10개 (순수익 기준):")
        top10 = results_df.sort_values('net_pnl', ascending=False).head(10)
        print(top10.to_string(index=False))
        
        print("\n상위 5개 (승률 기준):")
        top_wr = results_df.sort_values('win_rate', ascending=False).head(5)
        print(top_wr.to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    print("=" * 60)
    print("나우캐스트 준수 전략 V2 테스트")
    print("=" * 60)
    
    # 데이터 수집 (최근 2년)
    df = collect_binance_data(symbol='BTCUSDT', interval='15m', days=730)
    
    if df is None or len(df) == 0:
        print("데이터 수집 실패!")
        exit(1)
    
    # 데이터 저장
    df.to_csv('btc_15m_data.csv', index=False)
    print(f"\n데이터 저장: btc_15m_data.csv")
    
    # 전략 초기화
    strategy = NowcastStrategy()
    
    # 종합 테스트
    results_df = run_comprehensive_test(df, strategy)
    
    # 확장 최적화
    opt_results = run_extended_optimization(df, strategy)
    
    # 최종 권장 설정으로 상세 결과
    print("\n" + "=" * 60)
    print("최종 권장 설정 상세 분석")
    print("=" * 60)
    
    # 최고 성과 설정 찾기
    if len(opt_results) > 0:
        best = opt_results.sort_values('net_pnl', ascending=False).iloc[0]
        print(f"\n최적 파라미터: TP={best['tp']}%, SL={best['sl']}%")
        
        trades_df, stats = strategy.backtest_nowcast(
            df.copy(),
            tp_pct=best['tp'],
            sl_pct=best['sl'],
            min_interval=10,
            use_fvg_filter=False,
            use_dynamic_tp=False,
            verbose=True
        )
        
        print(f"\n상세 결과:")
        print(f"  총 거래: {stats['total_trades']}회")
        print(f"  승률: {stats['win_rate']:.1f}%")
        print(f"  평균 수익: {stats['avg_pnl']:.3f}%")
        print(f"  수수료 후: {stats['avg_pnl'] - 0.11:.3f}%")
        print(f"  총 수익: {stats['total_pnl']:.1f}%")
        print(f"  TP 횟수: {stats['tp_count']}")
        print(f"  SL 횟수: {stats['sl_count']}")
        print(f"  타임아웃: {stats['timeout_count']}")
        
        # 월별 추정
        months = len(df) / (4 * 24 * 30)  # 15분봉 기준
        monthly_trades = stats['total_trades'] / months
        monthly_return = stats['total_pnl'] / months
        
        print(f"\n월간 추정:")
        print(f"  월 거래: {monthly_trades:.1f}회")
        print(f"  월 수익: {monthly_return:.1f}%")
        
        # 거래 내역 저장
        if len(trades_df) > 0:
            trades_df.to_csv('nowcast_trades_v2.csv', index=False)
            print(f"\n거래 내역 저장: nowcast_trades_v2.csv")
    
    # 결과 저장
    if len(opt_results) > 0:
        opt_results.to_csv('optimization_results.csv', index=False)
        print(f"최적화 결과 저장: optimization_results.csv")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)
