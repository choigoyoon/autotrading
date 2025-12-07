"""
빠른 나우캐스트 테스트 - 짧은 기간
"""

import pandas as pd
import numpy as np
from datetime import datetime
import requests
import time


class NowcastStrategy:
    """나우캐스트 준수 전략"""
    
    def __init__(self, macd_fast=12, macd_slow=26, macd_signal=9):
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        
    def calculate_macd(self, df):
        df = df.copy()
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        return df
    
    def generate_hl_labels(self, df):
        df = df.copy()
        df['label'] = None
        df['label_price'] = np.nan
        
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
                        df.loc[i, 'label'] = 'H'
                        df.loc[i, 'label_price'] = segment.loc[max_idx, 'high']
                    else:
                        min_idx = segment['low'].idxmin()
                        df.loc[i, 'label'] = 'L'
                        df.loc[i, 'label_price'] = segment.loc[min_idx, 'low']
                segment_start = i
            current_sign = sign
        return df
    
    def get_trendlines_at(self, df, current_idx, max_age=200):
        past_df = df.iloc[:current_idx]
        h_labels = past_df[past_df['label'] == 'H']
        l_labels = past_df[past_df['label'] == 'L']
        
        trendlines = []
        
        # 하락추세선
        recent_h = h_labels[h_labels.index >= current_idx - max_age]
        if len(recent_h) >= 2:
            h1_idx, h2_idx = recent_h.index[-2], recent_h.index[-1]
            h1_price = recent_h.loc[h1_idx, 'label_price']
            h2_price = recent_h.loc[h2_idx, 'label_price']
            if h2_price < h1_price:
                slope = (h2_price - h1_price) / (h2_idx - h1_idx)
                trendlines.append({'type': 'down', 'start_idx': h1_idx, 
                                   'start_price': h1_price, 'slope': slope})
        
        # 상승추세선
        recent_l = l_labels[l_labels.index >= current_idx - max_age]
        if len(recent_l) >= 2:
            l1_idx, l2_idx = recent_l.index[-2], recent_l.index[-1]
            l1_price = recent_l.loc[l1_idx, 'label_price']
            l2_price = recent_l.loc[l2_idx, 'label_price']
            if l2_price > l1_price:
                slope = (l2_price - l1_price) / (l2_idx - l1_idx)
                trendlines.append({'type': 'up', 'start_idx': l1_idx,
                                   'start_price': l1_price, 'slope': slope})
        
        return trendlines
    
    def get_tl_price(self, tl, idx):
        return tl['start_price'] + tl['slope'] * (idx - tl['start_idx'])
    
    def detect_breakout(self, df, idx, trendlines):
        if idx < 1 or not trendlines:
            return None
        
        curr_close = df.iloc[idx]['close']
        prev_close = df.iloc[idx - 1]['close']
        
        for tl in trendlines:
            tl_curr = self.get_tl_price(tl, idx)
            tl_prev = self.get_tl_price(tl, idx - 1)
            
            if tl['type'] == 'down' and prev_close <= tl_prev and curr_close > tl_curr:
                return {'type': 'long', 'price': curr_close}
            if tl['type'] == 'up' and prev_close >= tl_prev and curr_close < tl_curr:
                return {'type': 'short', 'price': curr_close}
        return None
    
    def backtest(self, df, tp=2.0, sl=2.0, interval=10, long_only=True):
        df = self.calculate_macd(df)
        df = self.generate_hl_labels(df)
        
        trades = []
        last_entry = -interval - 1
        start = max(self.macd_slow + self.macd_signal, 50)
        
        for i in range(start, len(df) - 50):
            if i - last_entry < interval:
                continue
            
            tls = self.get_trendlines_at(df, i)
            if not tls:
                continue
            
            brk = self.detect_breakout(df, i, tls)
            if not brk:
                continue
            
            if long_only and brk['type'] != 'long':
                continue
            
            entry_idx = i + 1
            entry_price = df.iloc[entry_idx]['open']
            direction = brk['type']
            
            if direction == 'long':
                tp_level = entry_price * (1 + tp / 100)
                sl_level = entry_price * (1 - sl / 100)
            else:
                tp_level = entry_price * (1 - tp / 100)
                sl_level = entry_price * (1 + sl / 100)
            
            result = 'TIMEOUT'
            exit_price = df.iloc[min(entry_idx + 49, len(df) - 1)]['close']
            
            for j in range(entry_idx + 1, min(entry_idx + 50, len(df))):
                h, l = df.iloc[j]['high'], df.iloc[j]['low']
                
                if direction == 'long':
                    if h >= tp_level:
                        result, exit_price = 'TP', tp_level
                        break
                    if l <= sl_level:
                        result, exit_price = 'SL', sl_level
                        break
                else:
                    if l <= tp_level:
                        result, exit_price = 'TP', tp_level
                        break
                    if h >= sl_level:
                        result, exit_price = 'SL', sl_level
                        break
            
            pnl = (exit_price - entry_price) / entry_price * 100
            if direction == 'short':
                pnl = -pnl
            
            trades.append({'pnl': pnl, 'result': result, 'direction': direction})
            last_entry = i
        
        if not trades:
            return {'trades': 0}
        
        trades_df = pd.DataFrame(trades)
        return {
            'trades': len(trades_df),
            'win_rate': (trades_df['pnl'] > 0).mean() * 100,
            'avg_pnl': trades_df['pnl'].mean(),
            'total_pnl': trades_df['pnl'].sum(),
            'tp_count': (trades_df['result'] == 'TP').sum(),
            'sl_count': (trades_df['result'] == 'SL').sum()
        }


def get_binance_data(days=180):
    """Binance에서 BTC 15분봉 수집"""
    print(f"Binance에서 최근 {days}일 데이터 수집...")
    
    url = "https://api.binance.com/api/v3/klines"
    end = int(datetime.now().timestamp() * 1000)
    start = int((datetime.now() - pd.Timedelta(days=days)).timestamp() * 1000)
    
    all_data = []
    current = start
    
    while current < end:
        params = {'symbol': 'BTCUSDT', 'interval': '15m', 'startTime': current, 'limit': 1000}
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if not data:
                break
            all_data.extend(data)
            current = data[-1][0] + 1
            time.sleep(0.05)
        except Exception as e:
            print(f"에러: {e}")
            break
    
    if not all_data:
        return None
    
    df = pd.DataFrame(all_data)
    df = df.iloc[:, :6]
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    
    print(f"수집 완료: {len(df)}봉 ({df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]})")
    return df


def main():
    print("=" * 60)
    print("나우캐스트 준수 전략 빠른 테스트")
    print("=" * 60)
    
    # 데이터 수집 (6개월)
    df = get_binance_data(days=180)
    if df is None:
        print("데이터 수집 실패")
        return
    
    strategy = NowcastStrategy()
    
    # 빠른 테스트
    print("\n파라미터 테스트 중...")
    
    results = []
    test_params = [
        (1.0, 1.0), (1.0, 2.0), (1.0, 3.0),
        (1.5, 1.5), (1.5, 2.0), (1.5, 3.0),
        (2.0, 2.0), (2.0, 3.0), (2.0, 4.0),
        (2.5, 2.5), (2.5, 3.0), (3.0, 3.0),
        (0.7, 2.0), (0.7, 3.0), (0.5, 2.0),
    ]
    
    for tp, sl in test_params:
        stats = strategy.backtest(df.copy(), tp=tp, sl=sl, interval=10)
        if stats['trades'] > 0:
            net = stats['avg_pnl'] - 0.11  # 수수료
            results.append({
                'tp': tp, 'sl': sl,
                'trades': stats['trades'],
                'win_rate': stats['win_rate'],
                'avg_pnl': stats['avg_pnl'],
                'net_pnl': net,
                'total': stats['total_pnl']
            })
            print(f"  TP:{tp} SL:{sl} -> 거래:{stats['trades']}, 승률:{stats['win_rate']:.1f}%, 평균:{stats['avg_pnl']:.3f}%")
    
    print("\n" + "=" * 60)
    print("결과 요약 (순수익 순)")
    print("=" * 60)
    
    results_df = pd.DataFrame(results).sort_values('net_pnl', ascending=False)
    print(results_df.to_string(index=False))
    
    # 최고 설정 상세
    if len(results_df) > 0:
        best = results_df.iloc[0]
        print(f"\n최적 설정: TP={best['tp']}%, SL={best['sl']}%")
        print(f"  거래수: {best['trades']}")
        print(f"  승률: {best['win_rate']:.1f}%")
        print(f"  평균 수익: {best['avg_pnl']:.3f}%")
        print(f"  수수료 후: {best['net_pnl']:.3f}%")
        
        # 월간 추정
        months = len(df) / (4 * 24 * 30)
        monthly_trades = best['trades'] / months
        monthly_return = best['total'] / months
        
        print(f"\n월간 추정:")
        print(f"  월 거래: {monthly_trades:.1f}회")
        print(f"  월 수익: {monthly_return:.1f}%")
    
    # 기존 전략과 비교
    print("\n" + "=" * 60)
    print("기존 전략 vs 나우캐스트 비교")
    print("=" * 60)
    print("""
기존 전략 (미래 참조):
  - 승률: 84-90%
  - 월 수익: 14%+
  - 문제: 미래 데이터로 추세선 완성

나우캐스트 전략 (수정):
  - 미래 참조 완전 제거
  - 현재 시점 데이터만 사용
  - 실전 적용 가능
""")
    
    if len(results_df) > 0:
        best_wr = results_df.sort_values('win_rate', ascending=False).iloc[0]
        print(f"나우캐스트 최고 승률: {best_wr['win_rate']:.1f}% (TP:{best_wr['tp']} SL:{best_wr['sl']})")
        
        # 승률 격차 분석
        gap = 84 - best_wr['win_rate']
        print(f"\n승률 격차: {gap:.1f}%p")
        print(f"격차 원인: 기존 전략의 미래 참조 효과")
    
    # 보완 전략 제안
    print("\n" + "=" * 60)
    print("성과 향상 방안")
    print("=" * 60)
    print("""
1. 추가 필터 강화:
   - FVG (Fair Value Gap) 필터: 강한 모멘텀만 진입
   - 거래량 필터: 평균 이상 거래량에서만 진입
   - MTF 정렬: 상위 타임프레임 추세 확인

2. 진입 타이밍 개선:
   - 되돌림 대기: 돌파 후 0.3-0.5% 되돌림에서 진입
   - 확인봉 대기: 돌파 후 1봉 확인 후 진입

3. TP/SL 최적화:
   - 넓은 SL (2-3%): 노이즈 허용
   - 빠른 TP (0.5-1%): 수익 조기 확정
   - 트레일링 스탑: 수익 보호

4. 추세선 개선:
   - 3점 이상 터치 필요
   - 기울기 제한 (너무 급한 추세선 제외)
   - 추세선 유효 기간 제한
""")
    
    print("\n테스트 완료!")


if __name__ == "__main__":
    main()
