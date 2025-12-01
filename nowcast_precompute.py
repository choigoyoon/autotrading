"""
나우캐스트 백테스트 - 사전계산 방식
1단계: H/L 라벨 생성
2단계: 추세선 사전 계산
3단계: 돌파 시점 사전 계산
4단계: 백테스트 (빠름)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("1단계: 데이터 로드 및 MACD 계산")
print("=" * 70)

# 데이터 로드
df = pd.read_csv('btc_15m_ohlcv.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna().reset_index(drop=True)

print(f"데이터: {len(df)}봉")
print(f"기간: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

# MACD 계산
fast, slow, signal = 12, 26, 9
ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
df['macd'] = ema_fast - ema_slow
df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']

print("MACD 계산 완료")

print("\n" + "=" * 70)
print("2단계: H/L 라벨 생성 (나우캐스트 준수)")
print("=" * 70)

# H/L 라벨 생성
df['label'] = ''
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
            if current_sign == 1:  # H
                max_idx = segment['high'].idxmax()
                df.loc[i, 'label'] = 'H'
                df.loc[i, 'label_price'] = segment.loc[max_idx, 'high']
                df.loc[i, 'label_bar_idx'] = max_idx
            else:  # L
                min_idx = segment['low'].idxmin()
                df.loc[i, 'label'] = 'L'
                df.loc[i, 'label_price'] = segment.loc[min_idx, 'low']
                df.loc[i, 'label_bar_idx'] = min_idx
        
        segment_start = i
    
    current_sign = sign

h_count = (df['label'] == 'H').sum()
l_count = (df['label'] == 'L').sum()
print(f"H: {h_count}개, L: {l_count}개")

# H/L 인덱스와 가격 미리 추출
h_df = df[df['label'] == 'H'][['label_price']].copy()
l_df = df[df['label'] == 'L'][['label_price']].copy()
h_indices = h_df.index.values
l_indices = l_df.index.values
h_prices = h_df['label_price'].values
l_prices = l_df['label_price'].values

print(f"H 인덱스: {len(h_indices)}개")
print(f"L 인덱스: {len(l_indices)}개")

print("\n" + "=" * 70)
print("3단계: 돌파 시점 사전 계산")
print("=" * 70)

# 돌파 리스트 계산
breakouts = []
max_age = 200

close = df['close'].values

# 각 봉에서 사용 가능한 최근 H/L 찾기
for i in range(100, len(df)):
    # 현재 시점 이전의 H 중 최근 2개
    valid_h = h_indices[h_indices < i]
    valid_h = valid_h[valid_h >= i - max_age]
    
    # 하락추세선 돌파 체크
    if len(valid_h) >= 2:
        h1_idx, h2_idx = valid_h[-2], valid_h[-1]
        h1_p = h_prices[h_indices == h1_idx][0]
        h2_p = h_prices[h_indices == h2_idx][0]
        
        if h2_p < h1_p:  # 하락 조건
            slope = (h2_p - h1_p) / (h2_idx - h1_idx)
            tl_curr = h1_p + slope * (i - h1_idx)
            tl_prev = h1_p + slope * (i - 1 - h1_idx)
            
            if close[i-1] <= tl_prev and close[i] > tl_curr:
                # FVG 체크
                has_fvg = i >= 2 and df.iloc[i]['low'] > df.iloc[i-2]['high']
                breakouts.append({
                    'idx': i,
                    'type': 'long',
                    'price': close[i],
                    'has_fvg': has_fvg
                })
                continue  # 한 봉에 하나만
    
    # 상승추세선 돌파 체크 (숏)
    valid_l = l_indices[l_indices < i]
    valid_l = valid_l[valid_l >= i - max_age]
    
    if len(valid_l) >= 2:
        l1_idx, l2_idx = valid_l[-2], valid_l[-1]
        l1_p = l_prices[l_indices == l1_idx][0]
        l2_p = l_prices[l_indices == l2_idx][0]
        
        if l2_p > l1_p:  # 상승 조건
            slope = (l2_p - l1_p) / (l2_idx - l1_idx)
            tl_curr = l1_p + slope * (i - l1_idx)
            tl_prev = l1_p + slope * (i - 1 - l1_idx)
            
            if close[i-1] >= tl_prev and close[i] < tl_curr:
                has_fvg = i >= 2 and df.iloc[i]['high'] < df.iloc[i-2]['low']
                breakouts.append({
                    'idx': i,
                    'type': 'short',
                    'price': close[i],
                    'has_fvg': has_fvg
                })

print(f"총 돌파: {len(breakouts)}개")
print(f"  롱: {sum(1 for b in breakouts if b['type'] == 'long')}개")
print(f"  숏: {sum(1 for b in breakouts if b['type'] == 'short')}개")
print(f"  FVG 동반: {sum(1 for b in breakouts if b['has_fvg'])}개")

# 돌파 DataFrame 저장
breakouts_df = pd.DataFrame(breakouts)
breakouts_df.to_csv('nowcast_breakouts.csv', index=False)
print(f"\n돌파 시점 저장: nowcast_breakouts.csv")

print("\n" + "=" * 70)
print("4단계: 백테스트 (빠름)")
print("=" * 70)

def run_backtest(breakouts, df, tp=2.0, sl=2.0, interval=10, long_only=True,
                 use_fvg=False, dynamic_tp=False):
    """사전계산된 돌파로 백테스트"""
    
    open_p = df['open'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    trades = []
    last_entry = -interval - 1
    
    for brk in breakouts:
        i = brk['idx']
        
        # 간격 필터
        if i - last_entry < interval:
            continue
        
        # 롱 전용
        if long_only and brk['type'] != 'long':
            continue
        
        # FVG 필터
        if use_fvg and not brk['has_fvg']:
            continue
        
        # TP 설정
        if dynamic_tp:
            actual_tp = 2.5 if brk['has_fvg'] else 1.5
        else:
            actual_tp = tp
        
        # 진입 (다음 봉 시가)
        entry_idx = i + 1
        if entry_idx >= len(df) - 50:
            continue
            
        entry_price = open_p[entry_idx]
        direction = brk['type']
        
        # TP/SL 레벨
        if direction == 'long':
            tp_level = entry_price * (1 + actual_tp / 100)
            sl_level = entry_price * (1 - sl / 100)
        else:
            tp_level = entry_price * (1 - actual_tp / 100)
            sl_level = entry_price * (1 + sl / 100)
        
        # 결과
        result = 'TIMEOUT'
        exit_price = close[min(entry_idx + 49, len(df) - 1)]
        exit_idx = min(entry_idx + 49, len(df) - 1)
        
        for j in range(entry_idx + 1, min(entry_idx + 50, len(df))):
            if direction == 'long':
                if high[j] >= tp_level:
                    result, exit_price, exit_idx = 'TP', tp_level, j
                    break
                if low[j] <= sl_level:
                    result, exit_price, exit_idx = 'SL', sl_level, j
                    break
            else:
                if low[j] <= tp_level:
                    result, exit_price, exit_idx = 'TP', tp_level, j
                    break
                if high[j] >= sl_level:
                    result, exit_price, exit_idx = 'SL', sl_level, j
                    break
        
        # PnL
        if direction == 'long':
            pnl = (exit_price - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - exit_price) / entry_price * 100
        
        trades.append({
            'entry_idx': entry_idx,
            'direction': direction,
            'has_fvg': brk['has_fvg'],
            'tp_pct': actual_tp,
            'sl_pct': sl,
            'result': result,
            'pnl': pnl
        })
        
        last_entry = i
    
    return trades


def calc_stats(trades):
    if not trades:
        return {'total_trades': 0, 'win_rate': 0, 'avg_pnl': 0, 'total_pnl': 0}
    
    pnls = [t['pnl'] for t in trades]
    results = [t['result'] for t in trades]
    
    return {
        'total_trades': len(trades),
        'win_rate': sum(1 for p in pnls if p > 0) / len(pnls) * 100,
        'avg_pnl': np.mean(pnls),
        'total_pnl': sum(pnls),
        'tp_count': results.count('TP'),
        'sl_count': results.count('SL'),
        'timeout': results.count('TIMEOUT'),
    }


# 파라미터 테스트
params = [
    (1.0, 1.0, 'TP1_SL1'),
    (1.5, 1.5, 'TP1.5_SL1.5'),
    (2.0, 2.0, 'TP2_SL2'),
    (2.5, 2.5, 'TP2.5_SL2.5'),
    (3.0, 3.0, 'TP3_SL3'),
    (1.0, 2.0, 'TP1_SL2'),
    (1.5, 2.0, 'TP1.5_SL2'),
    (1.0, 3.0, 'TP1_SL3'),
    (1.5, 3.0, 'TP1.5_SL3'),
    (2.0, 3.0, 'TP2_SL3'),
    (0.5, 1.0, 'TP0.5_SL1'),
    (0.5, 2.0, 'TP0.5_SL2'),
    (0.7, 1.5, 'TP0.7_SL1.5'),
    (0.7, 2.0, 'TP0.7_SL2'),
]

results = []
print("\n파라미터 최적화:")
for tp, sl, name in params:
    trades = run_backtest(breakouts, df, tp=tp, sl=sl, interval=10)
    stats = calc_stats(trades)
    
    net_pnl = stats['avg_pnl'] - 0.11 if stats['total_trades'] > 0 else 0
    results.append({
        'name': name,
        'tp': tp,
        'sl': sl,
        'trades': stats['total_trades'],
        'win_rate': stats['win_rate'],
        'avg_pnl': stats['avg_pnl'],
        'net_pnl': net_pnl,
        'total_pnl': stats['total_pnl'],
    })
    
    print(f"  {name:12s}: 거래={stats['total_trades']:4d}, "
          f"승률={stats['win_rate']:5.1f}%, 평균={stats['avg_pnl']:+.3f}%")

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("상위 5개 (순수익 기준)")
print("=" * 70)
top5 = results_df.sort_values('net_pnl', ascending=False).head(5)
print(top5[['name', 'trades', 'win_rate', 'avg_pnl', 'net_pnl', 'total_pnl']].to_string(index=False))

print("\n" + "=" * 70)
print("상위 5개 (승률 기준)")
print("=" * 70)
top_wr = results_df.sort_values('win_rate', ascending=False).head(5)
print(top_wr[['name', 'trades', 'win_rate', 'avg_pnl', 'net_pnl', 'total_pnl']].to_string(index=False))

# FVG 테스트
print("\n" + "=" * 70)
print("FVG 필터 테스트")
print("=" * 70)
for tp, sl in [(2.0, 2.0), (1.5, 2.0), (1.0, 2.0)]:
    trades = run_backtest(breakouts, df, tp=tp, sl=sl, interval=10, use_fvg=True)
    stats = calc_stats(trades)
    print(f"  FVG TP{tp}_SL{sl}: 거래={stats['total_trades']}, "
          f"승률={stats['win_rate']:.1f}%, 평균={stats['avg_pnl']:.3f}%")

# 동적 TP
print("\n" + "=" * 70)
print("동적 TP 테스트")
print("=" * 70)
trades = run_backtest(breakouts, df, tp=2.0, sl=2.0, interval=10, dynamic_tp=True)
stats = calc_stats(trades)
print(f"  동적 TP: 거래={stats['total_trades']}, "
      f"승률={stats['win_rate']:.1f}%, 평균={stats['avg_pnl']:.3f}%")

# 최적 설정
best = results_df.sort_values('net_pnl', ascending=False).iloc[0]
print("\n" + "=" * 70)
print(f"최적 설정: {best['name']}")
print("=" * 70)

trades = run_backtest(breakouts, df, tp=best['tp'], sl=best['sl'], interval=10)
stats = calc_stats(trades)

total_days = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days
months = total_days / 30

print(f"\n  총 거래: {stats['total_trades']}회")
print(f"  승률: {stats['win_rate']:.1f}%")
print(f"  평균 수익: {stats['avg_pnl']:.3f}%")
print(f"  수수료 후: {stats['avg_pnl'] - 0.11:.3f}%")
print(f"  총 수익: {stats['total_pnl']:.1f}%")
print(f"\n월간 추정 ({months:.1f}개월):")
print(f"  월 거래: {stats['total_trades'] / months:.1f}회")
print(f"  월 수익: {stats['total_pnl'] / months:.1f}%")

# 비교
print("\n" + "=" * 70)
print("기존 전략 vs 나우캐스트")
print("=" * 70)
print("""
기존 (미래 참조):     나우캐스트 (수정):
  승률: 84-90%          승률: {:.1f}%
  월수익: 14%+          월수익: {:.1f}%
  실전: 불가            실전: 가능
""".format(best['win_rate'], stats['total_pnl'] / months))

# 저장
results_df.to_csv('nowcast_results.csv', index=False)
print(f"\n결과 저장: nowcast_results.csv")

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
