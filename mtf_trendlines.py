import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# 데이터 로드
df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

# 1H 데이터 생성
df_15m_copy = df_15m.copy()
df_15m_copy['hour'] = df_15m_copy['datetime'].dt.floor('H')
df_1h = df_15m_copy.groupby('hour').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).reset_index()
df_1h.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

def get_trendlines(df, order, name):
    """고점에서 하락 추세선 추출"""
    high_idx = argrelextrema(df['high'].values, np.greater, order=order)[0]
    
    trendlines = []
    for i in range(len(high_idx) - 1):
        idx1, idx2 = high_idx[i], high_idx[i+1]
        h1, h2 = df.iloc[idx1]['high'], df.iloc[idx2]['high']
        t1, t2 = df.iloc[idx1]['datetime'], df.iloc[idx2]['datetime']
        
        if h1 > h2:  # 하락 추세
            slope = (h2 - h1) / (idx2 - idx1)
            trendlines.append({
                'start_time': t1,
                'end_time': t2,
                'start_price': h1,
                'end_price': h2,
                'slope': slope,
                'start_idx': idx1,
                'tf': name
            })
    
    return trendlines, high_idx

# 최근 데이터만 사용
df_15m_recent = df_15m.tail(1000).reset_index(drop=True)
df_1h_recent = df_1h.tail(250).reset_index(drop=True)
df_4h_recent = df_4h.tail(100).reset_index(drop=True)

# 각 타임프레임별 추세선
tl_15m, hi_15m = get_trendlines(df_15m_recent, 5, '15m')
tl_1h, hi_1h = get_trendlines(df_1h_recent, 5, '1H')
tl_4h, hi_4h = get_trendlines(df_4h_recent, 3, '4H')

print("=" * 60)
print("📊 MTF 추세선 분석")
print("=" * 60)

print(f"\n[15분봉] 하락 추세선: {len(tl_15m)}개")
for t in tl_15m[-5:]:
    print(f"  {t['start_time'].strftime('%m/%d %H:%M')} → 기울기: {t['slope']:.1f}")

print(f"\n[1시간봉] 하락 추세선: {len(tl_1h)}개")
for t in tl_1h[-5:]:
    print(f"  {t['start_time'].strftime('%m/%d %H:%M')} → 기울기: {t['slope']:.1f}")

print(f"\n[4시간봉] 하락 추세선: {len(tl_4h)}개")
for t in tl_4h[-5:]:
    print(f"  {t['start_time'].strftime('%m/%d %H:%M')} → 기울기: {t['slope']:.1f}")

# 차트 그리기
fig, axes = plt.subplots(3, 1, figsize=(16, 14))

# 15분봉
ax = axes[0]
ax.plot(df_15m_recent.index, df_15m_recent['close'], 'gray', alpha=0.7, linewidth=0.5)
ax.scatter(hi_15m, df_15m_recent.iloc[hi_15m]['high'], color='red', marker='v', s=30)

for t in tl_15m:
    idx1 = t['start_idx']
    x_end = len(df_15m_recent) - 1
    y_end = t['start_price'] + t['slope'] * (x_end - idx1)
    ax.plot([idx1, x_end], [t['start_price'], y_end], 'r--', alpha=0.4, linewidth=1)

ax.set_title('15M Trendlines')
ax.grid(True, alpha=0.3)

# 1시간봉
ax = axes[1]
ax.plot(df_1h_recent.index, df_1h_recent['close'], 'gray', alpha=0.7, linewidth=0.5)
ax.scatter(hi_1h, df_1h_recent.iloc[hi_1h]['high'], color='red', marker='v', s=30)

for t in tl_1h:
    idx1 = t['start_idx']
    x_end = len(df_1h_recent) - 1
    y_end = t['start_price'] + t['slope'] * (x_end - idx1)
    ax.plot([idx1, x_end], [t['start_price'], y_end], 'orange', linestyle='--', alpha=0.5, linewidth=1.5)

ax.set_title('1H Trendlines')
ax.grid(True, alpha=0.3)

# 4시간봉
ax = axes[2]
ax.plot(df_4h_recent.index, df_4h_recent['close'], 'gray', alpha=0.7, linewidth=0.5)
ax.scatter(hi_4h, df_4h_recent.iloc[hi_4h]['high'], color='red', marker='v', s=30)

for t in tl_4h:
    idx1 = t['start_idx']
    x_end = len(df_4h_recent) - 1
    y_end = t['start_price'] + t['slope'] * (x_end - idx1)
    ax.plot([idx1, x_end], [t['start_price'], y_end], 'yellow', linestyle='--', alpha=0.7, linewidth=2)

ax.set_title('4H Trendlines')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mtf_trendlines.png', dpi=150, facecolor='black')
print("\n저장: mtf_trendlines.png")

# 통합 차트 (15분봉에 모든 추세선)
fig, ax = plt.subplots(figsize=(18, 10), facecolor='black')
ax.set_facecolor('black')

# 15분봉 가격
ax.plot(df_15m_recent['datetime'], df_15m_recent['close'], 'white', alpha=0.7, linewidth=0.5)

# 15분 추세선 (빨간색)
for t in tl_15m[-10:]:
    mask = df_15m_recent['datetime'] >= t['start_time']
    if mask.any():
        start_idx = mask.idxmax()
        x_vals = df_15m_recent.loc[start_idx:, 'datetime']
        y_start = t['start_price']
        y_vals = [y_start + t['slope'] * i for i in range(len(x_vals))]
        ax.plot(x_vals, y_vals, 'r--', alpha=0.5, linewidth=1, label='15m' if t == tl_15m[-10] else '')

# 1시간 추세선 (주황색) - 15분 차트에 매핑
for t in tl_1h[-7:]:
    mask = df_15m_recent['datetime'] >= t['start_time']
    if mask.any():
        start_idx = mask.idxmax()
        x_vals = df_15m_recent.loc[start_idx:, 'datetime']
        y_start = t['start_price']
        # 1시간 기울기를 15분으로 변환 (4배 느리게)
        y_vals = [y_start + t['slope']/4 * i for i in range(len(x_vals))]
        ax.plot(x_vals, y_vals, 'orange', linestyle='--', alpha=0.6, linewidth=1.5, label='1H' if t == tl_1h[-7] else '')

# 4시간 추세선 (노란색) - 15분 차트에 매핑
for t in tl_4h[-5:]:
    mask = df_15m_recent['datetime'] >= t['start_time']
    if mask.any():
        start_idx = mask.idxmax()
        x_vals = df_15m_recent.loc[start_idx:, 'datetime']
        y_start = t['start_price']
        # 4시간 기울기를 15분으로 변환 (16배 느리게)
        y_vals = [y_start + t['slope']/16 * i for i in range(len(x_vals))]
        ax.plot(x_vals, y_vals, 'yellow', linestyle='--', alpha=0.8, linewidth=2, label='4H' if t == tl_4h[-5] else '')

ax.set_title('BTC - MTF Trendlines (15m base)', color='white', fontsize=14)
ax.tick_params(colors='white')
ax.grid(True, alpha=0.2)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('mtf_combined.png', dpi=150, facecolor='black')
print("저장: mtf_combined.png")

