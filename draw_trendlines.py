import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# 데이터 로드
df = pd.read_csv('btc_15m_ohlcv.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# 최근 500봉만
df = df.tail(500).reset_index(drop=True)

# 고점/저점 찾기 (로컬 extrema)
order = 10  # 앞뒤 10봉 비교

# 고점 (High의 local maxima)
high_idx = argrelextrema(df['high'].values, np.greater, order=order)[0]

# 저점 (Low의 local minima)  
low_idx = argrelextrema(df['low'].values, np.less, order=order)[0]

print(f"고점 개수: {len(high_idx)}")
print(f"저점 개수: {len(low_idx)}")

# 하락 추세선 그리기 (고점들 연결)
# 과거 고점에서 현재까지 이어지는 선

fig, ax = plt.subplots(figsize=(16, 8))

# 캔들 그리기 (간단히 선으로)
ax.plot(df.index, df['close'], 'gray', alpha=0.5, linewidth=0.5)

# 고점 표시
ax.scatter(high_idx, df.loc[high_idx, 'high'], color='red', marker='v', s=50, label='Highs')

# 저점 표시
ax.scatter(low_idx, df.loc[low_idx, 'low'], color='green', marker='^', s=50, label='Lows')

# 하락 추세선 그리기 (연속 고점 연결)
print("\n[하락 추세선 - 고점 연결]")
for i in range(len(high_idx) - 1):
    idx1, idx2 = high_idx[i], high_idx[i+1]
    h1, h2 = df.loc[idx1, 'high'], df.loc[idx2, 'high']
    
    # 하락하는 경우만 (h1 > h2)
    if h1 > h2:
        # 추세선 연장 (현재까지)
        slope = (h2 - h1) / (idx2 - idx1)
        
        # 현재까지 연장
        x_end = len(df) - 1
        y_end = h1 + slope * (x_end - idx1)
        
        ax.plot([idx1, x_end], [h1, y_end], 'r--', alpha=0.5, linewidth=1)
        
        print(f"  {df.loc[idx1, 'datetime'].strftime('%m/%d %H:%M')} ~ 현재 | 기울기: {slope:.2f}")

# 상승 추세선 그리기 (연속 저점 연결)
print("\n[상승 추세선 - 저점 연결]")
for i in range(len(low_idx) - 1):
    idx1, idx2 = low_idx[i], low_idx[i+1]
    l1, l2 = df.loc[idx1, 'low'], df.loc[idx2, 'low']
    
    # 상승하는 경우만 (l1 < l2)
    if l1 < l2:
        slope = (l2 - l1) / (idx2 - idx1)
        
        x_end = len(df) - 1
        y_end = l1 + slope * (x_end - idx1)
        
        ax.plot([idx1, x_end], [l1, y_end], 'g--', alpha=0.5, linewidth=1)
        
        print(f"  {df.loc[idx1, 'datetime'].strftime('%m/%d %H:%M')} ~ 현재 | 기울기: {slope:.2f}")

ax.set_title('BTC 15m - Trendlines')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('trendlines.png', dpi=150)
print("\n저장: trendlines.png")

# 추세선 데이터 저장
trendlines = []
for i in range(len(high_idx) - 1):
    idx1, idx2 = high_idx[i], high_idx[i+1]
    h1, h2 = df.loc[idx1, 'high'], df.loc[idx2, 'high']
    if h1 > h2:
        slope = (h2 - h1) / (idx2 - idx1)
        trendlines.append({
            'type': 'down',
            'start_idx': idx1,
            'start_price': h1,
            'slope': slope,
            'start_time': df.loc[idx1, 'datetime']
        })

print(f"\n하락 추세선 {len(trendlines)}개 감지됨")

