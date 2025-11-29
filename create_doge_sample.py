"""
DOGE 샘플 데이터 생성 (실제 2021년 랠리 기반)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 실제 DOGE 주요 가격 포인트 (2021년 랠리)
price_points = [
    ('2020-01-01', 0.0020),
    ('2020-06-01', 0.0024),
    ('2020-12-01', 0.0045),
    ('2021-01-01', 0.0047),
    ('2021-01-28', 0.0077),  # 첫 랠리 시작
    ('2021-02-01', 0.0520),  # 급등
    ('2021-02-08', 0.0880),  # 고점1
    ('2021-02-15', 0.0550),  # 조정
    ('2021-04-01', 0.0590),
    ('2021-04-14', 0.1200),  # 랠리2 시작
    ('2021-04-16', 0.4200),  # 급등
    ('2021-05-08', 0.7400),  # 역대 최고가
    ('2021-05-19', 0.3100),  # 급락
    ('2021-06-01', 0.3300),
    ('2021-07-01', 0.2100),
    ('2021-09-01', 0.2400),
    ('2021-11-01', 0.2700),
    ('2022-01-01', 0.1700),
    ('2022-05-01', 0.0800),  # 약세장
    ('2022-12-01', 0.0700),
    ('2023-01-01', 0.0750),
    ('2023-06-01', 0.0680),
    ('2023-12-01', 0.0950),
    ('2024-01-01', 0.0820),
    ('2024-03-01', 0.1800),  # 반등
    ('2024-06-01', 0.1400),
    ('2024-10-01', 0.1500),
    ('2024-11-01', 0.1650),
]

# 일별 데이터 생성
dates = []
prices = []

for i in range(len(price_points) - 1):
    start_date = datetime.strptime(price_points[i][0], '%Y-%m-%d')
    end_date = datetime.strptime(price_points[i+1][0], '%Y-%m-%d')
    start_price = price_points[i][1]
    end_price = price_points[i+1][1]

    days = (end_date - start_date).days

    for day in range(days):
        current_date = start_date + timedelta(days=day)

        # 선형 보간 + 노이즈
        progress = day / days
        base_price = start_price + (end_price - start_price) * progress

        # 변동성 추가 (5-10%)
        noise = np.random.uniform(-0.05, 0.05)
        price = base_price * (1 + noise)

        dates.append(current_date)
        prices.append(price)

# 데이터프레임 생성
df = pd.DataFrame({'date': dates})

# OHLCV 생성
np.random.seed(42)
for i, price in enumerate(prices):
    volatility = 0.03  # 3% 일일 변동성

    open_p = price * (1 + np.random.uniform(-volatility, volatility))
    close_p = price * (1 + np.random.uniform(-volatility, volatility))

    high_p = max(open_p, close_p) * (1 + abs(np.random.uniform(0, volatility)))
    low_p = min(open_p, close_p) * (1 - abs(np.random.uniform(0, volatility)))

    volume = np.random.uniform(1e9, 5e9) * (1 + abs(close_p - open_p) / open_p * 10)

    df.loc[i, 'open'] = open_p
    df.loc[i, 'high'] = high_p
    df.loc[i, 'low'] = low_p
    df.loc[i, 'close'] = close_p
    df.loc[i, 'volume'] = volume

# CSV 저장
df['Date'] = df['date'].dt.strftime('%Y-%m-%d')
df_export = df[['Date', 'open', 'high', 'low', 'close', 'volume']]
df_export.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

df_export.to_csv('doge_sample_data.csv', index=False)

print("=" * 70)
print("DOGE 샘플 데이터 생성 완료")
print("=" * 70)
print(f"\n파일: doge_sample_data.csv")
print(f"기간: {df['date'].min().date()} ~ {df['date'].max().date()}")
print(f"총 일수: {len(df)}일")
print(f"가격 범위: ${df['low'].min():.4f} ~ ${df['high'].max():.4f}")
print(f"\n주요 이벤트:")
print(f"  2021-02-08: $0.088 (첫 고점)")
print(f"  2021-05-08: $0.74 (역대 최고가)")
print(f"  2022년: 약세장 (70% 하락)")
print(f"  2024-03: $0.18 (반등)")
print("\n이 데이터로 전략 테스트 가능!")
