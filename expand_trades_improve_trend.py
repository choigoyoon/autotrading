import pandas as pd
import numpy as np

print("=" * 70)
print("1. 매매 건수 늘리기 + 2. 추세 판단 개선")
print("=" * 70)

# 데이터 로드
df = pd.read_csv('btc_1h_ohlcv.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

print(f"1시간봉: {len(df)}개")
print(f"기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
print()

# 지표 계산
df['EMA20'] = df['close'].ewm(span=20).mean()
df['EMA50'] = df['close'].ewm(span=50).mean()
df['EMA200'] = df['close'].ewm(span=200).mean()

# BB 30
df['BB_mid'] = df['close'].rolling(30).mean()
df['BB_std'] = df['close'].rolling(30).std()
df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid'] * 100

# 수축 감지 (BB width 기준)
df['squeeze'] = df['BB_width'] < df['BB_width'].rolling(50).mean()

# 모멘텀
df['momentum_50'] = (df['close'] - df['close'].shift(50)) / df['close'].shift(50) * 100
df['momentum_100'] = (df['close'] - df['close'].shift(100)) / df['close'].shift(100) * 100
df['momentum_200'] = (df['close'] - df['close'].shift(200)) / df['close'].shift(200) * 100

# 고점/저점
df['high_50'] = df['high'].rolling(50).max()
df['low_50'] = df['low'].rolling(50).min()
df['high_100'] = df['high'].rolling(100).max()
df['low_100'] = df['low'].rolling(100).min()
df['high_200'] = df['high'].rolling(200).max()
df['low_200'] = df['low'].rolling(200).min()

# 가격 위치 (0~1)
df['position_50'] = (df['close'] - df['low_50']) / (df['high_50'] - df['low_50'])
df['position_100'] = (df['close'] - df['low_100']) / (df['high_100'] - df['low_100'])
df['position_200'] = (df['close'] - df['low_200']) / (df['high_200'] - df['low_200'])

# 추세: HH/HL vs LH/LL (200봉 기준)
window = 100
df['prev_high'] = df['high'].shift(window).rolling(window).max()
df['prev_low'] = df['low'].shift(window).rolling(window).min()
df['curr_high'] = df['high'].rolling(window).max()
df['curr_low'] = df['low'].rolling(window).min()

df['HH'] = df['curr_high'] > df['prev_high']
df['HL'] = df['curr_low'] > df['prev_low']
df['LH'] = df['curr_high'] < df['prev_high']
df['LL'] = df['curr_low'] < df['prev_low']

df['swing_up'] = df['HH'] & df['HL']  # 계단식 상승
df['swing_down'] = df['LH'] & df['LL']  # 계단식 하락

# EMA 배열
df['ema_bull'] = (df['EMA20'] > df['EMA50']) & (df['EMA50'] > df['EMA200'])
df['ema_bear'] = (df['EMA20'] < df['EMA50']) & (df['EMA50'] < df['EMA200'])

print("=" * 70)
print("수축 → 발산 진입점 확장")
print("=" * 70)

# 기존: BB30 수축 후 발산
# 확장: 다양한 조건으로 진입점 찾기

# 수축 종료 시점 찾기
df['squeeze_end'] = (df['squeeze'].shift(1) == True) & (df['squeeze'] == False)

squeeze_ends = df[df['squeeze_end'] == True].copy()
print(f"수축 종료 시점: {len(squeeze_ends)}개")

# 각 수축 종료 시점에서 미래 수익 계산
def calculate_future_return(idx, df, holding_periods=[24, 48, 72, 168, 336]):
    """다양한 홀딩 기간별 수익 계산"""
    results = {}
    entry_price = df.loc[idx, 'close']
    
    for period in holding_periods:
        exit_idx = min(idx + period, len(df) - 1)
        exit_price = df.loc[exit_idx, 'close']
        
        # LONG 수익
        long_pnl = (exit_price - entry_price) / entry_price * 100
        results[f'long_{period}h'] = long_pnl
        
        # 기간 내 최대 수익/손실
        if exit_idx > idx:
            period_data = df.loc[idx:exit_idx]
            max_price = period_data['high'].max()
            min_price = period_data['low'].min()
            results[f'max_profit_{period}h'] = (max_price - entry_price) / entry_price * 100
            results[f'max_loss_{period}h'] = (min_price - entry_price) / entry_price * 100
    
    return results

# 모든 수축 종료 시점 분석
print("\n분석 중...")
all_results = []

for idx in squeeze_ends.index:
    row = df.loc[idx]
    
    future = calculate_future_return(idx, df)
    
    all_results.append({
        'datetime': row['datetime'],
        'close': row['close'],
        # 진입 시점 지표
        'momentum_50': row['momentum_50'],
        'momentum_100': row['momentum_100'],
        'momentum_200': row['momentum_200'],
        'position_50': row['position_50'],
        'position_100': row['position_100'],
        'position_200': row['position_200'],
        'swing_up': row['swing_up'],
        'swing_down': row['swing_down'],
        'ema_bull': row['ema_bull'],
        'ema_bear': row['ema_bear'],
        'above_ema200': row['close'] > row['EMA200'],
        'HH': row['HH'],
        'HL': row['HL'],
        'LH': row['LH'],
        'LL': row['LL'],
        # 미래 수익
        **future
    })

result_df = pd.DataFrame(all_results)
result_df = result_df.dropna()

print(f"분석 완료: {len(result_df)}건")
print()

# 168시간(1주일) 홀딩 기준 분석
print("=" * 70)
print("168시간(1주일) 홀딩 기준 분석")
print("=" * 70)

# 전체
total_win = (result_df['long_168h'] > 0).mean() * 100
total_avg = result_df['long_168h'].mean()
print(f"\n[전체] {len(result_df)}건")
print(f"  LONG 승률: {total_win:.1f}%, 평균: {total_avg:.2f}%")

# 조건별 분석
print("\n=== 단일 조건 ===")

conditions = [
    ('swing_up (HH+HL)', result_df['swing_up'] == True),
    ('swing_down (LH+LL)', result_df['swing_down'] == True),
    ('EMA 정배열', result_df['ema_bull'] == True),
    ('EMA 역배열', result_df['ema_bear'] == True),
    ('EMA200 위', result_df['above_ema200'] == True),
    ('EMA200 아래', result_df['above_ema200'] == False),
    ('모멘텀100 > 10%', result_df['momentum_100'] > 10),
    ('모멘텀100 > 20%', result_df['momentum_100'] > 20),
    ('모멘텀200 > 20%', result_df['momentum_200'] > 20),
    ('가격위치 상단 30%', result_df['position_100'] > 0.7),
    ('가격위치 상단 20%', result_df['position_100'] > 0.8),
    ('가격위치 하단 30%', result_df['position_100'] < 0.3),
    ('HH만', result_df['HH'] == True),
    ('HL만', result_df['HL'] == True),
]

for name, cond in conditions:
    subset = result_df[cond]
    if len(subset) < 20:
        continue
    
    win_rate = (subset['long_168h'] > 0).mean() * 100
    avg_pnl = subset['long_168h'].mean()
    max_profit = subset['max_profit_168h'].mean()
    
    marker = "⭐" if win_rate >= 60 and avg_pnl >= 3 else ""
    print(f"{name}: {len(subset)}건, 승률 {win_rate:.1f}%, 평균 {avg_pnl:.2f}%, max {max_profit:.1f}% {marker}")

print("\n=== 복합 조건 ===")

combos = [
    ('HH + HL (계단상승)', (result_df['HH'] == True) & (result_df['HL'] == True)),
    ('HH + EMA200↑', (result_df['HH'] == True) & (result_df['above_ema200'] == True)),
    ('HH + HL + EMA200↑', (result_df['HH'] == True) & (result_df['HL'] == True) & (result_df['above_ema200'] == True)),
    ('HH + 정배열', (result_df['HH'] == True) & (result_df['ema_bull'] == True)),
    ('HH + HL + 정배열', (result_df['HH'] == True) & (result_df['HL'] == True) & (result_df['ema_bull'] == True)),
    ('모멘텀100>10 + HH', (result_df['momentum_100'] > 10) & (result_df['HH'] == True)),
    ('모멘텀100>10 + HH + HL', (result_df['momentum_100'] > 10) & (result_df['HH'] == True) & (result_df['HL'] == True)),
    ('상단30% + HH', (result_df['position_100'] > 0.7) & (result_df['HH'] == True)),
    ('상단30% + HH + HL', (result_df['position_100'] > 0.7) & (result_df['HH'] == True) & (result_df['HL'] == True)),
    ('상단30% + 정배열', (result_df['position_100'] > 0.7) & (result_df['ema_bull'] == True)),
    ('모멘텀200>20 + HH', (result_df['momentum_200'] > 20) & (result_df['HH'] == True)),
]

for name, cond in combos:
    subset = result_df[cond]
    if len(subset) < 20:
        continue
    
    win_rate = (subset['long_168h'] > 0).mean() * 100
    avg_pnl = subset['long_168h'].mean()
    max_profit = subset['max_profit_168h'].mean()
    total_pnl = subset['long_168h'].sum()
    
    marker = "⭐" if win_rate >= 60 and avg_pnl >= 3 else ""
    print(f"{name}")
    print(f"  {len(subset)}건, 승률 {win_rate:.1f}%, 평균 {avg_pnl:.2f}%, 총 {total_pnl:.1f}% {marker}")

# 저장
result_df.to_csv('expanded_squeeze_analysis.csv', index=False)
print()
print("✅ 저장: expanded_squeeze_analysis.csv")

