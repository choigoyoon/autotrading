import pandas as pd
import numpy as np

print("=" * 80)
print("전략 확장: 롱 + 숏 시그널 (기존 방법 활용)")
print("=" * 80)

# 기존 롱 시그널 로드
long_signals = pd.read_csv('valid_signals.csv')
long_signals['direction'] = 'long'

print(f"\n✅ 기존 롱 시그널: {len(long_signals)}건")

# 숏 시그널 생성 (L-L 상승 추세선 돌파 하향)
# 간단하게: 기존 로직을 반대로 적용
candles_df = pd.read_csv('analysis_15m.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])

print("\n⏳ 숏 시그널 찾는 중 (L-L 상승 추세선 하향 돌파)...")

# L 값 추출
L_values = []
for idx, row in candles_df.iterrows():
    if pd.notna(row['swing_low']):
        L_values.append({
            'time': row['datetime'],
            'price': row['swing_low']
        })

L_df = pd.DataFrame(L_values)
print(f"  L 값: {len(L_df)}개")

# 간단한 상승 추세선 찾기 (샘플링으로 속도 향상)
short_signals_list = []

# 매 100개 L값마다 샘플링
for i in range(0, len(L_df), 10):
    if i + 1 >= len(L_df):
        break
    
    l1 = L_df.iloc[i]
    
    # 다음 20-50개 L값 중에서 찾기
    for j in range(i+20, min(i+50, len(L_df))):
        l2 = L_df.iloc[j]
        
        # 상승선인지 확인 (양의 기울기)
        if l2['price'] <= l1['price']:
            continue
        
        # 시간 차이 확인
        time_diff = (l2['time'] - l1['time']).total_seconds() / 3600
        if time_diff < 8:  # 최소 8시간
            continue
        
        # 간단한 돌파 확인 (L2 이후 20캔들 내)
        l2_idx = candles_df[candles_df['datetime'] == l2['time']].index[0]
        future = candles_df.iloc[l2_idx+1:l2_idx+21]
        
        for idx, candle in future.iterrows():
            # 추세선 계산
            hours = (candle['datetime'] - l1['time']).total_seconds() / 3600
            slope = (l2['price'] - l1['price']) / time_diff
            trendline = l1['price'] + slope * hours
            
            # 하향 돌파 체크
            if candle['close'] < trendline:
                gap_pct = (trendline - candle['close']) / trendline * 100
                
                short_signals_list.append({
                    'direction': 'short',
                    'h1_time': l1['time'],
                    'h1_price': l1['price'],
                    'h2_time': l2['time'],
                    'h2_price': l2['price'],
                    'breakout_time': candle['datetime'],
                    'breakout_price': candle['close'],
                    'trendline_price': trendline,
                    'gap_pct': gap_pct,
                    'hl_confirm_time': candle['datetime'],  # 간소화
                    'hl_price': candle['close']
                })
                break

short_signals = pd.DataFrame(short_signals_list)
print(f"  → {len(short_signals)}개 발견")

# 통합
all_signals = pd.concat([long_signals, short_signals], ignore_index=True)

print("\n" + "=" * 80)
print("시그널 통계")
print("=" * 80)

print(f"\n총 시그널: {len(all_signals)}개")
print(f"  롱: {len(long_signals)}개 ({len(long_signals)/len(all_signals)*100:.1f}%)")
print(f"  숏: {len(short_signals)}개 ({len(short_signals)/len(all_signals)*100:.1f}%)")

years = 5.7
print(f"\n연평균: {len(all_signals)/years:.1f}개")
print(f"  롱: {len(long_signals)/years:.1f}개")
print(f"  숏: {len(short_signals)/years:.1f}개")
print(f"월평균: {len(all_signals)/years/12:.1f}개")

# 저장
all_signals.to_csv('all_signals_long_short.csv', index=False)
print("\n✅ 저장: all_signals_long_short.csv")

# 간단 백테스트
print("\n" + "=" * 80)
print("간단 백테스트 (TP 5%, SL 2%, 72h)")
print("=" * 80)

def quick_backtest(signal):
    direction = signal['direction']
    entry_time = pd.to_datetime(signal['breakout_time'])
    entry_price = signal['breakout_price']
    
    entry_idx = candles_df[candles_df['datetime'] == entry_time].index
    if len(entry_idx) == 0:
        return None
    
    future = candles_df.iloc[entry_idx[0]+1:entry_idx[0]+289]
    
    if len(future) == 0:
        return None
    
    for idx, candle in future.iterrows():
        if direction == 'long':
            if candle['high'] >= entry_price * 1.05:
                return {'direction': direction, 'exit': 'TP', 'pnl': 5.0}
            if candle['low'] <= entry_price * 0.98:
                return {'direction': direction, 'exit': 'SL', 'pnl': -2.0}
        else:
            if candle['low'] <= entry_price * 0.95:
                return {'direction': direction, 'exit': 'TP', 'pnl': 5.0}
            if candle['high'] >= entry_price * 1.02:
                return {'direction': direction, 'exit': 'SL', 'pnl': -2.0}
    
    last = future.iloc[-1]
    if direction == 'long':
        pnl = (last['close'] - entry_price) / entry_price * 100
    else:
        pnl = (entry_price - last['close']) / entry_price * 100
    
    return {'direction': direction, 'exit': 'TIME', 'pnl': pnl}

results = []
for idx, signal in all_signals.iterrows():
    result = quick_backtest(signal)
    if result:
        results.append(result)

results_df = pd.DataFrame(results)

print(f"\n백테스트 결과: {len(results)}건")

for direction in ['long', 'short']:
    subset = results_df[results_df['direction'] == direction]
    if len(subset) > 0:
        wins = (subset['pnl'] > 0).sum()
        tp = (subset['exit'] == 'TP').sum()
        
        print(f"\n{direction.upper()} ({len(subset)}건):")
        print(f"  평균 수익: {subset['pnl'].mean():.2f}%")
        print(f"  승률: {wins/len(subset)*100:.1f}% ({wins}승)")
        print(f"  TP: {tp}건 ({tp/len(subset)*100:.1f}%)")

# 전체
wins = (results_df['pnl'] > 0).sum()
tp = (results_df['exit'] == 'TP').sum()

print(f"\n전체 통합:")
print(f"  평균 수익: {results_df['pnl'].mean():.2f}%")
print(f"  승률: {wins/len(results)*100:.1f}%")
print(f"  TP: {tp}건 ({tp/len(results)*100:.1f}%)")

# 복리
capital = 100
for pnl in results_df['pnl']:
    capital *= (1 + pnl / 100)

cagr = (capital / 100) ** (1 / years) - 1

print(f"\n복리:")
print(f"  초기: 100 → 최종: {capital:.2f}")
print(f"  총수익: {(capital-100):.2f}%")
print(f"  연평균: {cagr*100:.2f}%")

print("\n" + "=" * 80)
print("비교: 롱 Only vs 롱+숏")
print("=" * 80)

long_only = results_df[results_df['direction'] == 'long']
capital_long = 100
for pnl in long_only['pnl']:
    capital_long *= (1 + pnl / 100)
cagr_long = (capital_long / 100) ** (1 / years) - 1

print(f"\n롱 Only:")
print(f"  시그널: {len(long_only)}건 (연 {len(long_only)/years:.1f}건)")
print(f"  평균 수익: {long_only['pnl'].mean():.2f}%")
print(f"  연평균: {cagr_long*100:.2f}%")

print(f"\n롱+숏:")
print(f"  시그널: {len(results)}건 (연 {len(results)/years:.1f}건)")
print(f"  평균 수익: {results_df['pnl'].mean():.2f}%")
print(f"  연평균: {cagr*100:.2f}%")

increase = len(results) - len(long_only)
print(f"\n개선:")
print(f"  시그널 +{increase}건 (+{increase/len(long_only)*100:.1f}%)")
print(f"  연평균 {cagr_long*100:.2f}% → {cagr*100:.2f}% ({cagr*100-cagr_long*100:+.2f}%p)")

