import pandas as pd
import numpy as np

# CSV 읽기
df = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df = df.rename(columns={'datetime': 'timestamp'})
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"📊 데이터 로드: {len(df):,}개 캔들\n")

# Swing High/Low 찾기
def find_swing_highs(df, window=10):
    highs = []
    for i in range(window, len(df) - window):
        if df.loc[i, 'high'] == df.loc[i-window:i+window+1, 'high'].max():
            highs.append({
                'index': i,
                'time': df.loc[i, 'timestamp'],
                'price': df.loc[i, 'high']
            })
    return highs

highs = find_swing_highs(df)
print(f"📈 H값: {len(highs):,}개\n")

# 하락 추세선 찾기 (LH-LH-LH)
downtrends = []
for i in range(len(highs) - 2):
    h1, h2, h3 = highs[i], highs[i+1], highs[i+2]
    
    if h1['price'] > h2['price'] > h3['price']:
        if h3['index'] - h1['index'] < 500:
            downtrends.append({
                'h1': h1, 'h2': h2, 'h3': h3,
                'start_idx': h1['index'],
                'end_idx': h3['index']
            })

print(f"🔽 하락 추세선: {len(downtrends):,}개\n")

# 리테스트 진입 전략 분석
retest_cases = []

for dt_idx, dt in enumerate(downtrends[:100], 1):
    if dt_idx % 20 == 0:
        print(f"⏳ 분석 중... {dt_idx}/100")
    
    h1, h2, h3 = dt['h1'], dt['h2'], dt['h3']
    
    # 1단계: 추세선 돌파 찾기
    breakout_idx = None
    for i in range(dt['end_idx'] + 1, min(dt['end_idx'] + 300, len(df))):
        if df.loc[i, 'close'] > h3['price']:
            breakout_idx = i
            break
    
    if not breakout_idx:
        continue
    
    breakout_time = df.loc[breakout_idx, 'timestamp']
    breakout_price = df.loc[breakout_idx, 'close']
    
    # 2단계: H3 리테스트 찾기
    # 돌파 후 50캔들 이내에 H3 ±1% 범위 터치하는지 확인
    retest_idx = None
    retest_price = None
    retest_confirmed = False
    
    for i in range(breakout_idx + 1, min(breakout_idx + 51, len(df))):
        low = df.loc[i, 'low']
        close = df.loc[i, 'close']
        
        # H3 ±1% 범위 터치
        if abs(low - h3['price']) / h3['price'] < 0.01:
            retest_idx = i
            retest_price = df.loc[i, 'close']
            
            # 3단계: 리테스트 후 버티는지 확인 (다음 5캔들 평균이 H3 위)
            next_5_candles = df.loc[i+1:i+6]
            if len(next_5_candles) >= 5:
                avg_close = next_5_candles['close'].mean()
                if avg_close > h3['price']:
                    retest_confirmed = True
                    break
    
    if not retest_confirmed:
        continue
    
    # 4단계: 리테스트 확인 후 진입
    entry_idx = retest_idx + 1  # 리테스트 확인 후 다음 캔들
    entry_price = df.loc[entry_idx, 'open']
    entry_time = df.loc[entry_idx, 'timestamp']
    
    # SL: H3 아래 -0.5%
    sl_price = h3['price'] * 0.995
    
    # TP1: H2 (첫 번째 저항선)
    # TP2: H1 (두 번째 저항선)
    tp1_price = h2['price']
    tp2_price = h1['price']
    
    # 진입 후 100캔들 추적
    next_candles = df.loc[entry_idx:entry_idx+101]
    
    if len(next_candles) < 50:
        continue
    
    # SL 체크
    sl_hit = False
    sl_hit_time = None
    sl_hit_idx = None
    for idx, row in next_candles.iterrows():
        if row['low'] < sl_price:
            sl_hit = True
            sl_hit_time = row['timestamp']
            sl_hit_idx = idx
            break
    
    # TP1 체크
    tp1_hit = False
    tp1_hit_time = None
    tp1_hit_idx = None
    for idx, row in next_candles.iterrows():
        if sl_hit and idx >= sl_hit_idx:
            break  # SL 먼저 맞으면 TP 체크 중단
        if row['high'] >= tp1_price:
            tp1_hit = True
            tp1_hit_time = row['timestamp']
            tp1_hit_idx = idx
            break
    
    # TP2 체크
    tp2_hit = False
    tp2_hit_time = None
    for idx, row in next_candles.iterrows():
        if sl_hit and idx >= sl_hit_idx:
            break
        if row['high'] >= tp2_price:
            tp2_hit = True
            tp2_hit_time = row['timestamp']
            break
    
    # 최종 결과
    if sl_hit and (not tp1_hit or sl_hit_idx < tp1_hit_idx):
        result = 'SL'
        pnl_pct = (sl_price / entry_price - 1) * 100
    elif tp2_hit:
        result = 'TP2'
        pnl_pct = (tp2_price / entry_price - 1) * 100
    elif tp1_hit:
        result = 'TP1'
        pnl_pct = (tp1_price / entry_price - 1) * 100
    else:
        result = 'Open'
        final_price = next_candles.iloc[-1]['close']
        pnl_pct = (final_price / entry_price - 1) * 100
    
    # R:R 계산
    risk = abs(entry_price - sl_price)
    reward_tp1 = abs(tp1_price - entry_price)
    reward_tp2 = abs(tp2_price - entry_price)
    
    rr_tp1 = reward_tp1 / risk if risk > 0 else 0
    rr_tp2 = reward_tp2 / risk if risk > 0 else 0
    
    retest_cases.append({
        'case_num': dt_idx,
        'h1_price': h1['price'],
        'h1_time': h1['time'],
        'h2_price': h2['price'],
        'h2_time': h2['time'],
        'h3_price': h3['price'],
        'h3_time': h3['time'],
        'breakout_time': breakout_time,
        'breakout_price': breakout_price,
        'retest_time': df.loc[retest_idx, 'timestamp'],
        'retest_price': retest_price,
        'entry_time': entry_time,
        'entry_price': entry_price,
        'sl_price': sl_price,
        'tp1_price': tp1_price,
        'tp2_price': tp2_price,
        'rr_tp1': rr_tp1,
        'rr_tp2': rr_tp2,
        'result': result,
        'pnl_pct': pnl_pct,
        'sl_hit': sl_hit,
        'tp1_hit': tp1_hit,
        'tp2_hit': tp2_hit
    })

print(f"\n✅ 분석 완료!\n")

# DataFrame 생성
df_result = pd.DataFrame(retest_cases)

if len(df_result) == 0:
    print("❌ 리테스트 확인된 케이스 없음!")
    exit()

# 통계
print(f"{'='*80}")
print(f"🎯 리테스트 후 진입 전략 분석")
print(f"{'='*80}\n")

print(f"총 리테스트 확인 사례: {len(df_result)}개\n")

# 결과 분포
print(f"📈 결과 분포:")
result_counts = df_result['result'].value_counts()
for res, count in result_counts.items():
    pct = count / len(df_result) * 100
    print(f"   {res}: {count}개 ({pct:.1f}%)")

# TP 승률
tp1_wins = len(df_result[df_result['tp1_hit'] == True])
tp2_wins = len(df_result[df_result['tp2_hit'] == True])
sl_losses = len(df_result[df_result['sl_hit'] == True])

print(f"\n🎯 매매 결과:")
print(f"   ✅ TP1 도달: {tp1_wins}개 ({tp1_wins/len(df_result)*100:.1f}%)")
print(f"   ✅ TP2 도달: {tp2_wins}개 ({tp2_wins/len(df_result)*100:.1f}%)")
print(f"   🔴 SL 손절: {sl_losses}개 ({sl_losses/len(df_result)*100:.1f}%)")

# 승률 계산
win_count = tp1_wins + tp2_wins - tp2_wins  # TP1만 or TP2 도달
total_trades = len(df_result)
win_rate = ((tp1_wins + tp2_wins) / total_trades * 100) if total_trades > 0 else 0

print(f"\n💯 승률: {win_rate:.1f}%")

# 평균 수익률
avg_pnl = df_result['pnl_pct'].mean()
tp1_avg_pnl = df_result[df_result['tp1_hit']]['pnl_pct'].mean() if tp1_wins > 0 else 0
tp2_avg_pnl = df_result[df_result['tp2_hit']]['pnl_pct'].mean() if tp2_wins > 0 else 0
sl_avg_pnl = df_result[df_result['sl_hit']]['pnl_pct'].mean() if sl_losses > 0 else 0

print(f"\n💰 평균 수익률:")
print(f"   전체 평균: {avg_pnl:.2f}%")
if tp1_wins > 0:
    print(f"   TP1 평균: {tp1_avg_pnl:.2f}%")
if tp2_wins > 0:
    print(f"   TP2 평균: {tp2_avg_pnl:.2f}%")
if sl_losses > 0:
    print(f"   SL 평균: {sl_avg_pnl:.2f}%")

# 평균 R:R
avg_rr_tp1 = df_result['rr_tp1'].mean()
avg_rr_tp2 = df_result['rr_tp2'].mean()

print(f"\n📊 평균 R:R 비율:")
print(f"   TP1 R:R = 1:{avg_rr_tp1:.2f}")
print(f"   TP2 R:R = 1:{avg_rr_tp2:.2f}")

# Top 10 성공 사례
print(f"\n🔥 Top 10 성공 사례:")
top_cases = df_result.nlargest(10, 'pnl_pct')
for idx, row in top_cases.iterrows():
    print(f"\n사례 #{row['case_num']}: {row['entry_time']}")
    print(f"   추세선: H1 ${row['h1_price']:,.0f} → H2 ${row['h2_price']:,.0f} → H3 ${row['h3_price']:,.0f}")
    print(f"   돌파: {row['breakout_time']} @ ${row['breakout_price']:,.0f}")
    print(f"   리테스트: {row['retest_time']} @ ${row['retest_price']:,.0f}")
    print(f"   진입: ${row['entry_price']:,.0f}")
    print(f"   SL: ${row['sl_price']:,.0f} | TP1: ${row['tp1_price']:,.0f} (R:R 1:{row['rr_tp1']:.2f}) | TP2: ${row['tp2_price']:,.0f} (R:R 1:{row['rr_tp2']:.2f})")
    print(f"   결과: {row['result']} | 수익: {row['pnl_pct']:.2f}%")

# CSV 저장
df_result.to_csv('retest_entry_strategy_analysis.csv', index=False)
print(f"\n✅ 결과 저장: retest_entry_strategy_analysis.csv")

# 비교 분석
print(f"\n{'='*80}")
print(f"📊 전략 비교: 즉시 진입 vs 리테스트 후 진입")
print(f"{'='*80}\n")

print(f"❌ 즉시 진입 (이전 분석):")
print(f"   - 총 사례: 82개")
print(f"   - SL 손절: 76.8%")
print(f"   - 평균 수익: -0.37%")

print(f"\n✅ 리테스트 후 진입 (현재 분석):")
print(f"   - 총 사례: {len(df_result)}개")
print(f"   - SL 손절: {sl_losses/len(df_result)*100:.1f}%")
print(f"   - 승률: {win_rate:.1f}%")
print(f"   - 평균 수익: {avg_pnl:.2f}%")

improvement = avg_pnl - (-0.37)
print(f"\n🚀 개선도: +{improvement:.2f}%p")

# 핵심 인사이트
print(f"\n{'='*80}")
print(f"💡 핵심 인사이트")
print(f"{'='*80}")

print(f"\n1️⃣ 리테스트 전략의 핵심:")
print(f"   ① 추세선 돌파 확인")
print(f"   ② H3 리테스트 대기 (H3 ±1% 터치)")
print(f"   ③ 리테스트 후 버티는지 확인 (다음 5캔들 평균 > H3)")
print(f"   ④ 확인되면 진입!")

print(f"\n2️⃣ 진입 조건:")
print(f"   - H3 터치 후 반등 확인")
print(f"   - 다음 5캔들 평균이 H3 위에 있어야 함")

print(f"\n3️⃣ SL/TP:")
print(f"   - SL: H3 -0.5%")
print(f"   - TP1: H2 (평균 R:R 1:{avg_rr_tp1:.2f})")
print(f"   - TP2: H1 (평균 R:R 1:{avg_rr_tp2:.2f})")

print(f"\n4️⃣ 결과:")
print(f"   - 리테스트 확인 후 진입 시 승률: {win_rate:.1f}%")
print(f"   - 평균 수익: {avg_pnl:.2f}%")
print(f"   - 즉시 진입 대비 {improvement:.2f}%p 개선!")

