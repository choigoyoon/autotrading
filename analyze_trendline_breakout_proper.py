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

def find_swing_lows(df, window=10):
    lows = []
    for i in range(window, len(df) - window):
        if df.loc[i, 'low'] == df.loc[i-window:i+window+1, 'low'].min():
            lows.append({
                'index': i,
                'time': df.loc[i, 'timestamp'],
                'price': df.loc[i, 'low']
            })
    return lows

highs = find_swing_highs(df)
lows = find_swing_lows(df)

print(f"📈 H값: {len(highs):,}개")
print(f"📉 L값: {len(lows):,}개\n")

# 하락 추세선 찾기 (LH-LH-LH)
downtrends = []
for i in range(len(highs) - 2):
    h1, h2, h3 = highs[i], highs[i+1], highs[i+2]
    
    # LH 패턴 확인
    if h1['price'] > h2['price'] > h3['price']:
        if h3['index'] - h1['index'] < 500:  # 최대 500캔들
            downtrends.append({
                'h1': h1, 'h2': h2, 'h3': h3,
                'start_idx': h1['index'],
                'end_idx': h3['index']
            })

print(f"🔽 하락 추세선 발견: {len(downtrends):,}개\n")

# 추세선 돌파 분석
breakout_cases = []

for dt_idx, dt in enumerate(downtrends[:100], 1):  # 상위 100개 분석
    if dt_idx % 20 == 0:
        print(f"⏳ 분석 중... {dt_idx}/100")
    
    h1, h2, h3 = dt['h1'], dt['h2'], dt['h3']
    
    # 추세선 돌파 찾기 (H3 이후 가격이 H3보다 높아지는 순간)
    breakout_idx = None
    breakout_price = None
    
    for i in range(dt['end_idx'] + 1, min(dt['end_idx'] + 300, len(df))):
        if df.loc[i, 'close'] > h3['price']:
            breakout_idx = i
            breakout_price = df.loc[i, 'close']
            break
    
    if not breakout_idx:
        continue
    
    breakout_time = df.loc[breakout_idx, 'timestamp']
    
    # ===== 핵심: H값들의 역할 전환 분석 =====
    
    # 롱 진입점: 돌파 시점
    entry_price = breakout_price
    entry_time = breakout_time
    
    # SL 계산: 가장 가까운 H값 아래 (저항→지지 전환 실패 시)
    # H3가 가장 가까움 → H3 아래가 SL
    sl_price = h3['price'] * 0.995  # H3 - 0.5%
    
    # TP 계산: 다음 저항선 찾기
    # 1) H2 (이전 저항선)
    # 2) H1 (더 이전 저항선)
    tp1_price = h2['price']
    tp2_price = h1['price']
    
    # 돌파 후 100캔들(25시간) 동안 추적
    next_candles = df.loc[breakout_idx:breakout_idx+101]
    
    if len(next_candles) < 50:
        continue
    
    # SL 체크
    sl_hit = False
    sl_hit_time = None
    for idx, row in next_candles.iterrows():
        if row['low'] < sl_price:
            sl_hit = True
            sl_hit_time = row['timestamp']
            break
    
    # TP1 체크
    tp1_hit = False
    tp1_hit_time = None
    for idx, row in next_candles.iterrows():
        if row['high'] >= tp1_price:
            tp1_hit = True
            tp1_hit_time = row['timestamp']
            break
    
    # TP2 체크
    tp2_hit = False
    tp2_hit_time = None
    for idx, row in next_candles.iterrows():
        if row['high'] >= tp2_price:
            tp2_hit = True
            tp2_hit_time = row['timestamp']
            break
    
    # H값 역할 전환 확인
    # H3가 지지선으로 작용했는가?
    h3_as_support = False
    for i in range(breakout_idx, min(breakout_idx + 100, len(df))):
        if abs(df.loc[i, 'low'] - h3['price']) / h3['price'] < 0.01:  # H3 ±1% 터치
            # 터치 후 반등했는가?
            next_5 = df.loc[i+1:i+6, 'close'].mean()
            if next_5 > df.loc[i, 'close']:
                h3_as_support = True
                break
    
    # 최종 결과
    if sl_hit:
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
    
    breakout_cases.append({
        'case_num': dt_idx,
        'h1_price': h1['price'],
        'h1_time': h1['time'],
        'h2_price': h2['price'],
        'h2_time': h2['time'],
        'h3_price': h3['price'],
        'h3_time': h3['time'],
        'breakout_time': breakout_time,
        'entry_price': entry_price,
        'sl_price': sl_price,
        'tp1_price': tp1_price,
        'tp2_price': tp2_price,
        'rr_tp1': rr_tp1,
        'rr_tp2': rr_tp2,
        'h3_as_support': h3_as_support,
        'result': result,
        'pnl_pct': pnl_pct,
        'sl_hit': sl_hit,
        'tp1_hit': tp1_hit,
        'tp2_hit': tp2_hit
    })

print(f"\n✅ 분석 완료!\n")

# DataFrame 생성
df_result = pd.DataFrame(breakout_cases)

# 통계
print(f"{'='*80}")
print(f"📊 추세선 돌파 후 H값 역할 전환 분석")
print(f"{'='*80}\n")

print(f"총 돌파 사례: {len(df_result)}개\n")

# 결과 분포
print(f"📈 결과 분포:")
result_counts = df_result['result'].value_counts()
for res, count in result_counts.items():
    pct = count / len(df_result) * 100
    print(f"   {res}: {count}개 ({pct:.1f}%)")

# H3 지지선 전환
h3_support_count = df_result['h3_as_support'].sum()
print(f"\n🔄 H3 저항→지지 전환 성공: {h3_support_count}개 ({h3_support_count/len(df_result)*100:.1f}%)")

# TP 승률
tp1_wins = len(df_result[df_result['tp1_hit'] == True])
tp2_wins = len(df_result[df_result['tp2_hit'] == True])
sl_losses = len(df_result[df_result['sl_hit'] == True])

print(f"\n🎯 매매 결과:")
print(f"   TP1 도달: {tp1_wins}개 ({tp1_wins/len(df_result)*100:.1f}%)")
print(f"   TP2 도달: {tp2_wins}개 ({tp2_wins/len(df_result)*100:.1f}%)")
print(f"   SL 손절: {sl_losses}개 ({sl_losses/len(df_result)*100:.1f}%)")

# 평균 수익률
avg_pnl = df_result['pnl_pct'].mean()
tp1_avg_pnl = df_result[df_result['tp1_hit']]['pnl_pct'].mean()
tp2_avg_pnl = df_result[df_result['tp2_hit']]['pnl_pct'].mean()
sl_avg_pnl = df_result[df_result['sl_hit']]['pnl_pct'].mean()

print(f"\n💰 평균 수익률:")
print(f"   전체 평균: {avg_pnl:.2f}%")
print(f"   TP1 평균: {tp1_avg_pnl:.2f}%")
print(f"   TP2 평균: {tp2_avg_pnl:.2f}%")
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
    print(f"\n사례 #{row['case_num']}: {row['breakout_time']}")
    print(f"   추세선: H1 ${row['h1_price']:,.0f} → H2 ${row['h2_price']:,.0f} → H3 ${row['h3_price']:,.0f}")
    print(f"   진입: ${row['entry_price']:,.0f}")
    print(f"   SL: ${row['sl_price']:,.0f} | TP1: ${row['tp1_price']:,.0f} (R:R 1:{row['rr_tp1']:.2f}) | TP2: ${row['tp2_price']:,.0f} (R:R 1:{row['rr_tp2']:.2f})")
    print(f"   H3 지지 전환: {'✅' if row['h3_as_support'] else '❌'}")
    print(f"   결과: {row['result']} | 수익: {row['pnl_pct']:.2f}%")

# CSV 저장
df_result.to_csv('trendline_breakout_hl_analysis.csv', index=False)
print(f"\n✅ 결과 저장: trendline_breakout_hl_analysis.csv")

# 핵심 인사이트
print(f"\n{'='*80}")
print(f"💡 핵심 인사이트")
print(f"{'='*80}")

print(f"\n1️⃣ H값의 역할 전환:")
print(f"   하락 추세선(H1→H2→H3) 돌파 후")
print(f"   H3가 저항→지지로 전환: {h3_support_count/len(df_result)*100:.1f}%")

print(f"\n2️⃣ SL 위치:")
print(f"   가장 가까운 저항선(H3) 아래 -0.5%")
print(f"   SL 터진 확률: {sl_losses/len(df_result)*100:.1f}%")

print(f"\n3️⃣ TP 위치:")
print(f"   TP1 (H2): 도달 확률 {tp1_wins/len(df_result)*100:.1f}% | 평균 R:R 1:{avg_rr_tp1:.2f}")
print(f"   TP2 (H1): 도달 확률 {tp2_wins/len(df_result)*100:.1f}% | 평균 R:R 1:{avg_rr_tp2:.2f}")

win_rate = (tp1_wins + tp2_wins) / len(df_result) * 100
print(f"\n4️⃣ 전체 승률: {win_rate:.1f}% (TP 도달)")
print(f"   평균 수익: {avg_pnl:.2f}%")

