import pandas as pd
import numpy as np

# CSV 읽기
df = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df = df.rename(columns={'datetime': 'timestamp'})
df = df.sort_values('timestamp').reset_index(drop=True)

# Swing High/Low 찾기
def find_swing_highs(df, window=10):
    highs = []
    for i in range(window, len(df) - window):
        if df.loc[i, 'high'] == df.loc[i-window:i+window+1, 'high'].max():
            highs.append({'index': i, 'time': df.loc[i, 'timestamp'], 'price': df.loc[i, 'high']})
    return highs

def find_swing_lows(df, window=10):
    lows = []
    for i in range(window, len(df) - window):
        if df.loc[i, 'low'] == df.loc[i-window:i+window+1, 'low'].min():
            lows.append({'index': i, 'time': df.loc[i, 'timestamp'], 'price': df.loc[i, 'low']})
    return lows

highs = find_swing_highs(df)
lows = find_swing_lows(df)

print(f"📊 총 {len(highs)}개 H값, {len(lows)}개 L값 발견\n")

# 하락 추세선 찾기
downtrends = []
for i in range(len(highs) - 2):
    h1, h2, h3 = highs[i], highs[i+1], highs[i+2]
    
    if h1['price'] > h2['price'] > h3['price']:
        if h3['index'] - h1['index'] < 500:  # 최대 500캔들
            downtrends.append({
                'h1': h1, 'h2': h2, 'h3': h3,
                'start_idx': h1['index'],
                'end_idx': h3['index']
            })

print(f"🔽 총 {len(downtrends)}개 하락 추세선 발견\n")

# 추세선 돌파 후 H값들이 저항선으로 작용하는지 분석
breakout_analysis = []

for dt_idx, dt in enumerate(downtrends[:20], 1):  # 상위 20개만
    h1, h2, h3 = dt['h1'], dt['h2'], dt['h3']
    
    # 추세선 돌파 찾기
    breakout_idx = None
    for i in range(dt['end_idx'] + 1, min(dt['end_idx'] + 200, len(df))):
        if df.loc[i, 'close'] > h3['price']:
            breakout_idx = i
            break
    
    if not breakout_idx:
        continue
    
    breakout_time = df.loc[breakout_idx, 'timestamp']
    breakout_price = df.loc[breakout_idx, 'close']
    
    print(f"\n{'='*80}")
    print(f"📈 추세선 돌파 사례 #{dt_idx}")
    print(f"{'='*80}")
    print(f"🔴 하락 추세선 H값들 (저항선으로 작용할 가로선들):")
    print(f"   H1: ${h1['price']:,.2f} ({h1['time']})")
    print(f"   H2: ${h2['price']:,.2f} ({h2['time']}) [LH: {(h2['price']/h1['price']-1)*100:.2f}%]")
    print(f"   H3: ${h3['price']:,.2f} ({h3['time']}) [LH: {(h3['price']/h2['price']-1)*100:.2f}%]")
    print(f"\n💥 추세선 돌파: {breakout_time} @ ${breakout_price:,.2f}")
    
    # 돌파 후 각 H값 가로선 테스트
    print(f"\n🧪 돌파 후 각 H값 저항선 테스트:")
    
    h_levels = [
        ('H3', h3['price']),
        ('H2', h2['price']),
        ('H1', h1['price'])
    ]
    
    resistance_tests = []
    
    for h_name, h_price in h_levels:
        # 돌파 후 이 가격대 근처(±0.5%)에 접근했는지 확인
        for i in range(breakout_idx, min(breakout_idx + 300, len(df))):
            high = df.loc[i, 'high']
            close = df.loc[i, 'close']
            low = df.loc[i, 'low']
            
            # 저항선 테스트: 캔들이 H값에 접근
            if abs(high - h_price) / h_price < 0.005:  # ±0.5%
                test_time = df.loc[i, 'timestamp']
                
                # 저항 여부 확인: 다음 5캔들 평균 종가
                next_candles = df.loc[i+1:i+6, 'close'].mean()
                
                if next_candles < close:  # 저항 성공
                    result = "🔴 저항 (하락)"
                    resistance_tests.append({
                        'h_name': h_name,
                        'h_price': h_price,
                        'test_time': test_time,
                        'test_high': high,
                        'result': '저항',
                        'next_close': next_candles
                    })
                    print(f"   {h_name} ${h_price:,.2f} ━━━ {test_time}: 고가 ${high:,.2f} → 🔴 저항선 작용! (다음 평균: ${next_candles:,.2f})")
                else:  # 돌파 성공
                    result = "✅ 돌파 (상승)"
                    resistance_tests.append({
                        'h_name': h_name,
                        'h_price': h_price,
                        'test_time': test_time,
                        'test_high': high,
                        'result': '돌파',
                        'next_close': next_candles
                    })
                    print(f"   {h_name} ${h_price:,.2f} ━━━ {test_time}: 고가 ${high:,.2f} → ✅ 저항선 돌파! (다음 평균: ${next_candles:,.2f})")
                
                break  # 첫 번째 테스트만
    
    if resistance_tests:
        breakout_analysis.append({
            'breakout_time': breakout_time,
            'breakout_price': breakout_price,
            'h1': h1['price'],
            'h2': h2['price'],
            'h3': h3['price'],
            'resistance_tests': resistance_tests
        })

# 통계
print(f"\n\n{'='*80}")
print(f"📊 통계 요약")
print(f"{'='*80}")

total_tests = sum(len(case['resistance_tests']) for case in breakout_analysis)
resistance_success = sum(1 for case in breakout_analysis for test in case['resistance_tests'] if test['result'] == '저항')
breakout_success = total_tests - resistance_success

print(f"총 추세선 돌파 사례: {len(breakout_analysis)}개")
print(f"총 H값 저항선 테스트: {total_tests}회")
print(f"  🔴 저항 성공: {resistance_success}회 ({resistance_success/total_tests*100:.1f}%)")
print(f"  ✅ 돌파 성공: {breakout_success}회 ({breakout_success/total_tests*100:.1f}%)")

print(f"\n💡 핵심 인사이트:")
print(f"   추세선을 형성한 H값들(LH, LH, LH)은")
print(f"   추세선 돌파 후에도 **{resistance_success/total_tests*100:.1f}%** 확률로")
print(f"   **가로선 저항선**으로 작용합니다!")
print(f"\n   즉, H3 → H2 → H1 순서대로")
print(f"   각 가로선을 돌파해야 진짜 상승 확정!")

# CSV 저장
results_df = pd.DataFrame([
    {
        'breakout_time': case['breakout_time'],
        'breakout_price': case['breakout_price'],
        'h1_price': case['h1'],
        'h2_price': case['h2'],
        'h3_price': case['h3'],
        'test_h_name': test['h_name'],
        'test_h_price': test['h_price'],
        'test_time': test['test_time'],
        'test_result': test['result']
    }
    for case in breakout_analysis
    for test in case['resistance_tests']
])

results_df.to_csv('h_resistance_after_breakout.csv', index=False)
print(f"\n✅ 결과 저장: h_resistance_after_breakout.csv")

