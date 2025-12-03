"""
H↑ L↓ 패턴에서 페이크(손절) vs 성공(TP) 비교
뭐가 달랐는지?
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
pattern_df = pd.read_csv('hl_pattern_analysis.csv')
signals = pd.read_csv('valid_signals.csv')
df_15m = pd.read_csv('analysis_15m.csv')

pattern_df['breakout_time'] = pd.to_datetime(pattern_df['breakout_time'])
signals['breakout_time'] = pd.to_datetime(signals['breakout_time'])
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])

print("="*80)
print("H↑ L↓ 패턴 - 페이크 vs 성공 분석")
print("="*80)

# H↑ L↓ 패턴만
best_pattern = pattern_df[pattern_df['pattern'] == 'H↑ L↓'].copy()
print(f"\nH↑ L↓ 패턴: {len(best_pattern)}건")

# 성공 vs 실패 분류
success = best_pattern[best_pattern['sl_done'] == False]  # TP 달성
fail = best_pattern[best_pattern['sl_done'] == True]  # 손절

print(f"  성공 (TP): {len(success)}건 ({len(success)/len(best_pattern)*100:.1f}%)")
print(f"  실패 (SL): {len(fail)}건 ({len(fail)/len(best_pattern)*100:.1f}%)")

# 비교 분석
print("\n" + "="*80)
print("📊 성공 vs 실패 비교")
print("="*80)

# 1. H 변화율
print(f"\n1️⃣ H(고점) 변화율")
print(f"  성공: {success['h_change'].mean():+.2f}%")
print(f"  실패: {fail['h_change'].mean():+.2f}%")

# 2. L 변화율
print(f"\n2️⃣ L(저점) 변화율")
print(f"  성공: {success['l_change'].mean():+.2f}%")
print(f"  실패: {fail['l_change'].mean():+.2f}%")

# 3. Entry - L 거리 (손절까지 거리)
# signals에서 매칭
success_merged = success.merge(signals[['breakout_time', 'breakout_price', 'gap_pct']], 
                                on='breakout_time', how='left', suffixes=('', '_sig'))
fail_merged = fail.merge(signals[['breakout_time', 'breakout_price', 'gap_pct']], 
                          on='breakout_time', how='left', suffixes=('', '_sig'))

# Entry - L2 거리
success_merged['entry_L_dist'] = (success_merged['entry_price'] - success_merged['l2']) / success_merged['entry_price'] * 100
fail_merged['entry_L_dist'] = (fail_merged['entry_price'] - fail_merged['l2']) / fail_merged['entry_price'] * 100

print(f"\n3️⃣ Entry - L 거리 (손절까지)")
print(f"  성공: {success_merged['entry_L_dist'].mean():.2f}%")
print(f"  실패: {fail_merged['entry_L_dist'].mean():.2f}%")

# 4. Gap (돌파 강도)
print(f"\n4️⃣ Gap (돌파 강도)")
print(f"  성공: {success_merged['gap_pct'].mean():.2f}%")
print(f"  실패: {fail_merged['gap_pct'].mean():.2f}%")

# 5. 돌파 전 캔들 분석 (거래량, 몸통 등)
print("\n" + "="*80)
print("📊 돌파 캔들 분석")
print("="*80)

def get_breakout_candle_info(breakout_time, df_15m):
    """돌파 캔들 정보"""
    candle = df_15m[df_15m['datetime'] == breakout_time]
    if len(candle) == 0:
        return None
    c = candle.iloc[0]
    body = abs(c['close'] - c['open'])
    wick_up = c['high'] - max(c['close'], c['open'])
    wick_down = min(c['close'], c['open']) - c['low']
    total_range = c['high'] - c['low']
    body_ratio = body / total_range if total_range > 0 else 0
    
    return {
        'volume': c['volume'],
        'body_ratio': body_ratio,
        'is_bullish': c['close'] > c['open'],
        'range_pct': total_range / c['close'] * 100
    }

# 성공 케이스 캔들 정보
success_candles = []
for _, row in success.iterrows():
    info = get_breakout_candle_info(row['breakout_time'], df_15m)
    if info:
        success_candles.append(info)

# 실패 케이스 캔들 정보  
fail_candles = []
for _, row in fail.iterrows():
    info = get_breakout_candle_info(row['breakout_time'], df_15m)
    if info:
        fail_candles.append(info)

if success_candles and fail_candles:
    success_candle_df = pd.DataFrame(success_candles)
    fail_candle_df = pd.DataFrame(fail_candles)
    
    print(f"\n5️⃣ 돌파 캔들 몸통 비율 (Body Ratio)")
    print(f"  성공: {success_candle_df['body_ratio'].mean():.2f}")
    print(f"  실패: {fail_candle_df['body_ratio'].mean():.2f}")
    
    print(f"\n6️⃣ 돌파 캔들 양봉 비율")
    print(f"  성공: {success_candle_df['is_bullish'].mean()*100:.1f}%")
    print(f"  실패: {fail_candle_df['is_bullish'].mean()*100:.1f}%")
    
    print(f"\n7️⃣ 돌파 캔들 변동폭")
    print(f"  성공: {success_candle_df['range_pct'].mean():.2f}%")
    print(f"  실패: {fail_candle_df['range_pct'].mean():.2f}%")

# 6. 시간대 분석
print("\n" + "="*80)
print("📊 시간대 분석")
print("="*80)

success['hour'] = success['breakout_time'].dt.hour
fail['hour'] = fail['breakout_time'].dt.hour

print(f"\n8️⃣ 돌파 시간대")
print(f"  성공 평균: {success['hour'].mean():.1f}시")
print(f"  실패 평균: {fail['hour'].mean():.1f}시")

# 결론
print("\n" + "="*80)
print("💡 페이크 원인 분석")
print("="*80)

print("""
성공 vs 실패 차이점:
""")

# 차이 계산
h_diff = success['h_change'].mean() - fail['h_change'].mean()
l_diff = abs(success['l_change'].mean()) - abs(fail['l_change'].mean())
gap_diff = success_merged['gap_pct'].mean() - fail_merged['gap_pct'].mean()
dist_diff = success_merged['entry_L_dist'].mean() - fail_merged['entry_L_dist'].mean()

if h_diff > 0:
    print(f"  ✅ H 상승폭 더 큼: 성공이 {h_diff:+.2f}% 더 상승")
if l_diff > 0:
    print(f"  ✅ L 하락폭 더 큼: 성공이 {l_diff:.2f}% 더 하락")
if gap_diff > 0:
    print(f"  ✅ Gap 더 큼: 성공이 {gap_diff:.2f}% 더 강한 돌파")
if dist_diff > 0:
    print(f"  ✅ Entry-L 거리 더 큼: 성공이 {dist_diff:.2f}% 더 여유")

