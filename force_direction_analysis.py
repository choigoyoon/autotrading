import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 80)
print("🔋 힘의 방향(Force Direction) 논리로 실패 사유 분석")
print("=" * 80)

# Load all data
df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m.sort_values('datetime').reset_index(drop=True)

df_1h = pd.read_csv('btc_1h_ohlcv.csv')
df_1h['datetime'] = pd.to_datetime(df_1h['datetime'])
df_1h = df_1h.sort_values('datetime').reset_index(drop=True)

df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])
df_4h = df_4h.sort_values('datetime').reset_index(drop=True)

trades = pd.read_csv('backtest_confirmation_space_results.csv')
trades['entry_time'] = pd.to_datetime(trades['entry_time'])

mtf_df = pd.read_csv('mtf_context_analysis.csv')
mtf_df['entry_time'] = pd.to_datetime(mtf_df['entry_time'])

print(f"\n데이터 로드 완료")
print(f"  - 거래: {len(trades)}개")

def calculate_force_vector(entry_time, entry_price):
    """힘의 방향 벡터 계산 - MTF 관점"""
    
    # 15분 힘의 방향 (진입 전 5캔들)
    entry_15m_idx = df_15m[df_15m['datetime'] == entry_time].index
    if len(entry_15m_idx) == 0:
        return None
    entry_15m_idx = entry_15m_idx[0]
    
    candles_15m_before = df_15m.iloc[max(0, entry_15m_idx - 5):entry_15m_idx].copy()
    candles_15m_after = df_15m.iloc[entry_15m_idx:entry_15m_idx + 10].copy()
    
    if len(candles_15m_before) < 2 or len(candles_15m_after) < 2:
        return None
    
    # 1H 힘의 방향 (진입 전 3캔들)
    entry_1h_idx = df_1h[df_1h['datetime'] <= entry_time].index
    if len(entry_1h_idx) == 0:
        return None
    entry_1h_idx = entry_1h_idx[-1]
    
    candles_1h_before = df_1h.iloc[max(0, entry_1h_idx - 3):entry_1h_idx + 1].copy()
    
    # 4H 힘의 방향 (진입 전 2캔들)
    entry_4h_idx = df_4h[df_4h['datetime'] <= entry_time].index
    if len(entry_4h_idx) == 0:
        return None
    entry_4h_idx = entry_4h_idx[-1]
    
    candles_4h_before = df_4h.iloc[max(0, entry_4h_idx - 2):entry_4h_idx + 1].copy()
    
    # === 15분 힘의 방향 ===
    # 진입 전 상승 강도
    force_15m_before = ((candles_15m_before['close'].iloc[-1] - candles_15m_before['close'].iloc[0]) / 
                        candles_15m_before['close'].iloc[0]) * 100
    
    # 진입 후 상승 강도
    force_15m_after = ((candles_15m_after['high'].max() - entry_price) / entry_price) * 100
    
    # 진입 후 즉시 하락 강도
    force_15m_down = ((candles_15m_after['low'].min() - entry_price) / entry_price) * 100
    
    # 진입 후 첫 캔들 방향
    first_candle_15m = candles_15m_after.iloc[1] if len(candles_15m_after) > 1 else candles_15m_after.iloc[0]
    force_15m_immediate = ((first_candle_15m['close'] - entry_price) / entry_price) * 100
    
    # === 1H 힘의 방향 ===
    force_1h = ((candles_1h_before['close'].iloc[-1] - candles_1h_before['close'].iloc[0]) / 
                candles_1h_before['close'].iloc[0]) * 100
    
    # 1H 상승 연속성
    up_count_1h = sum(candles_1h_before['close'].diff() > 0)
    down_count_1h = sum(candles_1h_before['close'].diff() < 0)
    
    # === 4H 힘의 방향 ===
    force_4h = ((candles_4h_before['close'].iloc[-1] - candles_4h_before['close'].iloc[0]) / 
                candles_4h_before['close'].iloc[0]) * 100
    
    # 4H 상승 연속성
    up_count_4h = sum(candles_4h_before['close'].diff() > 0)
    down_count_4h = sum(candles_4h_before['close'].diff() < 0)
    
    # === 힘의 충돌 감지 ===
    # 상위 타임프레임 vs 15분
    force_collision_1h = (force_1h < 0 and force_15m_before > 0)  # 1H 하락 중 15분 상승
    force_collision_4h = (force_4h < 0 and force_15m_before > 0)  # 4H 하락 중 15분 상승
    
    # 힘의 소진 감지
    force_exhausted_1h = (force_1h > 2.0 and up_count_1h >= 3)  # 1H 연속 상승 후
    force_exhausted_4h = (force_4h > 3.0 and up_count_4h >= 2)  # 4H 연속 상승 후
    
    # 힘의 정렬 (상승 방향 일치)
    force_aligned = (force_4h > 0 and force_1h > 0 and force_15m_before > 0)
    
    # 힘의 약화 (진입 후 즉시 역전)
    force_weakened = (force_15m_immediate < 0)
    
    # 힘의 지속성 (진입 후 계속 상승)
    force_sustained = (force_15m_after > 0.5)
    
    return {
        'force_15m_before': force_15m_before,
        'force_15m_after': force_15m_after,
        'force_15m_down': force_15m_down,
        'force_15m_immediate': force_15m_immediate,
        'force_1h': force_1h,
        'force_4h': force_4h,
        'up_count_1h': up_count_1h,
        'down_count_1h': down_count_1h,
        'up_count_4h': up_count_4h,
        'down_count_4h': down_count_4h,
        'force_collision_1h': force_collision_1h,
        'force_collision_4h': force_collision_4h,
        'force_exhausted_1h': force_exhausted_1h,
        'force_exhausted_4h': force_exhausted_4h,
        'force_aligned': force_aligned,
        'force_weakened': force_weakened,
        'force_sustained': force_sustained,
    }

print("\n" + "=" * 80)
print("모든 거래의 힘의 방향 분석 중...")
print("=" * 80)

force_results = []
for idx, trade in trades.iterrows():
    if idx % 50 == 0:
        print(f"진행: {idx}/{len(trades)}")
    
    force = calculate_force_vector(trade['entry_time'], trade['entry_price'])
    if force:
        force_results.append({
            'entry_time': trade['entry_time'],
            'exit_reason': trade['exit_reason'],
            'power_score': trade['power_score'],
            'pnl_pct': trade['pnl_pct'],
            'year': trade['year'],
            **force
        })

force_df = pd.DataFrame(force_results)

# Merge with MTF data
force_df = force_df.merge(mtf_df[['entry_time', 'trend_1h', 'trend_4h']], on='entry_time', how='left')

print(f"\n총 {len(force_df)}개 거래의 힘의 방향 분석 완료")

# === Analysis ===
print("\n" + "=" * 80)
print("1단계: 힘의 방향으로 실패 이유 분류")
print("=" * 80)

for exit_reason in ['SL', 'TP1_Breakeven', 'TP2_Full']:
    subset = force_df[force_df['exit_reason'] == exit_reason]
    if len(subset) == 0:
        continue
    
    print(f"\n{'='*60}")
    print(f"[{exit_reason}] {len(subset)}개")
    print(f"{'='*60}")
    
    print(f"\n📊 진입 전 힘의 방향:")
    print(f"  - 15분 힘: {subset['force_15m_before'].mean():.2f}%")
    print(f"  - 1H 힘: {subset['force_1h'].mean():.2f}%")
    print(f"  - 4H 힘: {subset['force_4h'].mean():.2f}%")
    
    print(f"\n📊 진입 후 힘의 반응:")
    print(f"  - 최대 상승: {subset['force_15m_after'].mean():.2f}%")
    print(f"  - 최대 하락: {subset['force_15m_down'].mean():.2f}%")
    print(f"  - 첫 캔들 반응: {subset['force_15m_immediate'].mean():.2f}%")
    
    print(f"\n⚡ 힘의 충돌/소진:")
    print(f"  - 1H 충돌 (1H↓ 15m↑): {(subset['force_collision_1h'].sum() / len(subset) * 100):.1f}%")
    print(f"  - 4H 충돌 (4H↓ 15m↑): {(subset['force_collision_4h'].sum() / len(subset) * 100):.1f}%")
    print(f"  - 1H 힘 소진: {(subset['force_exhausted_1h'].sum() / len(subset) * 100):.1f}%")
    print(f"  - 4H 힘 소진: {(subset['force_exhausted_4h'].sum() / len(subset) * 100):.1f}%")
    
    print(f"\n🎯 힘의 상태:")
    print(f"  - 힘의 정렬 (모두 ↑): {(subset['force_aligned'].sum() / len(subset) * 100):.1f}%")
    print(f"  - 진입 후 힘 약화: {(subset['force_weakened'].sum() / len(subset) * 100):.1f}%")
    print(f"  - 진입 후 힘 지속: {(subset['force_sustained'].sum() / len(subset) * 100):.1f}%")

print("\n" + "=" * 80)
print("2단계: SL 실패 - 힘의 방향 상세 분석")
print("=" * 80)

sl_cases = force_df[force_df['exit_reason'] == 'SL'].copy()

# 실패 유형 분류 (힘의 논리로)
def classify_failure_by_force(row):
    """힘의 방향 논리로 실패 유형 분류"""
    
    # Type 1: 상위 타임프레임 힘 소진
    if row['force_exhausted_1h'] or row['force_exhausted_4h']:
        return "상위TF 힘 소진 (과매수)"
    
    # Type 2: 힘의 충돌 (역추세)
    if row['force_collision_1h'] or row['force_collision_4h']:
        return "힘의 충돌 (역추세)"
    
    # Type 3: 15분 힘 자체가 약함
    if row['force_15m_before'] < 1.0:
        return "15분 힘 부족"
    
    # Type 4: 진입 후 즉시 힘 약화
    if row['force_weakened']:
        return "진입 후 즉시 힘 약화"
    
    # Type 5: 상위 TF 하락 중 (저항)
    if row['force_1h'] < 0 or row['force_4h'] < 0:
        return "상위TF 하락 저항"
    
    return "기타"

sl_cases['failure_type_force'] = sl_cases.apply(classify_failure_by_force, axis=1)

print(f"\nSL 실패 유형 (힘의 논리로 분류):")
print(sl_cases['failure_type_force'].value_counts())

for ftype in sl_cases['failure_type_force'].unique():
    subset = sl_cases[sl_cases['failure_type_force'] == ftype]
    print(f"\n[{ftype}] {len(subset)}개:")
    print(f"  - 15분 힘: {subset['force_15m_before'].mean():.2f}%")
    print(f"  - 1H 힘: {subset['force_1h'].mean():.2f}%")
    print(f"  - 4H 힘: {subset['force_4h'].mean():.2f}%")
    print(f"  - 진입 후 최대 상승: {subset['force_15m_after'].mean():.2f}%")
    print(f"  - 진입 후 첫 반응: {subset['force_15m_immediate'].mean():.2f}%")

print("\n" + "=" * 80)
print("3단계: TP1 Breakeven 실패 - 힘의 지속성 문제")
print("=" * 80)

tp1_cases = force_df[force_df['exit_reason'] == 'TP1_Breakeven'].copy()

# TP1 실패 유형
def classify_tp1_failure_by_force(row):
    """TP1 BE 실패를 힘의 지속성으로 분류"""
    
    # 힘은 있었지만 지속 안 됨
    if row['force_15m_after'] > 1.0:
        return "힘은 있었으나 지속 실패"
    
    # 초기 힘이 약함
    if row['force_15m_before'] < 1.5:
        return "초기 힘 부족"
    
    # 상위 TF 저항
    if row['force_1h'] < 0:
        return "1H 하락 저항"
    
    return "기타"

tp1_cases['failure_type_force'] = tp1_cases.apply(classify_tp1_failure_by_force, axis=1)

print(f"\nTP1 Breakeven 실패 유형 (힘의 논리):")
print(tp1_cases['failure_type_force'].value_counts())

for ftype in tp1_cases['failure_type_force'].unique():
    subset = tp1_cases[tp1_cases['failure_type_force'] == ftype]
    print(f"\n[{ftype}] {len(subset)}개:")
    print(f"  - 15분 힘: {subset['force_15m_before'].mean():.2f}%")
    print(f"  - 1H 힘: {subset['force_1h'].mean():.2f}%")
    print(f"  - 4H 힘: {subset['force_4h'].mean():.2f}%")
    print(f"  - 진입 후 최대 상승: {subset['force_15m_after'].mean():.2f}%")

print("\n" + "=" * 80)
print("4단계: TP2 성공 - 힘의 조건")
print("=" * 80)

tp2_cases = force_df[force_df['exit_reason'] == 'TP2_Full'].copy()

print(f"\nTP2 Full 성공 케이스 ({len(tp2_cases)}개):")
print(f"  - 15분 힘: {tp2_cases['force_15m_before'].mean():.2f}%")
print(f"  - 1H 힘: {tp2_cases['force_1h'].mean():.2f}%")
print(f"  - 4H 힘: {tp2_cases['force_4h'].mean():.2f}%")
print(f"  - 진입 후 최대 상승: {tp2_cases['force_15m_after'].mean():.2f}%")
print(f"  - 진입 후 첫 반응: {tp2_cases['force_15m_immediate'].mean():.2f}%")
print(f"  - 힘의 정렬: {(tp2_cases['force_aligned'].sum() / len(tp2_cases) * 100):.1f}%")
print(f"  - 힘의 지속: {(tp2_cases['force_sustained'].sum() / len(tp2_cases) * 100):.1f}%")

print("\n" + "=" * 80)
print("5단계: 성공 vs 실패 힘의 차이")
print("=" * 80)

comparison = pd.DataFrame({
    '지표': [
        '15분 진입 전 힘',
        '1H 힘',
        '4H 힘',
        '진입 후 최대 상승',
        '진입 후 첫 반응',
        '힘의 정렬 비율',
        '힘 소진 비율 (1H)',
        '힘 약화 비율',
    ],
    'TP2 성공': [
        f"{tp2_cases['force_15m_before'].mean():.2f}%",
        f"{tp2_cases['force_1h'].mean():.2f}%",
        f"{tp2_cases['force_4h'].mean():.2f}%",
        f"{tp2_cases['force_15m_after'].mean():.2f}%",
        f"{tp2_cases['force_15m_immediate'].mean():.2f}%",
        f"{(tp2_cases['force_aligned'].sum() / len(tp2_cases) * 100):.1f}%",
        f"{(tp2_cases['force_exhausted_1h'].sum() / len(tp2_cases) * 100):.1f}%",
        f"{(tp2_cases['force_weakened'].sum() / len(tp2_cases) * 100):.1f}%",
    ],
    'SL 실패': [
        f"{sl_cases['force_15m_before'].mean():.2f}%",
        f"{sl_cases['force_1h'].mean():.2f}%",
        f"{sl_cases['force_4h'].mean():.2f}%",
        f"{sl_cases['force_15m_after'].mean():.2f}%",
        f"{sl_cases['force_15m_immediate'].mean():.2f}%",
        f"{(sl_cases['force_aligned'].sum() / len(sl_cases) * 100):.1f}%",
        f"{(sl_cases['force_exhausted_1h'].sum() / len(sl_cases) * 100):.1f}%",
        f"{(sl_cases['force_weakened'].sum() / len(sl_cases) * 100):.1f}%",
    ],
})

print("\n")
print(comparison.to_string(index=False))

# Save results
force_df.to_csv('force_direction_analysis.csv', index=False)

print("\n" + "=" * 80)
print("6단계: 힘의 방향 필터 제안")
print("=" * 80)

print(f"""
🎯 성공 조건 (힘의 방향 논리):

1. 진입 전 15분 힘: 최소 1.5% 이상
2. 1H 힘: 0% 이상 (하락 중 아님)
3. 4H 힘: 0% 이상 (하락 중 아님)
4. 힘 소진 상태 아님: 1H 연속 3개 이상 상승 + 2% 이상 ❌
5. 진입 후 첫 캔들 반응: 양수 (즉시 상승)

현재 TP2 성공 케이스 특징:
- 15분 힘: {tp2_cases['force_15m_before'].mean():.2f}%
- 1H 힘: {tp2_cases['force_1h'].mean():.2f}%
- 4H 힘: {tp2_cases['force_4h'].mean():.2f}%
- 첫 반응: {tp2_cases['force_15m_immediate'].mean():.2f}%

SL 실패 케이스 특징:
- 15분 힘: {sl_cases['force_15m_before'].mean():.2f}%
- 1H 힘: {sl_cases['force_1h'].mean():.2f}%
- 4H 힘: {sl_cases['force_4h'].mean():.2f}%
- 첫 반응: {sl_cases['force_15m_immediate'].mean():.2f}%
""")

print("\n" + "=" * 80)
print("✅ 힘의 방향 분석 완료")
print("=" * 80)
print(f"결과 저장: force_direction_analysis.csv")
