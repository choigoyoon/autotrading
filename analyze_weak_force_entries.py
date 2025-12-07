import pandas as pd
import numpy as np

print("=" * 80)
print("🔍 '초기 힘 부족' 케이스 상세 분석 - 어디에 진입했는가?")
print("=" * 80)

# Load force analysis results
force_df = pd.read_csv('force_direction_analysis.csv')
force_df['entry_time'] = pd.to_datetime(force_df['entry_time'])

# Load HLHLHL data
hlhlhl_df = pd.read_csv('hlhlhl_full_labeling_analysis.csv')
hlhlhl_df['trendline_break_time'] = pd.to_datetime(hlhlhl_df['trendline_break_time'])

# Load backtest results
trades_df = pd.read_csv('backtest_confirmation_space_results.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

# Filter TP1 Breakeven cases with weak initial force
tp1_weak = force_df[
    (force_df['exit_reason'] == 'TP1_Breakeven') & 
    (force_df['force_15m_before'] < 1.5)
].copy()

sl_weak = force_df[
    (force_df['exit_reason'] == 'SL') & 
    (force_df['force_15m_before'] < 1.0)
].copy()

print(f"\nTP1 Breakeven + 초기 힘 부족 (<1.5%): {len(tp1_weak)}개")
print(f"SL 실패 + 15분 힘 부족 (<1.0%): {len(sl_weak)}개")

# Merge with trades to get H1/H2/H3 info
tp1_weak = tp1_weak.merge(
    trades_df[['entry_time', 'entry_price', 'h1_price', 'h2_price', 'h3_price']], 
    on='entry_time', 
    how='left'
)

sl_weak = sl_weak.merge(
    trades_df[['entry_time', 'entry_price', 'h1_price', 'h2_price', 'h3_price']], 
    on='entry_time', 
    how='left'
)

print("\n" + "=" * 80)
print("1단계: 진입 가격 위치 분석 (H3 대비)")
print("=" * 80)

def analyze_entry_position(df, label):
    """진입 가격이 H3 대비 어디인지 분석"""
    
    # H3 대비 진입 가격 위치
    df['entry_vs_h3_pct'] = ((df['entry_price'] - df['h3_price']) / df['h3_price']) * 100
    
    # H2 대비 진입 가격 위치
    df['entry_vs_h2_pct'] = ((df['entry_price'] - df['h2_price']) / df['h2_price']) * 100
    
    # H1 대비 진입 가격 위치
    df['entry_vs_h1_pct'] = ((df['entry_price'] - df['h1_price']) / df['h1_price']) * 100
    
    print(f"\n[{label}] {len(df)}개")
    print(f"  - H3 대비 진입 위치: {df['entry_vs_h3_pct'].mean():.2f}% (평균)")
    print(f"  - H2 대비 진입 위치: {df['entry_vs_h2_pct'].mean():.2f}% (평균)")
    print(f"  - H1 대비 진입 위치: {df['entry_vs_h1_pct'].mean():.2f}% (평균)")
    
    # H3 위/아래 분류
    h3_above = len(df[df['entry_price'] > df['h3_price']])
    h3_below = len(df[df['entry_price'] <= df['h3_price']])
    
    print(f"\n  H3 기준:")
    print(f"    - H3 위에서 진입: {h3_above}개 ({h3_above/len(df)*100:.1f}%)")
    print(f"    - H3 아래 진입: {h3_below}개 ({h3_below/len(df)*100:.1f}%)")
    
    # H2 도달 여부
    h2_reached = len(df[df['entry_price'] >= df['h2_price']])
    h2_not_reached = len(df[df['entry_price'] < df['h2_price']])
    
    print(f"\n  H2 기준:")
    print(f"    - H2 이미 도달 후 진입: {h2_reached}개 ({h2_reached/len(df)*100:.1f}%)")
    print(f"    - H2 미도달 상태 진입: {h2_not_reached}개 ({h2_not_reached/len(df)*100:.1f}%)")
    
    return df

tp1_weak = analyze_entry_position(tp1_weak, "TP1 BE + 초기 힘 부족")
sl_weak = analyze_entry_position(sl_weak, "SL + 15분 힘 부족")

print("\n" + "=" * 80)
print("2단계: 확정 공간 룰 통과 시점 vs 실제 진입 타이밍")
print("=" * 80)

# Load 15m data
df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m.sort_values('datetime').reset_index(drop=True)

def analyze_entry_timing(row):
    """진입이 H3 돌파 후 몇 캔들 후인지 분석"""
    
    entry_idx = df_15m[df_15m['datetime'] == row['entry_time']].index
    if len(entry_idx) == 0:
        return None
    
    entry_idx = entry_idx[0]
    
    # H3 가격
    h3_price = row['h3_price']
    
    # 역순으로 H3 돌파 시점 찾기
    h3_break_candle = None
    for i in range(entry_idx - 1, max(0, entry_idx - 30), -1):
        candle = df_15m.iloc[i]
        prev_candle = df_15m.iloc[i-1] if i > 0 else None
        
        if prev_candle is not None:
            # 이전 캔들이 H3 아래, 현재 캔들이 H3 위
            if prev_candle['high'] < h3_price and candle['high'] >= h3_price:
                h3_break_candle = i
                break
    
    if h3_break_candle is None:
        return None
    
    candles_after_break = entry_idx - h3_break_candle
    
    # H3 돌파 캔들의 상승 강도
    break_candle = df_15m.iloc[h3_break_candle]
    break_strength = ((break_candle['close'] - break_candle['open']) / break_candle['open']) * 100
    
    return {
        'candles_after_h3_break': candles_after_break,
        'h3_break_strength': break_strength,
    }

print("\nTP1 BE + 초기 힘 부족 케이스:")
timing_results = []
for idx, row in tp1_weak.head(10).iterrows():
    timing = analyze_entry_timing(row)
    if timing:
        timing_results.append(timing)
        print(f"  - H3 돌파 후 {timing['candles_after_h3_break']}캔들 후 진입 (돌파 강도: {timing['h3_break_strength']:.2f}%)")

if timing_results:
    avg_candles = np.mean([t['candles_after_h3_break'] for t in timing_results])
    avg_strength = np.mean([t['h3_break_strength'] for t in timing_results])
    print(f"\n평균: H3 돌파 후 {avg_candles:.1f}캔들 후 진입 (평균 돌파 강도: {avg_strength:.2f}%)")

print("\n" + "=" * 80)
print("3단계: 초기 힘 부족의 진짜 이유")
print("=" * 80)

print(f"""
🔍 분석 결과:

TP1 BE + 초기 힘 부족 ({len(tp1_weak)}개):
  - H3 대비 진입 위치: {tp1_weak['entry_vs_h3_pct'].mean():.2f}%
  - 15분 힘: {tp1_weak['force_15m_before'].mean():.2f}%
  - 진입 후 최대 상승: {tp1_weak['force_15m_after'].mean():.2f}%
  
SL + 15분 힘 부족 ({len(sl_weak)}개):
  - H3 대비 진입 위치: {sl_weak['entry_vs_h3_pct'].mean():.2f}%
  - 15분 힘: {sl_weak['force_15m_before'].mean():.2f}%
  - 진입 후 최대 상승: {sl_weak['force_15m_after'].mean():.2f}%

💡 핵심 통찰:
1. "초기 힘 부족"의 의미:
   - 진입 전 5캔들의 상승률이 약함 (< 1.5%)
   - H3 돌파는 했지만, 돌파 후 상승 모멘텀이 약해짐
   - 또는 H3 돌파 후 너무 늦게 진입

2. 진입 위치:
   - H3를 돌파했지만, 돌파 후 충분한 리테스트 없이 진입
   - 또는 H3 돌파 후 이미 많이 올라간 상태에서 진입

3. 실패 원인:
   - 확정 공간 룰은 통과했지만
   - 실제 상승 모멘텀은 이미 약화됨
   - "늦은 진입" 또는 "약한 돌파"
""")

print("\n" + "=" * 80)
print("4단계: 성공 케이스와 비교")
print("=" * 80)

tp2_success = force_df[force_df['exit_reason'] == 'TP2_Full'].copy()
tp2_success = tp2_success.merge(
    trades_df[['entry_time', 'entry_price', 'h1_price', 'h2_price', 'h3_price']], 
    on='entry_time', 
    how='left'
)
tp2_success = analyze_entry_position(tp2_success, "TP2 성공")

print("\n" + "=" * 80)
print("✅ 분석 완료")
print("=" * 80)
