import pandas as pd
import numpy as np

print("=" * 80)
print("💰 수익 나는 상황 완전 분석 - TP2 성공 케이스")
print("=" * 80)

# Load all data
force_df = pd.read_csv('force_direction_analysis.csv')
force_df['entry_time'] = pd.to_datetime(force_df['entry_time'])

trades_df = pd.read_csv('backtest_confirmation_space_results.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

mtf_df = pd.read_csv('mtf_context_analysis.csv')
mtf_df['entry_time'] = pd.to_datetime(mtf_df['entry_time'])

df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m.sort_values('datetime').reset_index(drop=True)

# TP2 Success cases
tp2_success = force_df[force_df['exit_reason'] == 'TP2_Full'].copy()
tp2_success = tp2_success.merge(
    trades_df[['entry_time', 'entry_price', 'h1_price', 'h2_price', 'h3_price', 'power_score']], 
    on='entry_time', 
    how='left'
)
tp2_success = tp2_success.merge(
    mtf_df[['entry_time', 'trend_1h', 'trend_4h', 'momentum_1h', 'momentum_4h']], 
    on='entry_time', 
    how='left'
)

print(f"\n✅ TP2 성공 케이스: {len(tp2_success)}개")

# Calculate entry position
tp2_success['entry_vs_h3_pct'] = ((tp2_success['entry_price'] - tp2_success['h3_price']) / tp2_success['h3_price']) * 100
tp2_success['entry_vs_h2_pct'] = ((tp2_success['entry_price'] - tp2_success['h2_price']) / tp2_success['h2_price']) * 100
tp2_success['entry_vs_h1_pct'] = ((tp2_success['entry_price'] - tp2_success['h1_price']) / tp2_success['h1_price']) * 100

print("\n" + "=" * 80)
print("1단계: 수익 나는 진입 위치")
print("=" * 80)

print(f"\n진입 가격 위치:")
print(f"  - H3 대비: +{tp2_success['entry_vs_h3_pct'].mean():.2f}%")
print(f"  - H2 대비: {tp2_success['entry_vs_h2_pct'].mean():.2f}%")
print(f"  - H1 대비: {tp2_success['entry_vs_h1_pct'].mean():.2f}%")

# H3 위 분포
h3_ranges = [
    (0.0, 0.5, "H3 바로 위 (0~0.5%)"),
    (0.5, 1.0, "H3 위 (0.5~1.0%)"),
    (1.0, 1.5, "H3 위 (1.0~1.5%)"),
    (1.5, 999, "H3 위 (1.5%+)")
]

print(f"\nH3 대비 진입 위치 분포:")
for min_pct, max_pct, label in h3_ranges:
    count = len(tp2_success[(tp2_success['entry_vs_h3_pct'] >= min_pct) & 
                            (tp2_success['entry_vs_h3_pct'] < max_pct)])
    pct = count / len(tp2_success) * 100
    print(f"  - {label}: {count}개 ({pct:.1f}%)")

# H2 도달 여부
h2_reached = len(tp2_success[tp2_success['entry_price'] >= tp2_success['h2_price']])
print(f"\nH2 이미 도달 후 진입: {h2_reached}개 ({h2_reached/len(tp2_success)*100:.1f}%)")

print("\n" + "=" * 80)
print("2단계: 수익 나는 힘의 방향")
print("=" * 80)

print(f"\n📊 진입 전 힘:")
print(f"  - 15분 힘: {tp2_success['force_15m_before'].mean():.2f}%")
print(f"  - 1H 힘: {tp2_success['force_1h'].mean():.2f}%")
print(f"  - 4H 힘: {tp2_success['force_4h'].mean():.2f}%")

print(f"\n📊 진입 후 반응:")
print(f"  - 최대 상승: {tp2_success['force_15m_after'].mean():.2f}%")
print(f"  - 첫 캔들: {tp2_success['force_15m_immediate'].mean():.2f}%")

print(f"\n⚡ 힘의 상태:")
print(f"  - 힘의 정렬: {(tp2_success['force_aligned'].sum() / len(tp2_success) * 100):.1f}%")
print(f"  - 힘 소진 (1H): {(tp2_success['force_exhausted_1h'].sum() / len(tp2_success) * 100):.1f}%")
print(f"  - 힘 소진 (4H): {(tp2_success['force_exhausted_4h'].sum() / len(tp2_success) * 100):.1f}%")

# 힘의 범위별 분류
force_ranges = [
    (0.0, 1.0, "15분 힘 약함 (<1.0%)"),
    (1.0, 1.5, "15분 힘 보통 (1.0~1.5%)"),
    (1.5, 2.0, "15분 힘 강함 (1.5~2.0%)"),
    (2.0, 999, "15분 힘 매우 강함 (2.0%+)")
]

print(f"\n15분 힘 분포:")
for min_f, max_f, label in force_ranges:
    count = len(tp2_success[(tp2_success['force_15m_before'] >= min_f) & 
                            (tp2_success['force_15m_before'] < max_f)])
    pct = count / len(tp2_success) * 100
    avg_pnl = tp2_success[(tp2_success['force_15m_before'] >= min_f) & 
                          (tp2_success['force_15m_before'] < max_f)]['pnl_pct'].mean()
    print(f"  - {label}: {count}개 ({pct:.1f}%), 평균 PnL: {avg_pnl:.2f}%")

print("\n" + "=" * 80)
print("3단계: 수익 나는 MTF 상태")
print("=" * 80)

print(f"\nMTF 추세:")
print(f"  - 1H 상승추세: {(tp2_success['trend_1h'] == '상승추세').sum()}개 ({(tp2_success['trend_1h'] == '상승추세').sum()/len(tp2_success)*100:.1f}%)")
print(f"  - 1H 횡보/전환: {(tp2_success['trend_1h'] == '횡보/전환').sum()}개 ({(tp2_success['trend_1h'] == '횡보/전환').sum()/len(tp2_success)*100:.1f}%)")
print(f"  - 4H 상승추세: {(tp2_success['trend_4h'] == '상승추세').sum()}개 ({(tp2_success['trend_4h'] == '상승추세').sum()/len(tp2_success)*100:.1f}%)")
print(f"  - 4H 횡보/전환: {(tp2_success['trend_4h'] == '횡보/전환').sum()}개 ({(tp2_success['trend_4h'] == '횡보/전환').sum()/len(tp2_success)*100:.1f}%)")

print(f"\nMTF 모멘텀:")
print(f"  - 1H 모멘텀: {tp2_success['momentum_1h'].mean():.2f}%")
print(f"  - 4H 모멘텀: {tp2_success['momentum_4h'].mean():.2f}%")

print("\n" + "=" * 80)
print("4단계: 수익 나는 Power Score")
print("=" * 80)

print(f"\nPower Score 분포:")
for score in sorted(tp2_success['power_score'].unique()):
    subset = tp2_success[tp2_success['power_score'] == score]
    avg_pnl = subset['pnl_pct'].mean()
    print(f"  - Score {int(score)}: {len(subset)}개 ({len(subset)/len(tp2_success)*100:.1f}%), 평균 PnL: {avg_pnl:.2f}%")

print("\n" + "=" * 80)
print("5단계: 수익 나는 조건 조합 - TOP 패턴")
print("=" * 80)

# 최고 수익 조건 찾기
def find_best_conditions(df, min_trades=10):
    """최고 수익 조건 조합 찾기"""
    
    conditions = []
    
    # Condition 1: 4H 힘 강함
    c1 = df[df['force_4h'] > 2.0].copy()
    if len(c1) >= min_trades:
        conditions.append({
            'name': '4H 힘 > 2.0%',
            'trades': len(c1),
            'avg_pnl': c1['pnl_pct'].mean(),
            'total_pnl': c1['pnl_pct'].sum(),
            'tp2_rate': len(c1) / len(c1) * 100
        })
    
    # Condition 2: 15분 힘 강함 + 4H 힘 강함
    c2 = df[(df['force_15m_before'] > 1.5) & (df['force_4h'] > 2.0)].copy()
    if len(c2) >= min_trades:
        conditions.append({
            'name': '15m > 1.5% + 4H > 2.0%',
            'trades': len(c2),
            'avg_pnl': c2['pnl_pct'].mean(),
            'total_pnl': c2['pnl_pct'].sum(),
            'tp2_rate': len(c2) / len(c2) * 100
        })
    
    # Condition 3: H2 이미 도달 + 4H 힘 강함
    c3 = df[(df['entry_price'] >= df['h2_price']) & (df['force_4h'] > 2.0)].copy()
    if len(c3) >= min_trades:
        conditions.append({
            'name': 'H2 도달 + 4H > 2.0%',
            'trades': len(c3),
            'avg_pnl': c3['pnl_pct'].mean(),
            'total_pnl': c3['pnl_pct'].sum(),
            'tp2_rate': len(c3) / len(c3) * 100
        })
    
    # Condition 4: 1H 횡보 + 4H 횡보 + 4H 힘 > 1.5%
    c4 = df[(df['trend_1h'] == '횡보/전환') & 
            (df['trend_4h'] == '횡보/전환') & 
            (df['force_4h'] > 1.5)].copy()
    if len(c4) >= min_trades:
        conditions.append({
            'name': '1H 횡보 + 4H 횡보 + 4H힘 > 1.5%',
            'trades': len(c4),
            'avg_pnl': c4['pnl_pct'].mean(),
            'total_pnl': c4['pnl_pct'].sum(),
            'tp2_rate': len(c4) / len(c4) * 100
        })
    
    # Condition 5: 진입 후 첫 캔들 양수
    c5 = df[df['force_15m_immediate'] > 0].copy()
    if len(c5) >= min_trades:
        conditions.append({
            'name': '첫 캔들 반응 > 0',
            'trades': len(c5),
            'avg_pnl': c5['pnl_pct'].mean(),
            'total_pnl': c5['pnl_pct'].sum(),
            'tp2_rate': len(c5) / len(c5) * 100
        })
    
    # Condition 6: 힘 소진 아님
    c6 = df[~df['force_exhausted_1h']].copy()
    if len(c6) >= min_trades:
        conditions.append({
            'name': '1H 힘 소진 아님',
            'trades': len(c6),
            'avg_pnl': c6['pnl_pct'].mean(),
            'total_pnl': c6['pnl_pct'].sum(),
            'tp2_rate': len(c6) / len(c6) * 100
        })
    
    return sorted(conditions, key=lambda x: x['avg_pnl'], reverse=True)

best_conditions = find_best_conditions(tp2_success)

print("\n🏆 TOP 수익 조건 (TP2 성공 케이스 내에서):")
for i, cond in enumerate(best_conditions[:5], 1):
    print(f"\n{i}. {cond['name']}")
    print(f"   - 거래수: {cond['trades']}개")
    print(f"   - 평균 PnL: {cond['avg_pnl']:.2f}%")
    print(f"   - 총 PnL: {cond['total_pnl']:.2f}%")

print("\n" + "=" * 80)
print("6단계: 전체 거래에서 수익 조건 적용 시뮬레이션")
print("=" * 80)

# Apply to all trades
all_trades = force_df.merge(
    trades_df[['entry_time', 'entry_price', 'h1_price', 'h2_price', 'h3_price', 'power_score']], 
    on='entry_time', 
    how='left'
)
all_trades = all_trades.merge(
    mtf_df[['entry_time', 'trend_1h', 'trend_4h']], 
    on='entry_time', 
    how='left'
)

# Test condition: 4H force > 2.0%
filtered = all_trades[all_trades['force_4h'] > 2.0].copy()

print(f"\n조건: 4H 힘 > 2.0%")
print(f"  - 전체 거래: {len(all_trades)}개 → 필터 후: {len(filtered)}개 ({len(filtered)/len(all_trades)*100:.1f}%)")
print(f"  - TP2 성공률: {(filtered['exit_reason'] == 'TP2_Full').sum() / len(filtered) * 100:.1f}%")
print(f"  - SL 비율: {(filtered['exit_reason'] == 'SL').sum() / len(filtered) * 100:.1f}%")
print(f"  - 평균 PnL: {filtered['pnl_pct'].mean():.2f}%")
print(f"  - 총 PnL: {filtered['pnl_pct'].sum():.2f}%")

# Test multiple conditions
filtered2 = all_trades[(all_trades['force_4h'] > 2.0) & 
                       (all_trades['force_15m_before'] > 1.0) & 
                       (~all_trades['force_exhausted_1h'])].copy()

print(f"\n조건: 4H > 2.0% + 15m > 1.0% + 힘 소진 아님")
print(f"  - 전체 거래: {len(all_trades)}개 → 필터 후: {len(filtered2)}개 ({len(filtered2)/len(all_trades)*100:.1f}%)")
print(f"  - TP2 성공률: {(filtered2['exit_reason'] == 'TP2_Full').sum() / len(filtered2) * 100:.1f}%")
print(f"  - SL 비율: {(filtered2['exit_reason'] == 'SL').sum() / len(filtered2) * 100:.1f}%")
print(f"  - 평균 PnL: {filtered2['pnl_pct'].mean():.2f}%")
print(f"  - 총 PnL: {filtered2['pnl_pct'].sum():.2f}%")

print("\n" + "=" * 80)
print("7단계: 수익 나는 황금 조건")
print("=" * 80)

print(f"""
💰 수익 나는 상황 - 황금 조건:

1. **진입 위치:**
   - H3 + 0.94% (평균)
   - H2 이미 도달: 63.8%
   - H1 근처: -0.10%

2. **힘의 방향:**
   - 15분 힘: 1.25%+ (최소 1.0% 이상)
   - 1H 힘: 1.83%+ (양수 필수)
   - 4H 힘: 2.37%+ (가장 중요! 2.0% 이상)
   
3. **MTF 상태:**
   - 4H 횡보/전환 또는 상승추세
   - 1H 횡보/전환 또는 상승추세
   - 4H 모멘텀: 2.37%+
   
4. **진입 후 반응:**
   - 첫 캔들: +0.09% (양수)
   - 최대 상승: 1.33%+
   
5. **힘 소진 여부:**
   - 1H 힘 소진 아님: 77.3%
   - 4H 힘 소진 아님: 77.9%

🎯 핵심:
- **4H 힘 > 2.0%가 결정적!**
- 15분/1H 힘도 중요하지만 4H가 가장 중요
- 힘 소진 상태면 진입 거부
- 첫 캔들 반응이 양수여야 함

📊 성공 확률:
- 기본 전략: 45.7% TP2 성공
- 4H > 2.0% 필터: TP2 성공률 향상 예상
""")

print("\n" + "=" * 80)
print("✅ 분석 완료")
print("=" * 80)
