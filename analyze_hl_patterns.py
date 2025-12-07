"""
HL 패턴별 성과 분석
- H(고점) 방향: 상승/하락
- L(저점) 방향: 상승/하락
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
signals = pd.read_csv('valid_signals.csv')
h_values = pd.read_csv('all_H_values.csv')
l_values = pd.read_csv('all_L_values.csv')
backtest = pd.read_csv('L_value_backtest.csv')

# 시간 변환
signals['breakout_time'] = pd.to_datetime(signals['breakout_time'])
h_values['datetime'] = pd.to_datetime(h_values['datetime'])
l_values['datetime'] = pd.to_datetime(l_values['datetime'])

print("="*80)
print("HL 패턴별 분석")
print("="*80)

# 각 시그널에 H1, H2, L1, L2 찾기
def find_recent_values(breakout_time, values_df, col_name, n=2):
    """돌파 전 최근 n개 값 찾기"""
    before = values_df[values_df['datetime'] < breakout_time].tail(n)
    if len(before) < n:
        return [None] * n
    return before[col_name].tolist()

results = []

for idx, signal in signals.iterrows():
    bt = signal['breakout_time']
    
    # 최근 H 2개
    h_list = find_recent_values(bt, h_values, 'H_value', 2)
    # 최근 L 2개  
    l_list = find_recent_values(bt, l_values, 'L_value', 2)
    
    if None in h_list or None in l_list:
        continue
    
    h1, h2 = h_list  # h1이 먼저, h2가 나중
    l1, l2 = l_list  # l1이 먼저, l2가 나중
    
    # 방향 판단
    h_direction = 'H↑' if h2 > h1 else 'H↓'
    l_direction = 'L↑' if l2 > l1 else 'L↓'
    
    pattern = f"{h_direction} {l_direction}"
    
    # 변화율
    h_change = (h2 - h1) / h1 * 100
    l_change = (l2 - l1) / l1 * 100
    
    results.append({
        'idx': idx,
        'breakout_time': bt,
        'entry_price': signal['breakout_price'],
        'h1': h1, 'h2': h2,
        'l1': l1, 'l2': l2,
        'h_direction': h_direction,
        'l_direction': l_direction,
        'pattern': pattern,
        'h_change': h_change,
        'l_change': l_change
    })

pattern_df = pd.DataFrame(results)

# 백테스트 결과 매칭
backtest['breakout_time'] = pd.to_datetime(backtest['entry_time'])
merged = pattern_df.merge(backtest[['breakout_time', 'total_pnl', 'sl_done', 'tp2_done']], 
                          on='breakout_time', how='left')

print(f"\n분석 대상: {len(merged)}건")

# 패턴별 분포
print("\n" + "="*80)
print("📊 패턴별 분포")
print("="*80)

pattern_dist = merged['pattern'].value_counts()
for pat, cnt in pattern_dist.items():
    pct = cnt / len(merged) * 100
    print(f"  {pat}: {cnt}건 ({pct:.1f}%)")

# 패턴별 성과
print("\n" + "="*80)
print("📊 패턴별 성과")
print("="*80)

pattern_stats = merged.groupby('pattern').agg({
    'total_pnl': ['count', 'mean', 'sum'],
    'sl_done': 'mean',
    'tp2_done': 'mean'
}).round(2)
pattern_stats.columns = ['건수', '평균수익', '총수익', '손절률', 'TP달성률']
pattern_stats['승률'] = merged.groupby('pattern')['total_pnl'].apply(lambda x: (x > 0).mean() * 100).round(1)
pattern_stats = pattern_stats.sort_values('평균수익', ascending=False)

print(pattern_stats)

# 상세 분석
print("\n" + "="*80)
print("📊 패턴별 상세")
print("="*80)

for pattern in ['H↑ L↑', 'H↓ L↑', 'H↑ L↓', 'H↓ L↓']:
    subset = merged[merged['pattern'] == pattern]
    if len(subset) == 0:
        continue
    
    print(f"\n【{pattern}】 {len(subset)}건")
    print(f"  평균 수익: {subset['total_pnl'].mean():+.2f}%")
    print(f"  승률: {(subset['total_pnl'] > 0).mean() * 100:.1f}%")
    print(f"  손절률: {subset['sl_done'].mean() * 100:.1f}%")
    print(f"  TP달성: {subset['tp2_done'].mean() * 100:.1f}%")
    print(f"  H 변화 평균: {subset['h_change'].mean():+.2f}%")
    print(f"  L 변화 평균: {subset['l_change'].mean():+.2f}%")

# 결론
print("\n" + "="*80)
print("💡 결론")
print("="*80)

best = pattern_stats.iloc[0]
worst = pattern_stats.iloc[-1]

print(f"\n✅ 최고 패턴: {pattern_stats.index[0]}")
print(f"   평균수익 {best['평균수익']:+.2f}%, 승률 {best['승률']}%")

print(f"\n❌ 최악 패턴: {pattern_stats.index[-1]}")
print(f"   평균수익 {worst['평균수익']:+.2f}%, 승률 {worst['승률']}%")

# 저장
merged.to_csv('hl_pattern_analysis.csv', index=False)
print(f"\n저장: hl_pattern_analysis.csv")

