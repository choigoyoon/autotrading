import pandas as pd
import numpy as np

"""
거래 수와 승률/MDD 균형 잡힌 필터 찾기
"""

df = pd.read_csv('trade_loss_reasons_analysis.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'])

print("="*100)
print("🎯 거래 수 vs 승률/MDD 균형 최적화")
print("="*100)

def calc_metrics(data):
    if len(data) == 0:
        return None
    data = data.reset_index(drop=True)
    win_rate = (1 - data['is_loss'].sum() / len(data)) * 100
    total_pnl = data['pnl'].sum()
    avg_pnl = data['pnl'].mean()
    cum = data['pnl'].cumsum()
    running_max = cum.cummax()
    mdd = (cum - running_max).min()
    return {'trades': len(data), 'win_rate': win_rate, 'total_pnl': total_pnl, 'avg_pnl': avg_pnl, 'mdd': mdd}

base = calc_metrics(df)
print(f"\n현재: 거래 {base['trades']}건, 승률 {base['win_rate']:.1f}%, MDD {base['mdd']:.1f}%")

# 핵심 발견을 바탕으로 단계별 필터 적용
print("\n" + "="*100)
print("📊 단계별 필터 적용 효과")
print("="*100)

# 조건들
conditions = {
    'mom_extreme': df['momentum_4h'].isin(['4H급락', '4H강한상승']),
    'mom_not_weak': ~df['momentum_4h'].isin(['4H약보합', '4H약한하락']),
    'mom_strong': df['momentum_4h'].isin(['4H상승', '4H강한상승']),
    'rsi_strong': df['rsi'] > 55,
    'rsi_high': df['rsi'] > 60,
    'trend_clear': df['trend'] != '횡보/혼조',
    'avoid_danger': ~((df['trend'] == '상승추세(EMA정배열)') & (df['momentum_4h'] == '4H약보합')),
}

# 거래 수 구간별 최적 필터 찾기
print(f"\n{'거래 수 목표':<15} {'최적 필터':<50} {'실제거래':>8} {'승률':>7} {'MDD':>8} {'총PNL':>8}")
print("-"*110)

# 다양한 거래 수 목표
targets = [500, 400, 300, 200, 150, 100, 50]

for target in targets:
    best_filter = None
    best_score = -999
    best_result = None
    
    # 필터 조합 테스트
    test_filters = {
        f'약보합회피': conditions['mom_not_weak'],
        f'약보합회피+RSI55': conditions['mom_not_weak'] & conditions['rsi_strong'],
        f'약보합회피+RSI60': conditions['mom_not_weak'] & conditions['rsi_high'],
        f'약보합회피+횡보회피': conditions['mom_not_weak'] & conditions['trend_clear'],
        f'약보합회피+RSI55+횡보회피': conditions['mom_not_weak'] & conditions['rsi_strong'] & conditions['trend_clear'],
        f'상승모멘텀만': conditions['mom_strong'],
        f'상승모멘텀+RSI55': conditions['mom_strong'] & conditions['rsi_strong'],
        f'상승모멘텀+RSI60': conditions['mom_strong'] & conditions['rsi_high'],
        f'극단모멘텀': conditions['mom_extreme'],
        f'위험회피': conditions['avoid_danger'],
        f'위험회피+RSI55': conditions['avoid_danger'] & conditions['rsi_strong'],
        f'위험회피+약보합회피': conditions['avoid_danger'] & conditions['mom_not_weak'],
    }
    
    for name, cond in test_filters.items():
        filtered = df[cond]
        metrics = calc_metrics(filtered)
        if metrics is None:
            continue
        
        # 목표 거래 수에 가까운 것 중 가장 좋은 성과
        trade_diff = abs(metrics['trades'] - target)
        if trade_diff > target * 0.3:  # 30% 오차 허용
            continue
        
        # 점수: 승률 + MDD개선 - 거래차이패널티
        score = metrics['win_rate'] + (metrics['mdd'] - base['mdd']) - trade_diff * 0.05
        
        if score > best_score:
            best_score = score
            best_filter = name
            best_result = metrics
    
    if best_result:
        print(f"{target:>6}건 근처  {best_filter:<50} {best_result['trades']:>8} {best_result['win_rate']:>6.1f}% {best_result['mdd']:>7.1f}% {best_result['total_pnl']:>7.1f}%")

# 상세 비교
print("\n" + "="*100)
print("📈 주요 필터 상세 비교")
print("="*100)

key_filters = {
    '기본 (필터없음)': df,
    '① 위험회피 (상승+약보합)': df[conditions['avoid_danger']],
    '② 약보합/약하락 회피': df[conditions['mom_not_weak']],
    '③ ②+횡보회피': df[conditions['mom_not_weak'] & conditions['trend_clear']],
    '④ ②+RSI55': df[conditions['mom_not_weak'] & conditions['rsi_strong']],
    '⑤ ②+RSI55+횡보회피': df[conditions['mom_not_weak'] & conditions['rsi_strong'] & conditions['trend_clear']],
    '⑥ 상승모멘텀만': df[conditions['mom_strong']],
    '⑦ 상승모멘텀+RSI55': df[conditions['mom_strong'] & conditions['rsi_strong']],
    '⑧ 극단모멘텀(급락/강상승)': df[conditions['mom_extreme']],
}

print(f"\n{'필터':<30} {'거래':>6} {'차단%':>7} {'승률':>7} {'MDD':>8} {'총PNL':>9} {'평균PNL':>8} {'연평균':>8}")
print("-"*100)

for name, filtered in key_filters.items():
    m = calc_metrics(filtered)
    if m is None:
        continue
    block = (base['trades'] - m['trades']) / base['trades'] * 100
    years = 5.5
    annual = m['total_pnl'] / years
    
    # 플래그
    flags = []
    if m['win_rate'] >= 75: flags.append("⭐")
    elif m['win_rate'] >= 70: flags.append("✓")
    if m['mdd'] >= -10: flags.append("💎")
    elif m['mdd'] >= -15: flags.append("✓")
    
    flag_str = "".join(flags)
    print(f"{name:<30} {m['trades']:>6} {block:>6.1f}% {m['win_rate']:>6.1f}% {m['mdd']:>7.1f}% {m['total_pnl']:>8.1f}% {m['avg_pnl']:>7.2f}% {annual:>7.1f}% {flag_str}")

# 최적 조합 추천
print("\n" + "="*100)
print("🏆 최종 추천")
print("="*100)

recommendations = [
    ('거래 많이 유지', conditions['avoid_danger'], '위험회피만'),
    ('균형잡힌', conditions['mom_not_weak'], '약보합/약하락 회피'),
    ('승률 중시', conditions['mom_not_weak'] & conditions['rsi_strong'] & conditions['trend_clear'], '②+RSI55+횡보회피'),
    ('고승률 (MDD최소)', conditions['mom_extreme'], '극단모멘텀'),
]

for style, cond, name in recommendations:
    m = calc_metrics(df[cond])
    if m:
        print(f"\n💡 {style}: {name}")
        print(f"   거래: {base['trades']} → {m['trades']}건 (-{(base['trades']-m['trades'])/base['trades']*100:.0f}%)")
        print(f"   승률: {base['win_rate']:.1f}% → {m['win_rate']:.1f}% ({m['win_rate']-base['win_rate']:+.1f}%p)")
        print(f"   MDD: {base['mdd']:.1f}% → {m['mdd']:.1f}% ({m['mdd']-base['mdd']:+.1f}%p)")
        print(f"   총PNL: {base['total_pnl']:.1f}% → {m['total_pnl']:.1f}%")
        print(f"   평균PNL: {base['avg_pnl']:.2f}% → {m['avg_pnl']:.2f}%")

# 트레이드오프 요약
print("\n" + "="*100)
print("📋 트레이드오프 요약")
print("="*100)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  선택지           │ 거래수  │ 승률    │ MDD     │ 총PNL    │ 특징              │
├─────────────────────────────────────────────────────────────────────────────┤
│  기본 (필터없음)    │ 690건  │ 63.3%  │ -19.5% │ 169.4% │ 고PNL, 고MDD       │
│  ② 약보합회피      │ 520건  │ 67.7%  │ -13.9% │ 164.0% │ 균형 (추천)        │
│  ⑤ ②+RSI+횡보회피 │ 312건  │ 70.1%  │ -13.7% │ 121.6% │ 승률↑, 거래↓       │
│  ⑧ 극단모멘텀      │ 97건   │ 81.4%  │ -5.9%  │ 99.9%  │ 최고승률+MDD       │
└─────────────────────────────────────────────────────────────────────────────┘

💡 목표에 따른 선택:
   - 거래 수 중시 → ② 약보합/약하락 회피 (520건, 승률 67.7%, MDD -13.9%)
   - 승률/MDD 중시 → ⑧ 극단모멘텀 (97건, 승률 81.4%, MDD -5.9%)
   - 균형 → ⑤ 약보합회피+RSI55+횡보회피 (312건, 승률 70.1%, MDD -13.7%)
""")
