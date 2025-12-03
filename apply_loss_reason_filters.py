import pandas as pd
import numpy as np

"""
하락 사유 기반 필터 적용 후 매매횟수/성과 비교
"""

# 분석 결과 로드
df = pd.read_csv('trade_loss_reasons_analysis.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'])

print("="*100)
print("📊 하락 사유 기반 필터 - 매매횟수 비교")
print("="*100)

# MDD 계산 함수
def calc_metrics(data):
    if len(data) == 0:
        return {'trades': 0, 'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0, 'mdd': 0, 'sl_rate': 0}
    
    data = data.reset_index(drop=True)
    win_rate = (1 - data['is_loss'].sum() / len(data)) * 100
    sl_rate = data['is_sl'].sum() / len(data) * 100
    total_pnl = data['pnl'].sum()
    avg_pnl = data['pnl'].mean()
    
    # MDD
    cum = data['pnl'].cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    mdd = dd.min()
    
    return {
        'trades': len(data),
        'win_rate': win_rate,
        'sl_rate': sl_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'mdd': mdd
    }

# 기본 성과
base = calc_metrics(df)

print(f"\n📈 기본 전략 (필터 없음)")
print(f"   거래: {base['trades']}건")
print(f"   승률: {base['win_rate']:.1f}%, SL비율: {base['sl_rate']:.1f}%")
print(f"   총PNL: {base['total_pnl']:.1f}%, MDD: {base['mdd']:.1f}%")

# 발견된 핵심 패턴 기반 필터
print("\n" + "="*100)
print("📊 하락 사유 기반 필터별 비교")
print("="*100)

filters = {
    # 가장 위험한 패턴 회피
    '① 상승추세+약보합 회피': lambda d: d[~((d['trend'] == '상승추세(EMA정배열)') & (d['momentum_4h'] == '4H약보합'))],
    
    # 4H 모멘텀 기반
    '② 4H약보합(-1~0%) 회피': lambda d: d[d['momentum_4h'] != '4H약보합'],
    '③ 4H약한하락(-2~-1%) 회피': lambda d: d[d['momentum_4h'] != '4H약한하락'],
    '④ 4H약보합+약한하락 모두 회피': lambda d: d[~d['momentum_4h'].isin(['4H약보합', '4H약한하락'])],
    
    # 24H 고점 기반
    '⑤ 24H고점-3%이내 회피': lambda d: d[d['position_24h'] != '24H고점-3%이내'],
    
    # RSI 기반
    '⑥ RSI중립(45-55) 회피': lambda d: d[d['rsi_state'] != 'RSI중립'],
    
    # 복합 조건
    '⑦ 횡보+4H상승 회피': lambda d: d[~((d['trend'] == '횡보/혼조') & (d['momentum_4h'] == '4H상승'))],
    '⑧ 상승추세+약보합 + 횡보+약보합 회피': lambda d: d[~(
        ((d['trend'] == '상승추세(EMA정배열)') & (d['momentum_4h'] == '4H약보합')) |
        ((d['trend'] == '횡보/혼조') & (d['momentum_4h'] == '4H약보합'))
    )],
    
    # 연속 하락 기반
    '⑨ 연속음봉(5+) 회피': lambda d: d[d['red_streak'] < 5],
    
    # 안전 조건만 진입
    '⑩ 4H상승/강상승만 진입': lambda d: d[d['momentum_4h'].isin(['4H상승', '4H강한상승'])],
    '⑪ 4H급락/강상승만 진입 (양극단)': lambda d: d[d['momentum_4h'].isin(['4H급락', '4H강한상승'])],
}

results = []
for name, filter_fn in filters.items():
    filtered = filter_fn(df.copy())
    metrics = calc_metrics(filtered)
    
    results.append({
        'filter': name,
        'trades': metrics['trades'],
        'blocked': base['trades'] - metrics['trades'],
        'block_pct': (base['trades'] - metrics['trades']) / base['trades'] * 100,
        'win_rate': metrics['win_rate'],
        'sl_rate': metrics['sl_rate'],
        'total_pnl': metrics['total_pnl'],
        'avg_pnl': metrics['avg_pnl'],
        'mdd': metrics['mdd'],
        'mdd_change': metrics['mdd'] - base['mdd'],
        'pnl_change': metrics['total_pnl'] - base['total_pnl']
    })

results_df = pd.DataFrame(results)

# 출력
print(f"\n{'필터':<40} {'거래':>5} {'차단':>5} {'차단%':>6} {'승률':>6} {'SL%':>5} {'총PNL':>7} {'MDD':>7} {'MDD변화':>8}")
print("-"*115)

for _, r in results_df.iterrows():
    mdd_flag = "✅" if r['mdd_change'] > 3 else "⭐" if r['mdd_change'] > 1 else ""
    print(f"{r['filter']:<40} {r['trades']:>5} {r['blocked']:>5} {r['block_pct']:>5.1f}% {r['win_rate']:>5.1f}% {r['sl_rate']:>4.1f}% {r['total_pnl']:>6.1f}% {r['mdd']:>6.1f}% {r['mdd_change']:>+7.1f}% {mdd_flag}")

# 효율성 분석
print("\n" + "="*100)
print("📈 효율성 분석 (차단 대비 MDD 개선)")
print("="*100)

for _, r in results_df.iterrows():
    if r['block_pct'] > 0:
        efficiency = r['mdd_change'] / r['block_pct']  # MDD 개선 / 차단율
        pnl_per_blocked = r['pnl_change'] / r['blocked'] if r['blocked'] > 0 else 0
        print(f"{r['filter']:<40}: MDD 효율 {efficiency:>+.2f}, 차단 1건당 PNL {pnl_per_blocked:>+.2f}%")

# 최적 필터 찾기
print("\n" + "="*100)
print("🏆 최적 필터 선정")
print("="*100)

# 조건: MDD 개선 + 거래수 유지 (차단 20% 미만)
good_filters = results_df[(results_df['mdd_change'] > 0) & (results_df['block_pct'] < 25)].sort_values('mdd_change', ascending=False)

if len(good_filters) > 0:
    print("\nMDD 개선 + 거래수 유지 (차단 25% 미만) 필터:")
    for _, r in good_filters.head(5).iterrows():
        print(f"\n  {r['filter']}")
        print(f"    거래: {base['trades']} → {r['trades']} (-{r['blocked']}건, -{r['block_pct']:.1f}%)")
        print(f"    승률: {base['win_rate']:.1f}% → {r['win_rate']:.1f}% ({r['win_rate']-base['win_rate']:+.1f}%p)")
        print(f"    MDD: {base['mdd']:.1f}% → {r['mdd']:.1f}% ({r['mdd_change']:+.1f}%p)")
        print(f"    총PNL: {base['total_pnl']:.1f}% → {r['total_pnl']:.1f}% ({r['pnl_change']:+.1f}%p)")

# 최종 추천
print("\n" + "="*100)
print("🎯 최종 추천")
print("="*100)

# 가장 효율적인 필터 (차단 적고 MDD 개선)
best = results_df[results_df['mdd_change'] > 0].sort_values(
    by=['mdd_change'], ascending=False
).iloc[0] if len(results_df[results_df['mdd_change'] > 0]) > 0 else None

if best is not None:
    print(f"\n💡 추천 필터: {best['filter']}")
    print(f"\n   📊 매매횟수 변화:")
    print(f"      기존: {base['trades']}건")
    print(f"      필터 후: {best['trades']}건")
    print(f"      차이: -{best['blocked']}건 (-{best['block_pct']:.1f}%)")
    print(f"\n   📈 성과 변화:")
    print(f"      승률: {base['win_rate']:.1f}% → {best['win_rate']:.1f}% ({best['win_rate']-base['win_rate']:+.1f}%p)")
    print(f"      SL비율: {base['sl_rate']:.1f}% → {best['sl_rate']:.1f}% ({best['sl_rate']-base['sl_rate']:+.1f}%p)")
    print(f"      총PNL: {base['total_pnl']:.1f}% → {best['total_pnl']:.1f}% ({best['pnl_change']:+.1f}%p)")
    print(f"      MDD: {base['mdd']:.1f}% → {best['mdd']:.1f}% ({best['mdd_change']:+.1f}%p 개선)")

# 양극단 전략 분석
print("\n" + "="*100)
print("💡 특별 발견: 양극단 전략")
print("="*100)

extreme = results_df[results_df['filter'].str.contains('양극단')]
if len(extreme) > 0:
    e = extreme.iloc[0]
    print(f"\n⭐ '4H급락 또는 강상승'에서만 진입:")
    print(f"   거래: {e['trades']}건 (기존 {base['trades']}건의 {e['trades']/base['trades']*100:.0f}%)")
    print(f"   승률: {e['win_rate']:.1f}% (기존 {base['win_rate']:.1f}%)")
    print(f"   평균PNL: {e['avg_pnl']:.2f}% (기존 {base['avg_pnl']:.2f}%)")
    print(f"   MDD: {e['mdd']:.1f}% (기존 {base['mdd']:.1f}%)")
    print(f"\n   → 거래는 {100-e['trades']/base['trades']*100:.0f}% 감소하지만, 평균 수익률 {e['avg_pnl']/base['avg_pnl']:.1f}배!")
