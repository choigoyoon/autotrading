import pandas as pd
import numpy as np

"""
승률 + MDD 동시 개선을 위한 복합 필터 최적화
목표: 승률 77%+, MDD -10% 이내
"""

# 데이터 로드
df = pd.read_csv('trade_loss_reasons_analysis.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'])

print("="*100)
print("🎯 승률 + MDD 동시 개선 최적화")
print("="*100)
print("목표: 승률 77%+, MDD -10% 이내")

# MDD 계산 함수
def calc_metrics(data):
    if len(data) == 0:
        return None
    
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
    
    # 연간 PNL (약 5.5년 데이터)
    years = (data['entry_time'].max() - data['entry_time'].min()).days / 365
    annual_pnl = total_pnl / years if years > 0 else 0
    
    return {
        'trades': len(data),
        'win_rate': win_rate,
        'sl_rate': sl_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'mdd': mdd,
        'annual_pnl': annual_pnl
    }

base = calc_metrics(df)
print(f"\n📊 현재 성과:")
print(f"   거래: {base['trades']}건, 승률: {base['win_rate']:.1f}%, MDD: {base['mdd']:.1f}%")
print(f"   총PNL: {base['total_pnl']:.1f}%, 연평균: {base['annual_pnl']:.1f}%")

# 복합 필터 조합 생성
print("\n" + "="*100)
print("🔬 복합 필터 조합 테스트")
print("="*100)

results = []

# 개별 조건들
conditions = {
    # 모멘텀 기반
    'mom_strong_up': df['momentum_4h'] == '4H강한상승',
    'mom_up': df['momentum_4h'] == '4H상승',
    'mom_crash': df['momentum_4h'] == '4H급락',
    'mom_not_weak': ~df['momentum_4h'].isin(['4H약보합', '4H약한하락']),
    'mom_extreme': df['momentum_4h'].isin(['4H급락', '4H강한상승']),
    
    # 추세 기반
    'trend_up': df['trend'] == '상승추세(EMA정배열)',
    'trend_down': df['trend'] == '하락추세(EMA역배열)',
    'trend_not_sideways': df['trend'] != '횡보/혼조',
    
    # RSI 기반
    'rsi_strong': df['rsi_state'].isin(['RSI강세', 'RSI과매수']),
    'rsi_not_neutral': df['rsi_state'] != 'RSI중립',
    'rsi_oversold': df['rsi_state'].isin(['RSI과매도', 'RSI약세']),
    'rsi_high': df['rsi'] > 60,
    'rsi_very_high': df['rsi'] > 70,
    
    # 위치 기반
    'pos_not_mid': df['position_24h'] != '24H고점-3%이내',
    'pos_low': df['position_24h'].isin(['24H고점-5%이내', '24H고점-5%이상하락']),
    'pos_near_high': df['position_24h'] == '24H고점근처',
    
    # 기타
    'low_red_streak': df['red_streak'] < 5,
    'ema20_up': df['ema20_slope'] > 0,
    'ema20_strong_up': df['ema20_slope'] > 0.2,
}

# 위험 패턴 회피 조건
avoid_conditions = {
    'avoid_up_weak': ~((df['trend'] == '상승추세(EMA정배열)') & (df['momentum_4h'] == '4H약보합')),
    'avoid_sideways_weak': ~((df['trend'] == '횡보/혼조') & (df['momentum_4h'] == '4H약보합')),
    'avoid_sideways_up': ~((df['trend'] == '횡보/혼조') & (df['momentum_4h'] == '4H상승')),
}

# 복합 필터 조합
filter_combos = {
    # 기본 위험 회피
    'A1: 상승+약보합 회피': avoid_conditions['avoid_up_weak'],
    'A2: 상승+약보합 + 횡보+약보합 회피': avoid_conditions['avoid_up_weak'] & avoid_conditions['avoid_sideways_weak'],
    'A3: 모든 약보합/약하락 회피': conditions['mom_not_weak'],
    
    # 모멘텀 + 추세 조합
    'B1: 4H상승 + RSI강세': conditions['mom_up'] & conditions['rsi_strong'],
    'B2: 4H강상승만': conditions['mom_strong_up'],
    'B3: 4H강상승 or 급락': conditions['mom_extreme'],
    'B4: 4H상승/강상승 + RSI>60': (conditions['mom_up'] | conditions['mom_strong_up']) & conditions['rsi_high'],
    
    # 추세 + RSI
    'C1: 상승추세 + RSI강세': conditions['trend_up'] & conditions['rsi_strong'],
    'C2: 상승추세 + RSI>70': conditions['trend_up'] & conditions['rsi_very_high'],
    'C3: 하락추세 + 급락': conditions['trend_down'] & conditions['mom_crash'],
    
    # 복합 안전 조건
    'D1: 약보합회피 + RSI강세': conditions['mom_not_weak'] & conditions['rsi_strong'],
    'D2: 약보합회피 + EMA20상승': conditions['mom_not_weak'] & conditions['ema20_up'],
    'D3: 약보합회피 + RSI>60': conditions['mom_not_weak'] & conditions['rsi_high'],
    'D4: 약보합회피 + RSI>60 + EMA20상승': conditions['mom_not_weak'] & conditions['rsi_high'] & conditions['ema20_up'],
    
    # 위치 기반 복합
    'E1: 고점근처 + 강상승': conditions['pos_near_high'] & conditions['mom_strong_up'],
    'E2: 저점 + 급락후': conditions['pos_low'] & conditions['mom_crash'],
    'E3: 24H중간 회피 + 약보합회피': conditions['pos_not_mid'] & conditions['mom_not_weak'],
    
    # 최적화 조합
    'F1: 상승+약보합회피 + RSI강세': avoid_conditions['avoid_up_weak'] & conditions['rsi_strong'],
    'F2: 상승+약보합회피 + RSI>60': avoid_conditions['avoid_up_weak'] & conditions['rsi_high'],
    'F3: 상승+약보합회피 + 4H상승': avoid_conditions['avoid_up_weak'] & conditions['mom_up'],
    'F4: 상승+약보합회피 + EMA20상승': avoid_conditions['avoid_up_weak'] & conditions['ema20_up'],
    
    # 고승률 타겟
    'G1: 극단모멘텀 + RSI강세': conditions['mom_extreme'] & conditions['rsi_strong'],
    'G2: 강상승 + RSI>60': conditions['mom_strong_up'] & conditions['rsi_high'],
    'G3: 급락 + RSI약세': conditions['mom_crash'] & conditions['rsi_oversold'],
    'G4: 강상승 + 상승추세': conditions['mom_strong_up'] & conditions['trend_up'],
    
    # MDD 집중 개선
    'H1: 약보합회피 + 중립RSI회피': conditions['mom_not_weak'] & conditions['rsi_not_neutral'],
    'H2: 약보합회피 + 횡보회피': conditions['mom_not_weak'] & conditions['trend_not_sideways'],
    'H3: 전체위험회피(상승약보합+횡보약보합+횡보상승)': avoid_conditions['avoid_up_weak'] & avoid_conditions['avoid_sideways_weak'] & avoid_conditions['avoid_sideways_up'],
    
    # 초고승률 (거래수 적어도 OK)
    'I1: 급락만': conditions['mom_crash'],
    'I2: 급락 + 하락추세': conditions['mom_crash'] & conditions['trend_down'],
    'I3: 강상승 + 상승추세 + RSI>60': conditions['mom_strong_up'] & conditions['trend_up'] & conditions['rsi_high'],
}

for name, condition in filter_combos.items():
    filtered = df[condition].copy()
    metrics = calc_metrics(filtered)
    
    if metrics is None or metrics['trades'] < 20:
        continue
    
    results.append({
        'filter': name,
        'trades': metrics['trades'],
        'block_pct': (base['trades'] - metrics['trades']) / base['trades'] * 100,
        'win_rate': metrics['win_rate'],
        'sl_rate': metrics['sl_rate'],
        'total_pnl': metrics['total_pnl'],
        'avg_pnl': metrics['avg_pnl'],
        'mdd': metrics['mdd'],
        'annual_pnl': metrics['annual_pnl'],
        'score': metrics['win_rate'] - abs(metrics['mdd'])  # 승률 - MDD절대값 (높을수록 좋음)
    })

results_df = pd.DataFrame(results).sort_values('score', ascending=False)

# 출력
print(f"\n{'필터':<45} {'거래':>5} {'차단%':>6} {'승률':>6} {'MDD':>7} {'총PNL':>7} {'평균':>6} {'점수':>6}")
print("-"*105)

for _, r in results_df.head(25).iterrows():
    win_flag = "⭐" if r['win_rate'] >= 75 else "✓" if r['win_rate'] >= 70 else ""
    mdd_flag = "⭐" if r['mdd'] >= -10 else "✓" if r['mdd'] >= -15 else ""
    print(f"{r['filter']:<45} {r['trades']:>5} {r['block_pct']:>5.0f}% {r['win_rate']:>5.1f}%{win_flag} {r['mdd']:>6.1f}%{mdd_flag} {r['total_pnl']:>6.1f}% {r['avg_pnl']:>5.2f}% {r['score']:>5.1f}")

# 목표 달성 필터 찾기
print("\n" + "="*100)
print("🏆 목표 달성 필터 (승률 70%+ AND MDD -15% 이내)")
print("="*100)

target_filters = results_df[(results_df['win_rate'] >= 70) & (results_df['mdd'] >= -15)]

if len(target_filters) > 0:
    print(f"\n{'필터':<45} {'거래':>5} {'승률':>7} {'MDD':>8} {'총PNL':>8} {'연평균':>8}")
    print("-"*95)
    for _, r in target_filters.iterrows():
        print(f"{r['filter']:<45} {r['trades']:>5} {r['win_rate']:>6.1f}% {r['mdd']:>7.1f}% {r['total_pnl']:>7.1f}% {r['annual_pnl']:>7.1f}%")
else:
    print("\n⚠️ 목표 달성 필터 없음. 조건 완화 필요.")

# 최고 승률 필터
print("\n" + "="*100)
print("⭐ 최고 승률 TOP 5 (거래 30건 이상)")
print("="*100)

high_win = results_df[results_df['trades'] >= 30].nlargest(5, 'win_rate')
for _, r in high_win.iterrows():
    print(f"\n{r['filter']}")
    print(f"  거래: {r['trades']}건, 승률: {r['win_rate']:.1f}%, MDD: {r['mdd']:.1f}%")
    print(f"  총PNL: {r['total_pnl']:.1f}%, 평균: {r['avg_pnl']:.2f}%")

# 최고 MDD 필터
print("\n" + "="*100)
print("⭐ 최저 MDD TOP 5 (거래 30건 이상)")
print("="*100)

low_mdd = results_df[results_df['trades'] >= 30].nlargest(5, 'mdd')
for _, r in low_mdd.iterrows():
    print(f"\n{r['filter']}")
    print(f"  거래: {r['trades']}건, 승률: {r['win_rate']:.1f}%, MDD: {r['mdd']:.1f}%")
    print(f"  총PNL: {r['total_pnl']:.1f}%, 평균: {r['avg_pnl']:.2f}%")

# 균형 잡힌 최적 필터
print("\n" + "="*100)
print("🎯 최종 추천 (승률 + MDD 균형)")
print("="*100)

# 점수 기준 최적
best = results_df.iloc[0]
print(f"\n💡 종합 최적: {best['filter']}")
print(f"   거래: {best['trades']}건 (기존 {base['trades']}건, -{best['block_pct']:.0f}%)")
print(f"   승률: {base['win_rate']:.1f}% → {best['win_rate']:.1f}% ({best['win_rate']-base['win_rate']:+.1f}%p)")
print(f"   MDD: {base['mdd']:.1f}% → {best['mdd']:.1f}% ({best['mdd']-base['mdd']:+.1f}%p)")
print(f"   총PNL: {base['total_pnl']:.1f}% → {best['total_pnl']:.1f}%")
print(f"   평균PNL: {base['avg_pnl']:.2f}% → {best['avg_pnl']:.2f}%")

# 저장
results_df.to_csv('filter_optimization_results.csv', index=False)
print(f"\n✅ 결과 저장: filter_optimization_results.csv")
