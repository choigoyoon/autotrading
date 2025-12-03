"""
BB 수축/확장 전략 - 파라미터 그리드 서치
모멘텀, EMA, HH/HL 등 모든 조합 테스트
"""

import pandas as pd
import numpy as np
from itertools import product

print("=" * 80)
print("BB 수축/확장 전략 - 파라미터 최적화 (그리드 서치)")
print("=" * 80)
print()

# 데이터 로드
df = pd.read_csv('expanded_squeeze_analysis.csv')

print(f"BB 수축→확장 이벤트: {len(df)}개 (5년)")
print()

# ═══════════════════════════════════════════════════════════════════
# 파라미터 그리드 정의
# ═══════════════════════════════════════════════════════════════════

# 모멘텀 임계값
m200_thresholds = [-5, 0, 2, 5, 8, 10, 12, 15, 20]
m100_thresholds = [-5, 0, 2, 5, 8, 10, 12, 15, 20]
m50_thresholds = [-5, 0, 2, 5, 8, 10, 15]

# 필터 조합
filter_types = [
    'none',           # 필터 없음
    'ema_bull',       # EMA 정배열
    'hh',             # Higher High
    'above_ema20',    # EMA20 위
    'above_ema50',    # EMA50 위
]

# 홀딩 기간
holding_periods = {
    '72h': 'long_72h',
    '168h': 'long_168h', 
    '336h': 'long_336h'
}

print("파라미터 그리드:")
print(f"  M200 임계값: {m200_thresholds}")
print(f"  M100 임계값: {m100_thresholds}")
print(f"  M50 임계값: {m50_thresholds}")
print(f"  필터 타입: {filter_types}")
print(f"  홀딩 기간: {list(holding_periods.keys())}")
print()

# ═══════════════════════════════════════════════════════════════════
# 그리드 서치 실행
# ═══════════════════════════════════════════════════════════════════

print("=" * 80)
print("그리드 서치 실행 중...")
print("=" * 80)
print()

results = []
total_combinations = 0

# 1. M200 단독
for m200 in m200_thresholds:
    for filter_type in filter_types:
        for hold_name, hold_col in holding_periods.items():
            
            # 기본 조건
            cond = df['momentum_200'] >= m200
            
            # 필터 적용
            if filter_type == 'ema_bull':
                cond = cond & (df['ema_bull'] == True)
            elif filter_type == 'hh':
                cond = cond & (df['HH'] == True)
            elif filter_type == 'above_ema20':
                cond = cond & (df['position_200'] > 0)  # close > ema20
            elif filter_type == 'above_ema50':
                cond = cond & (df['position_200'] > 0.5)
            
            subset = df[cond]
            
            if len(subset) >= 10:  # 최소 10개 거래
                wins = (subset[hold_col] > 0).sum()
                wr = wins / len(subset) * 100
                avg = subset[hold_col].mean()
                total = subset[hold_col].sum()
                
                results.append({
                    'strategy': f'M200≥{m200}',
                    'filter': filter_type,
                    'holding': hold_name,
                    'count': len(subset),
                    'annual': len(subset) / 5,
                    'win_rate': wr,
                    'avg_pnl': avg,
                    'total_pnl': total,
                    'score': wr * avg  # 복합 점수
                })
            
            total_combinations += 1

# 2. M100 단독
for m100 in m100_thresholds:
    for filter_type in filter_types:
        for hold_name, hold_col in holding_periods.items():
            
            cond = df['momentum_100'] >= m100
            
            if filter_type == 'ema_bull':
                cond = cond & (df['ema_bull'] == True)
            elif filter_type == 'hh':
                cond = cond & (df['HH'] == True)
            elif filter_type == 'above_ema20':
                cond = cond & (df['position_200'] > 0)
            elif filter_type == 'above_ema50':
                cond = cond & (df['position_200'] > 0.5)
            
            subset = df[cond]
            
            if len(subset) >= 10:
                wins = (subset[hold_col] > 0).sum()
                wr = wins / len(subset) * 100
                avg = subset[hold_col].mean()
                total = subset[hold_col].sum()
                
                results.append({
                    'strategy': f'M100≥{m100}',
                    'filter': filter_type,
                    'holding': hold_name,
                    'count': len(subset),
                    'annual': len(subset) / 5,
                    'win_rate': wr,
                    'avg_pnl': avg,
                    'total_pnl': total,
                    'score': wr * avg
                })
            
            total_combinations += 1

# 3. M200 + M100 복합
for m200 in [0, 5, 10]:
    for m100 in [0, 5, 10]:
        for filter_type in filter_types:
            for hold_name, hold_col in holding_periods.items():
                
                cond = (df['momentum_200'] >= m200) & (df['momentum_100'] >= m100)
                
                if filter_type == 'ema_bull':
                    cond = cond & (df['ema_bull'] == True)
                elif filter_type == 'hh':
                    cond = cond & (df['HH'] == True)
                elif filter_type == 'above_ema20':
                    cond = cond & (df['position_200'] > 0)
                elif filter_type == 'above_ema50':
                    cond = cond & (df['position_200'] > 0.5)
                
                subset = df[cond]
                
                if len(subset) >= 10:
                    wins = (subset[hold_col] > 0).sum()
                    wr = wins / len(subset) * 100
                    avg = subset[hold_col].mean()
                    total = subset[hold_col].sum()
                    
                    results.append({
                        'strategy': f'M200≥{m200}+M100≥{m100}',
                        'filter': filter_type,
                        'holding': hold_name,
                        'count': len(subset),
                        'annual': len(subset) / 5,
                        'win_rate': wr,
                        'avg_pnl': avg,
                        'total_pnl': total,
                        'score': wr * avg
                    })
                
                total_combinations += 1

print(f"테스트한 조합: {total_combinations}개")
print(f"유효한 결과: {len(results)}개")
print()

# ═══════════════════════════════════════════════════════════════════
# 결과 분석
# ═══════════════════════════════════════════════════════════════════

df_results = pd.DataFrame(results)

print("=" * 80)
print("TOP 20 전략 (복합 점수 기준)")
print("=" * 80)
print()

top20 = df_results.nlargest(20, 'score')

for idx, row in top20.iterrows():
    print(f"【{row['strategy']} + {row['filter']} | {row['holding']}】")
    print(f"  거래: {row['count']:.0f}개 (연 {row['annual']:.1f})")
    print(f"  승률: {row['win_rate']:.1f}% | 평균: {row['avg_pnl']:+.2f}% | 누적: {row['total_pnl']:+.1f}%")
    print(f"  점수: {row['score']:.1f}")
    print()

# ═══════════════════════════════════════════════════════════════════
# 매매 횟수별 최고 전략
# ═══════════════════════════════════════════════════════════════════

print("=" * 80)
print("매매 횟수별 최고 전략")
print("=" * 80)
print()

# 연간 50회 이상
print("🎯 고빈도 (연 50회 이상):")
high_freq = df_results[df_results['annual'] >= 50].nlargest(5, 'score')
if len(high_freq) > 0:
    for idx, row in high_freq.iterrows():
        print(f"  {row['strategy']:20s} + {row['filter']:15s} | {row['holding']:5s} | 연 {row['annual']:4.0f}회 | 승률 {row['win_rate']:5.1f}% | 평균 {row['avg_pnl']:+5.2f}%")
else:
    print("  조건 충족 전략 없음")
print()

# 연간 30-50회
print("🎯 중빈도 (연 30-50회):")
mid_freq = df_results[(df_results['annual'] >= 30) & (df_results['annual'] < 50)].nlargest(5, 'score')
if len(mid_freq) > 0:
    for idx, row in mid_freq.iterrows():
        print(f"  {row['strategy']:20s} + {row['filter']:15s} | {row['holding']:5s} | 연 {row['annual']:4.0f}회 | 승률 {row['win_rate']:5.1f}% | 평균 {row['avg_pnl']:+5.2f}%")
else:
    print("  조건 충족 전략 없음")
print()

# 연간 15-30회
print("🎯 저빈도 (연 15-30회):")
low_freq = df_results[(df_results['annual'] >= 15) & (df_results['annual'] < 30)].nlargest(5, 'score')
if len(low_freq) > 0:
    for idx, row in low_freq.iterrows():
        print(f"  {row['strategy']:20s} + {row['filter']:15s} | {row['holding']:5s} | 연 {row['annual']:4.0f}회 | 승률 {row['win_rate']:5.1f}% | 평균 {row['avg_pnl']:+5.2f}%")
else:
    print("  조건 충족 전략 없음")
print()

# ═══════════════════════════════════════════════════════════════════
# 홀딩 기간별 최고 전략
# ═══════════════════════════════════════════════════════════════════

print("=" * 80)
print("홀딩 기간별 최고 전략")
print("=" * 80)
print()

for hold in ['72h', '168h', '336h']:
    hold_best = df_results[df_results['holding'] == hold].nlargest(3, 'score')
    print(f"【{hold} 홀딩】")
    for idx, row in hold_best.iterrows():
        print(f"  {row['strategy']:20s} + {row['filter']:15s} | 연 {row['annual']:4.0f}회 | 승률 {row['win_rate']:5.1f}% | 평균 {row['avg_pnl']:+5.2f}%")
    print()

# ═══════════════════════════════════════════════════════════════════
# 최종 추천
# ═══════════════════════════════════════════════════════════════════

print("=" * 80)
print("최종 추천 전략")
print("=" * 80)
print()

# 기준: 연 30회 이상, 승률 55% 이상, 평균 3% 이상
recommended = df_results[
    (df_results['annual'] >= 30) & 
    (df_results['win_rate'] >= 55) & 
    (df_results['avg_pnl'] >= 3)
].sort_values('score', ascending=False)

if len(recommended) > 0:
    print("✅ 추천 조건 충족 전략:")
    print("   (연 30회 이상 & 승률 55% 이상 & 평균 3% 이상)")
    print()
    
    for idx, row in recommended.head(10).iterrows():
        print(f"【{row['strategy']} + {row['filter']} | {row['holding']}】")
        print(f"  거래: 연 {row['annual']:.1f}회 | 승률: {row['win_rate']:.1f}% | 평균: {row['avg_pnl']:+.2f}% | 누적: {row['total_pnl']:+.1f}%")
        print()
else:
    print("⚠️ 모든 조건을 충족하는 전략 없음")
    print()
    print("완화된 기준으로 추천:")
    relaxed = df_results[
        (df_results['annual'] >= 20) & 
        (df_results['win_rate'] >= 52) & 
        (df_results['avg_pnl'] >= 2)
    ].sort_values('score', ascending=False).head(5)
    
    for idx, row in relaxed.iterrows():
        print(f"  {row['strategy']:20s} + {row['filter']:15s} | {row['holding']:5s}")
        print(f"    연 {row['annual']:.1f}회 | 승률 {row['win_rate']:.1f}% | 평균 {row['avg_pnl']:+.2f}% | 누적 {row['total_pnl']:+.1f}%")

# 결과 저장
df_results.to_csv('parameter_optimization_results.csv', index=False)
print()
print("💾 결과 저장: parameter_optimization_results.csv")

print()
print("=" * 80)
print("✅ 최적화 완료")
print("=" * 80)
