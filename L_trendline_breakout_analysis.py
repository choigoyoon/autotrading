#!/usr/bin/env python3
"""
L-value와 추세선 돌파의 관계 심층 분석
=====================================
핵심 질문: L값(손절 기준)이 추세선 돌파 성공/실패에 어떤 영향을 미치는가?
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 80)
print("L-VALUE & TRENDLINE BREAKOUT DEEP ANALYSIS")
print("=" * 80)

# 데이터 로드
backtest = pd.read_csv('L_value_backtest.csv')
hl_pattern = pd.read_csv('hl_pattern_analysis.csv')

backtest['entry_time'] = pd.to_datetime(backtest['entry_time'])
backtest['exit_time'] = pd.to_datetime(backtest['exit_time'])
hl_pattern['breakout_time'] = pd.to_datetime(hl_pattern['breakout_time'])

# 데이터 병합
merged = pd.merge(backtest, hl_pattern, 
                  left_on='entry_time', 
                  right_on='breakout_time', 
                  how='inner')

print(f"\n총 거래 수: {len(merged)}")
print(f"승률: {(merged['total_pnl_x'] > 0).mean()*100:.1f}%")
print(f"평균 수익: {merged['total_pnl_x'].mean():.2f}%")

# ============================================================
# 1. L-VALUE의 의미 분석
# ============================================================
print("\n" + "=" * 80)
print("1. L-VALUE의 의미")
print("=" * 80)

print("""
L-value 정의:
- L1: H1-H2 추세선 돌파 전 첫 번째 저점
- L2: H1-H2 추세선 돌파 전 두 번째 저점 (더 최근)
- L-value = L2 (손절 기준점)

L_change = (L2 - L1) / L1 * 100
- L_change > 0: 저점 상승 (저점이 높아짐) = 매수세 유입
- L_change < 0: 저점 하락 (저점이 낮아짐) = 매수세 약화
""")

# ============================================================
# 2. L_CHANGE별 성과 분석
# ============================================================
print("\n" + "=" * 80)
print("2. L_CHANGE (저점 변화율)별 성과")
print("=" * 80)

# L_change 구간별 분석
bins = [-np.inf, -2, -1, -0.5, 0, 0.5, 1, 2, np.inf]
labels = ['<-2%', '-2~-1%', '-1~-0.5%', '-0.5~0%', '0~0.5%', '0.5~1%', '1~2%', '>2%']
merged['l_change_bin'] = pd.cut(merged['l_change'], bins=bins, labels=labels)

l_change_perf = merged.groupby('l_change_bin', observed=True).agg({
    'total_pnl_x': ['count', 'mean', lambda x: (x > 0).mean() * 100],
    'sl_done_x': 'mean',
    'tp2_done_x': 'mean'
}).round(2)
l_change_perf.columns = ['거래수', '평균수익%', '승률%', 'SL율', 'TP율']
l_change_perf['SL율'] = (l_change_perf['SL율'] * 100).round(1)
l_change_perf['TP율'] = (l_change_perf['TP율'] * 100).round(1)

print("\n[L_change 구간별 성과]")
print(l_change_perf.to_string())

# ============================================================
# 3. ENTRY-L GAP 분석 (진입가-손절선 거리)
# ============================================================
print("\n" + "=" * 80)
print("3. ENTRY-L GAP (진입가 - L값 거리) 분석")
print("=" * 80)

print("""
Entry-L Gap의 의미:
- Entry-L Gap = (진입가 - L2) / L2 * 100
- 클수록: 진입가가 L값보다 높음 = 손절 여유 있음
- 작을수록: 진입가가 L값에 가까움 = 손절 위험 높음
""")

bins = [0, 0.5, 1, 1.5, 2, 3, np.inf]
labels = ['0-0.5%', '0.5-1%', '1-1.5%', '1.5-2%', '2-3%', '3%+']
merged['entry_L_bin'] = pd.cut(merged['entry_L_gap'], bins=bins, labels=labels)

entry_L_perf = merged.groupby('entry_L_bin', observed=True).agg({
    'total_pnl_x': ['count', 'mean', lambda x: (x > 0).mean() * 100],
    'sl_done_x': 'mean',
    'tp2_done_x': 'mean'
}).round(2)
entry_L_perf.columns = ['거래수', '평균수익%', '승률%', 'SL율', 'TP율']
entry_L_perf['SL율'] = (entry_L_perf['SL율'] * 100).round(1)
entry_L_perf['TP율'] = (entry_L_perf['TP율'] * 100).round(1)

print("\n[Entry-L Gap 구간별 성과]")
print(entry_L_perf.to_string())

# ============================================================
# 4. H_CHANGE와 L_CHANGE 조합 분석
# ============================================================
print("\n" + "=" * 80)
print("4. H_CHANGE & L_CHANGE 패턴 조합 분석")
print("=" * 80)

print("""
패턴 해석:
- H↑ L↑: 고점↑ 저점↑ = 상승 추세
- H↑ L↓: 고점↑ 저점↓ = 확장 패턴 (변동성 증가)
- H↓ L↑: 고점↓ 저점↑ = 수렴 패턴 (삼각수렴)
- H↓ L↓: 고점↓ 저점↓ = 하락 추세
""")

pattern_perf = merged.groupby('pattern', observed=True).agg({
    'total_pnl_x': ['count', 'mean', lambda x: (x > 0).mean() * 100],
    'sl_done_x': 'mean',
    'tp2_done_x': 'mean',
    'hold_hours': 'mean'
}).round(2)
pattern_perf.columns = ['거래수', '평균수익%', '승률%', 'SL율', 'TP율', '보유시간']
pattern_perf['SL율'] = (pattern_perf['SL율'] * 100).round(1)
pattern_perf['TP율'] = (pattern_perf['TP율'] * 100).round(1)

print("\n[패턴별 성과]")
print(pattern_perf.sort_values('승률%', ascending=False).to_string())

# ============================================================
# 5. L_CHANGE 크기별 상세 분석
# ============================================================
print("\n" + "=" * 80)
print("5. L-VALUE 움직임 크기 분석")
print("=" * 80)

# L_change 절대값 분석
merged['l_change_abs'] = merged['l_change'].abs()
merged['l_change_sign'] = np.where(merged['l_change'] > 0, '저점상승(L↑)', '저점하락(L↓)')

# 저점 상승 vs 저점 하락
print("\n[저점 방향별 성과]")
direction_perf = merged.groupby('l_change_sign', observed=True).agg({
    'total_pnl_x': ['count', 'mean', lambda x: (x > 0).mean() * 100],
    'sl_done_x': 'mean',
    'tp2_done_x': 'mean'
}).round(2)
direction_perf.columns = ['거래수', '평균수익%', '승률%', 'SL율', 'TP율']
direction_perf['SL율'] = (direction_perf['SL율'] * 100).round(1)
direction_perf['TP율'] = (direction_perf['TP율'] * 100).round(1)
print(direction_perf.to_string())

# L_change 절대값 크기별
print("\n[L 변화 크기별 성과]")
merged['l_change_size'] = pd.cut(merged['l_change_abs'], 
                                  bins=[0, 0.3, 0.5, 1, 2, np.inf],
                                  labels=['미미(~0.3%)', '작음(0.3~0.5%)', '보통(0.5~1%)', '큼(1~2%)', '매우큼(2%+)'])
size_perf = merged.groupby('l_change_size', observed=True).agg({
    'total_pnl_x': ['count', 'mean', lambda x: (x > 0).mean() * 100],
    'sl_done_x': 'mean',
    'tp2_done_x': 'mean'
}).round(2)
size_perf.columns = ['거래수', '평균수익%', '승률%', 'SL율', 'TP율']
size_perf['SL율'] = (size_perf['SL율'] * 100).round(1)
size_perf['TP율'] = (size_perf['TP율'] * 100).round(1)
print(size_perf.to_string())

# ============================================================
# 6. L-VALUE와 추세선 돌파 실패 원인 분석
# ============================================================
print("\n" + "=" * 80)
print("6. L-VALUE 관련 추세선 돌파 실패 원인 분석")
print("=" * 80)

# 실패 케이스 (SL 터짐)
fail_cases = merged[merged['sl_done_x'] == True]
success_cases = merged[merged['tp2_done_x'] == True]

print(f"\n실패 케이스: {len(fail_cases)}건 ({len(fail_cases)/len(merged)*100:.1f}%)")
print(f"성공 케이스: {len(success_cases)}건 ({len(success_cases)/len(merged)*100:.1f}%)")

print("\n[실패 vs 성공 비교]")
comparison = pd.DataFrame({
    '항목': ['Entry-L Gap', 'L_change', 'H_change', '보유시간'],
    '실패 평균': [
        fail_cases['entry_L_gap'].mean(),
        fail_cases['l_change'].mean(),
        fail_cases['h_change'].mean(),
        fail_cases['hold_hours'].mean()
    ],
    '성공 평균': [
        success_cases['entry_L_gap'].mean(),
        success_cases['l_change'].mean(),
        success_cases['h_change'].mean(),
        success_cases['hold_hours'].mean()
    ]
})
comparison['차이'] = comparison['성공 평균'] - comparison['실패 평균']
print(comparison.to_string(index=False))

# ============================================================
# 7. 최적 조건 탐색
# ============================================================
print("\n" + "=" * 80)
print("7. 최적 조건 탐색")
print("=" * 80)

# Entry-L Gap >= 1% 필터
entry_l_filter = merged['entry_L_gap'] >= 1.0
print(f"\n[필터: Entry-L Gap >= 1%]")
print(f"거래수: {entry_l_filter.sum()} -> {len(merged)}에서 감소")
print(f"승률: {(merged[entry_l_filter]['total_pnl_x'] > 0).mean()*100:.1f}%")
print(f"평균수익: {merged[entry_l_filter]['total_pnl_x'].mean():.2f}%")
print(f"총수익: {merged[entry_l_filter]['total_pnl_x'].sum():.1f}%")

# L_change < 0 (저점 하락) 필터
l_down_filter = merged['l_change'] < 0
print(f"\n[필터: L_change < 0 (저점 하락)]")
print(f"거래수: {l_down_filter.sum()}")
print(f"승률: {(merged[l_down_filter]['total_pnl_x'] > 0).mean()*100:.1f}%")
print(f"평균수익: {merged[l_down_filter]['total_pnl_x'].mean():.2f}%")
print(f"총수익: {merged[l_down_filter]['total_pnl_x'].sum():.1f}%")

# L_change < -1% (저점 큰 하락) 필터
l_big_down_filter = merged['l_change'] < -1
print(f"\n[필터: L_change < -1% (저점 큰 하락)]")
print(f"거래수: {l_big_down_filter.sum()}")
print(f"승률: {(merged[l_big_down_filter]['total_pnl_x'] > 0).mean()*100:.1f}%")
print(f"평균수익: {merged[l_big_down_filter]['total_pnl_x'].mean():.2f}%")
print(f"총수익: {merged[l_big_down_filter]['total_pnl_x'].sum():.1f}%")

# 조합: Entry-L >= 1% AND L_change < 0
combo_filter = entry_l_filter & l_down_filter
print(f"\n[조합필터: Entry-L >= 1% AND L_change < 0]")
print(f"거래수: {combo_filter.sum()}")
print(f"승률: {(merged[combo_filter]['total_pnl_x'] > 0).mean()*100:.1f}%")
print(f"평균수익: {merged[combo_filter]['total_pnl_x'].mean():.2f}%")
print(f"총수익: {merged[combo_filter]['total_pnl_x'].sum():.1f}%")

# ============================================================
# 8. L-VALUE와 추세선의 관계 시각화
# ============================================================
print("\n" + "=" * 80)
print("8. L-VALUE와 추세선 돌파의 핵심 인사이트")
print("=" * 80)

print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                    L-VALUE와 추세선 돌파 관계                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  추세선 (H1-H2 연결)                                                    │
│      H1 ─────────────────────                                           │
│         ╲                                                               │
│          ╲  H2                                                          │
│           ╲──────╲                                                      │
│                   ╲   ← 추세선 돌파 지점 (Entry)                        │
│                    ╲                                                    │
│                                                                         │
│  저점 형성                                                              │
│                                                                         │
│     L1 ●─────────●                                                      │
│                   L2  ← L-value (손절 기준점)                           │
│                                                                         │
│  Entry-L Gap = Entry Price - L2                                         │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  핵심 발견:                                                             │
│                                                                         │
│  1. L_change < 0 (저점 하락)                                            │
│     - L2 < L1 → 저점이 낮아짐                                           │
│     - 해석: 추가 매도 압력으로 더 눌림                                  │
│     - 결과: 돌파 시 반등 에너지 축적 → 성공률 높음                      │
│                                                                         │
│  2. L_change > 0 (저점 상승)                                            │
│     - L2 > L1 → 저점이 높아짐                                           │
│     - 해석: 매수세가 일찍 들어옴                                        │
│     - 결과: 이미 가격이 올라 돌파 힘 약함 → 페이크 가능성               │
│                                                                         │
│  3. Entry-L Gap 중요성                                                  │
│     - Gap 작음: L값에 가까움 → 손절 위험 높음                           │
│     - Gap 큼: L값 여유 있음 → 버틸 공간 있음                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 9. 실전 적용 가이드
# ============================================================
print("\n" + "=" * 80)
print("9. 실전 적용 가이드")
print("=" * 80)

# 최종 추천 조건 테스트
conditions = [
    ('기본 (필터없음)', merged['total_pnl_x'] > -999),
    ('Entry-L >= 1%', merged['entry_L_gap'] >= 1.0),
    ('Entry-L >= 1.5%', merged['entry_L_gap'] >= 1.5),
    ('L_change < 0', merged['l_change'] < 0),
    ('L_change < -0.5%', merged['l_change'] < -0.5),
    ('H↑ L↓ 또는 H↓ L↓', merged['pattern'].isin(['H↑ L↓', 'H↓ L↓'])),
    ('Entry-L>=1% + L↓', (merged['entry_L_gap'] >= 1.0) & (merged['l_change'] < 0)),
    ('Entry-L>=1% + L<-0.5%', (merged['entry_L_gap'] >= 1.0) & (merged['l_change'] < -0.5)),
]

print("\n[조건별 성과 비교]")
print("-" * 90)
print(f"{'조건':<30} {'거래수':>8} {'승률%':>8} {'평균수익%':>10} {'총수익%':>10} {'SL율%':>8}")
print("-" * 90)

for name, cond in conditions:
    subset = merged[cond]
    if len(subset) > 0:
        trades = len(subset)
        winrate = (subset['total_pnl_x'] > 0).mean() * 100
        avg_pnl = subset['total_pnl_x'].mean()
        total_pnl = subset['total_pnl_x'].sum()
        sl_rate = subset['sl_done_x'].mean() * 100
        print(f"{name:<30} {trades:>8} {winrate:>8.1f} {avg_pnl:>10.2f} {total_pnl:>10.1f} {sl_rate:>8.1f}")

print("-" * 90)

# ============================================================
# 10. 결론
# ============================================================
print("\n" + "=" * 80)
print("10. 최종 결론")
print("=" * 80)

# 최적 조합 찾기
best_combo = (merged['entry_L_gap'] >= 1.0) & (merged['l_change'] < -0.5)
best_subset = merged[best_combo]

print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│                     L-VALUE 분석 최종 결론                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ▶ L-value의 핵심 역할:                                                 │
│    - 손절 기준점 (L2 = 돌파 직전 저점)                                  │
│    - Entry-L Gap이 클수록 안전한 거래                                   │
│                                                                         │
│  ▶ 핵심 발견:                                                           │
│    1. 저점 하락 (L_change < 0) 시 승률/수익률 상승                      │
│       → 충분히 눌린 후 돌파할 때 반등력이 강함                          │
│                                                                         │
│    2. Entry-L Gap >= 1% 필터로 페이크아웃 감소                          │
│       → 손절선과 거리 있어야 버틸 공간 확보                             │
│                                                                         │
│  ▶ 추천 필터 조합:                                                      │
│    - Entry-L Gap >= 1%                                                  │
│    - L_change < -0.5% (저점이 0.5% 이상 하락한 케이스)                  │
│                                                                         │
│  ▶ 예상 성과 (필터 적용 시):                                            │
│    - 거래수: {len(best_subset)}건 (원본 {len(merged)}건에서 감소)                               │
│    - 승률: {(best_subset['total_pnl_x'] > 0).mean()*100:.1f}% (원본 {(merged['total_pnl_x'] > 0).mean()*100:.1f}%에서 상승)                               │
│    - 평균수익: {best_subset['total_pnl_x'].mean():.2f}% (원본 {merged['total_pnl_x'].mean():.2f}%에서 상승)                            │
│                                                                         │
│  ▶ 핵심 인사이트:                                                       │
│    "저점이 충분히 눌린 후(L_change < 0)                                 │
│     진입가가 L값보다 여유있을 때(Entry-L >= 1%)                         │
│     추세선 돌파의 성공 확률이 높다"                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
""")

# 결과 저장
merged.to_csv('L_trendline_analysis_detailed.csv', index=False)
print("\n분석 결과가 'L_trendline_analysis_detailed.csv'에 저장되었습니다.")
