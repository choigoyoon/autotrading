#!/usr/bin/env python3
"""
L-value 심층 패턴 분석
- L_change 구간별 비선형적 성과 분석
- 최적 진입 구간 탐색
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("L-VALUE 심층 패턴 분석")
print("=" * 80)

# 데이터 로드
backtest = pd.read_csv('L_value_backtest.csv')
hl_pattern = pd.read_csv('hl_pattern_analysis.csv')

backtest['entry_time'] = pd.to_datetime(backtest['entry_time'])
hl_pattern['breakout_time'] = pd.to_datetime(hl_pattern['breakout_time'])

# 병합
merged = pd.merge(backtest, hl_pattern, 
                  left_on='entry_time', 
                  right_on='breakout_time', 
                  how='inner')

# ============================================================
# 1. L_CHANGE 비선형 패턴 발견
# ============================================================
print("\n" + "=" * 80)
print("1. L_CHANGE 비선형 패턴 발견")
print("=" * 80)

print("""
관찰된 패턴:
- L < -2%    : 75% 승률 (크게 눌림 → 강한 반등)
- L = -2~-1% : 74% 승률 (적당히 눌림 → 좋은 반등)
- L = -1~-0.5%: 38% 승률 (애매한 구간)
- L = -0.5~0%: 38% 승률 (방향 불명확)
- L = 0~0.5% : 56% 승률 (저점 상승 시작)
- L = 0.5~1% : 39% 승률 (과한 상승)
- L > 2%     : 58% 승률 (강한 상승 모멘텀)

→ "U자형 커브": 극단적인 하락(-2% 이하) 또는 적당한 상승(0~0.5%)에서 좋은 성과
""")

# ============================================================
# 2. "골디락스 존" 찾기
# ============================================================
print("\n" + "=" * 80)
print("2. '골디락스 존' (최적 구간) 찾기")
print("=" * 80)

# 세밀한 구간 분석
l_ranges = [
    ('큰 하락 (L < -2%)', merged['l_change'] < -2),
    ('하락 (-2% ~ -1%)', (merged['l_change'] >= -2) & (merged['l_change'] < -1)),
    ('약하락 (-1% ~ -0.5%)', (merged['l_change'] >= -1) & (merged['l_change'] < -0.5)),
    ('보합 (-0.5% ~ 0%)', (merged['l_change'] >= -0.5) & (merged['l_change'] < 0)),
    ('약상승 (0% ~ 0.5%)', (merged['l_change'] >= 0) & (merged['l_change'] < 0.5)),
    ('상승 (0.5% ~ 1%)', (merged['l_change'] >= 0.5) & (merged['l_change'] < 1)),
    ('큰 상승 (1% ~ 2%)', (merged['l_change'] >= 1) & (merged['l_change'] < 2)),
    ('매우큰상승 (L > 2%)', merged['l_change'] >= 2),
]

print("\n[L_change 세밀 구간 분석]")
print("-" * 70)
print(f"{'구간':<25} {'거래수':>8} {'승률%':>8} {'평균PnL%':>10} {'총PnL%':>10}")
print("-" * 70)

good_zones = []
for name, cond in l_ranges:
    subset = merged[cond]
    if len(subset) > 0:
        trades = len(subset)
        winrate = (subset['total_pnl_x'] > 0).mean() * 100
        avg_pnl = subset['total_pnl_x'].mean()
        total_pnl = subset['total_pnl_x'].sum()
        marker = " ★★★" if winrate >= 60 else (" ★" if winrate >= 50 else "")
        print(f"{name:<25} {trades:>8} {winrate:>8.1f} {avg_pnl:>10.2f} {total_pnl:>10.1f}{marker}")
        if winrate >= 55 and trades >= 10:
            good_zones.append((name, cond, winrate, avg_pnl))
print("-" * 70)

# ============================================================
# 3. Entry-L Gap + L_change 교차 분석
# ============================================================
print("\n" + "=" * 80)
print("3. Entry-L Gap × L_change 교차 분석")
print("=" * 80)

# Entry-L Gap 구간
entry_l_ranges = [
    ('Entry-L 0-0.5%', (merged['entry_L_gap'] >= 0) & (merged['entry_L_gap'] < 0.5)),
    ('Entry-L 0.5-1%', (merged['entry_L_gap'] >= 0.5) & (merged['entry_L_gap'] < 1)),
    ('Entry-L 1-1.5%', (merged['entry_L_gap'] >= 1) & (merged['entry_L_gap'] < 1.5)),
    ('Entry-L 1.5-2%', (merged['entry_L_gap'] >= 1.5) & (merged['entry_L_gap'] < 2)),
    ('Entry-L 2%+', merged['entry_L_gap'] >= 2),
]

# L_change 방향
l_directions = [
    ('L↓ 큰하락(<-1%)', merged['l_change'] < -1),
    ('L↓ 하락(-1~0%)', (merged['l_change'] >= -1) & (merged['l_change'] < 0)),
    ('L↑ 상승(0~1%)', (merged['l_change'] >= 0) & (merged['l_change'] < 1)),
    ('L↑ 큰상승(>1%)', merged['l_change'] >= 1),
]

print("\n[교차 분석 테이블 - 승률%]")
print("-" * 90)
print(f"{'Entry-L \\ L_change':<20}", end="")
for l_name, _ in l_directions:
    print(f"{l_name:>16}", end="")
print()
print("-" * 90)

for e_name, e_cond in entry_l_ranges:
    print(f"{e_name:<20}", end="")
    for l_name, l_cond in l_directions:
        subset = merged[e_cond & l_cond]
        if len(subset) >= 3:
            winrate = (subset['total_pnl_x'] > 0).mean() * 100
            trades = len(subset)
            marker = "★" if winrate >= 60 else ""
            print(f"{winrate:>10.0f}%({trades:>2}){marker:1}", end="")
        else:
            print(f"{'N/A':>16}", end="")
    print()
print("-" * 90)

# ============================================================
# 4. 가장 좋은 조합 찾기
# ============================================================
print("\n" + "=" * 80)
print("4. 최적 조합 탐색")
print("=" * 80)

combinations = []
for e_name, e_cond in entry_l_ranges:
    for l_name, l_cond in l_directions:
        subset = merged[e_cond & l_cond]
        if len(subset) >= 5:
            winrate = (subset['total_pnl_x'] > 0).mean() * 100
            avg_pnl = subset['total_pnl_x'].mean()
            total_pnl = subset['total_pnl_x'].sum()
            combinations.append({
                'Entry-L': e_name,
                'L_change': l_name,
                'trades': len(subset),
                'winrate': winrate,
                'avg_pnl': avg_pnl,
                'total_pnl': total_pnl,
                'score': winrate * 0.3 + avg_pnl * 20 + (total_pnl / 10) * 0.2
            })

combo_df = pd.DataFrame(combinations)
combo_df = combo_df.sort_values('winrate', ascending=False)

print("\n[승률 기준 TOP 10 조합]")
print(combo_df.head(10).to_string(index=False))

# ============================================================
# 5. 최종 추천 필터
# ============================================================
print("\n" + "=" * 80)
print("5. 최종 추천 필터 조합")
print("=" * 80)

# 최적 조합 1: Entry-L >= 1% AND (L < -1% OR L = 0~0.5%)
filter1 = (merged['entry_L_gap'] >= 1.0) & ((merged['l_change'] < -1) | ((merged['l_change'] >= 0) & (merged['l_change'] < 0.5)))
subset1 = merged[filter1]
print(f"\n[조합1] Entry-L >= 1% AND (L < -1% OR L = 0~0.5%)")
print(f"  거래수: {len(subset1)}")
print(f"  승률: {(subset1['total_pnl_x'] > 0).mean()*100:.1f}%")
print(f"  평균수익: {subset1['total_pnl_x'].mean():.2f}%")
print(f"  총수익: {subset1['total_pnl_x'].sum():.1f}%")

# 최적 조합 2: Entry-L 1-2% (최적 구간)
filter2 = (merged['entry_L_gap'] >= 1.0) & (merged['entry_L_gap'] < 2.0)
subset2 = merged[filter2]
print(f"\n[조합2] Entry-L 1-2% (스윗스팟)")
print(f"  거래수: {len(subset2)}")
print(f"  승률: {(subset2['total_pnl_x'] > 0).mean()*100:.1f}%")
print(f"  평균수익: {subset2['total_pnl_x'].mean():.2f}%")
print(f"  총수익: {subset2['total_pnl_x'].sum():.1f}%")

# 최적 조합 3: L 큰하락 (<-1%)
filter3 = merged['l_change'] < -1
subset3 = merged[filter3]
print(f"\n[조합3] L < -1% (큰 하락 후 돌파)")
print(f"  거래수: {len(subset3)}")
print(f"  승률: {(subset3['total_pnl_x'] > 0).mean()*100:.1f}%")
print(f"  평균수익: {subset3['total_pnl_x'].mean():.2f}%")
print(f"  총수익: {subset3['total_pnl_x'].sum():.1f}%")

# 최적 조합 4: Entry-L 1-2% + L < 0
filter4 = (merged['entry_L_gap'] >= 1.0) & (merged['entry_L_gap'] < 2.0) & (merged['l_change'] < 0)
subset4 = merged[filter4]
print(f"\n[조합4] Entry-L 1-2% AND L < 0 (스윗스팟 + 저점하락)")
print(f"  거래수: {len(subset4)}")
print(f"  승률: {(subset4['total_pnl_x'] > 0).mean()*100:.1f}%")
print(f"  평균수익: {subset4['total_pnl_x'].mean():.2f}%")
print(f"  총수익: {subset4['total_pnl_x'].sum():.1f}%")

# ============================================================
# 6. 결론 다이어그램
# ============================================================
print("\n" + "=" * 80)
print("6. L-VALUE 분석 최종 결론")
print("=" * 80)

print("""
┌────────────────────────────────────────────────────────────────────────────┐
│              L-VALUE와 추세선 돌파 관계 - 핵심 발견                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  1. L_change 비선형 패턴 (U자형 커브):                                     │
│     ┌──────────────────────────────────────────────────────────────────┐  │
│     │  승률 75% ★★★          승률 56% ★                               │  │
│     │     ▲                      ▲                                     │  │
│     │    ╱ ╲                   ╱  ╲                                    │  │
│     │   ╱   ╲    38%         ╱    ╲  39%                               │  │
│     │  ╱     ╲──────────────╱      ╲───                                │  │
│     │ ╱       │           │        │                                   │  │
│     │<-2%   -1%   -0.5%   0%     0.5%   1%   >2%  → L_change           │  │
│     │                                                                  │  │
│     │ [큰하락]      [데드존]      [적당상승]  [과상승]                  │  │
│     └──────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  2. Entry-L Gap 영향:                                                      │
│     ┌──────────────────────────────────────────────────────────────────┐  │
│     │                          76% ★★                                  │  │
│     │                           ▲                                       │  │
│     │  27%                ▲74% │                                       │  │
│     │   │         44%    │     │     47%    49%                        │  │
│     │   ├────────────────┤     ├──────┴──────┤                         │  │
│     │  0%  0.5%    1%   1.5%   2%   2.5%   3%+  → Entry-L Gap          │  │
│     │                                                                  │  │
│     │ [위험]      [스윗스팟 1-2%]        [안전하지만 힘약함]           │  │
│     └──────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  3. 최적 진입 조건:                                                        │
│     ✓ Entry-L Gap: 1% ~ 2% (손절 여유 + 돌파력 유지)                      │
│     ✓ L_change: < -1% (크게 눌린 후) 또는 0~0.5% (저점 다지기 완료)       │
│                                                                            │
│  4. 피해야 할 구간:                                                        │
│     ✗ Entry-L Gap < 0.5%: 손절 위험 너무 높음 (27% 승률)                  │
│     ✗ L_change -0.5%~0%: 방향 불명확 (38% 승률)                           │
│     ✗ L_change 0.5%~1%: 이미 상승 많이 함 (39% 승률)                      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 7. 실전 매매 체크리스트
# ============================================================
print("\n" + "=" * 80)
print("7. 실전 매매 체크리스트")
print("=" * 80)

print("""
┌────────────────────────────────────────────────────────────────────────────┐
│                        추세선 돌파 진입 체크리스트                         │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  □ 1. H1-H2 하향 추세선 형성 확인                                          │
│                                                                            │
│  □ 2. L1, L2 저점 확인                                                     │
│     - L2 = 손절 기준선 (L-value)                                           │
│                                                                            │
│  □ 3. L_change 체크: (L2 - L1) / L1 × 100                                  │
│     ✓ L_change < -1% : 최고 (충분히 눌림)                                  │
│     ✓ L_change 0~0.5%: 좋음 (저점 다지기)                                  │
│     ✗ L_change -0.5%~0%: 위험 (방향 불명확)                                │
│     ✗ L_change 0.5%~1%: 위험 (이미 올라감)                                 │
│                                                                            │
│  □ 4. Entry-L Gap 체크: (Entry - L2) / L2 × 100                            │
│     ✓ 1.5% ~ 2%: 최적 (76% 승률)                                          │
│     ✓ 1% ~ 1.5%: 좋음 (74% 승률)                                          │
│     ✗ < 0.5%: 진입 금지 (27% 승률)                                         │
│     △ > 3%: 안전하지만 힘 약함 (49% 승률)                                  │
│                                                                            │
│  □ 5. 손절가: L-value (L2) 이탈 시                                         │
│                                                                            │
│  □ 6. 목표가: 추세선(저항선) 도달 시                                       │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                              진입 결정                                     │
│                                                                            │
│  모든 조건 만족 → 진입                                                     │
│  Entry-L Gap < 1% → 진입 금지                                              │
│  L_change 데드존(-0.5%~0%) → 진입 주의                                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

print("\n분석 완료!")
