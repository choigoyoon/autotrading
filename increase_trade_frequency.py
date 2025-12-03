"""
BB 수축/확장 전략 - 매매 횟수 증가 방안
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("BB 수축/확장 전략 - 매매 횟수 증가 분석")
print("=" * 80)
print()

# 데이터 로드
df = pd.read_csv('expanded_squeeze_analysis.csv')

print(f"총 BB 수축→확장 이벤트: {len(df)}개 (5년)")
print(f"연평균: {len(df)/5:.0f}개")
print()

# ═══════════════════════════════════════════════════════════════════
# 현재 전략들의 필터 강도 분석
# ═══════════════════════════════════════════════════════════════════

print("=" * 80)
print("1. 현재 전략 필터 강도")
print("=" * 80)
print()

strategies = [
    ('M200≥20%', (df['momentum_200'] >= 20)),
    ('M200≥15%', (df['momentum_200'] >= 15)),
    ('M200≥10%', (df['momentum_200'] >= 10)),
    ('M200≥5%', (df['momentum_200'] >= 5)),
    ('M200≥0%', (df['momentum_200'] >= 0)),
    ('M100≥20%', (df['momentum_100'] >= 20)),
    ('M100≥15%', (df['momentum_100'] >= 15)),
    ('M100≥10%', (df['momentum_100'] >= 10)),
    ('M100≥5%', (df['momentum_100'] >= 5)),
    ('M100≥0%', (df['momentum_100'] >= 0)),
]

for name, cond in strategies:
    count = cond.sum()
    annual = count / 5
    pct = count / len(df) * 100
    
    # 간단한 성과 (336h 기준)
    subset = df[cond]
    if len(subset) > 0:
        wins = (subset['long_336h'] > 0).sum()
        wr = wins / len(subset) * 100
        avg = subset['long_336h'].mean()
        
        print(f"{name:20s}: {count:4d}개 (연 {annual:4.0f}) | 승률 {wr:5.1f}% | 평균 {avg:+6.2f}%")

print()

# ═══════════════════════════════════════════════════════════════════
# 방안 1: 모멘텀 기준 완화
# ═══════════════════════════════════════════════════════════════════

print("=" * 80)
print("2. 방안 1: 모멘텀 기준 완화")
print("=" * 80)
print()

relaxed_strategies = [
    ('M200≥5% + EMA정배열', (df['momentum_200'] >= 5) & (df['ema_bull'] == True)),
    ('M200≥0% + EMA정배열', (df['momentum_200'] >= 0) & (df['ema_bull'] == True)),
    ('M100≥5% + HH', (df['momentum_100'] >= 5) & (df['HH'] == True)),
    ('M100≥0% + HH', (df['momentum_100'] >= 0) & (df['HH'] == True)),
    ('M200≥5% + M100≥5%', (df['momentum_200'] >= 5) & (df['momentum_100'] >= 5)),
]

print("완화된 조건 성과:")
for name, cond in relaxed_strategies:
    subset = df[cond]
    if len(subset) > 0:
        count = len(subset)
        annual = count / 5
        wins = (subset['long_336h'] > 0).sum()
        wr = wins / len(subset) * 100
        avg = subset['long_336h'].mean()
        total = subset['long_336h'].sum()
        
        print(f"{name:25s}: {count:4d}개 (연 {annual:4.0f}) | 승률 {wr:5.1f}% | 평균 {avg:+6.2f}% | 총 {total:+7.1f}%")

print()

# ═══════════════════════════════════════════════════════════════════
# 방안 2: 짧은 홀딩 기간 (빠른 회전)
# ═══════════════════════════════════════════════════════════════════

print("=" * 80)
print("3. 방안 2: 짧은 홀딩 기간")
print("=" * 80)
print()

print("홀딩 기간별 성과 (M200≥10% + 정배열):")
base_cond = (df['momentum_200'] >= 10) & (df['ema_bull'] == True)
base_subset = df[base_cond]

holding_periods = [
    ('24h', 'long_24h'),
    ('48h', 'long_48h'),
    ('72h', 'long_72h'),
    ('168h', 'long_168h'),
    ('336h', 'long_336h'),
]

for period_name, col in holding_periods:
    if col in base_subset.columns:
        wins = (base_subset[col] > 0).sum()
        wr = wins / len(base_subset) * 100
        avg = base_subset[col].mean()
        
        print(f"  {period_name:6s}: 승률 {wr:5.1f}% | 평균 {avg:+6.2f}%")

print()
print("💡 짧은 홀딩(24~72h)은 승률 높지만 평균 수익 낮음")
print("   → 거래 횟수는 동일하지만 회전율 증가")
print()

# ═══════════════════════════════════════════════════════════════════
# 방안 3: 다중 타임프레임 동시 운영
# ═══════════════════════════════════════════════════════════════════

print("=" * 80)
print("4. 방안 3: 다중 타임프레임 병행")
print("=" * 80)
print()

# 1시간봉 최대
print("1시간봉 (현재):")
print("  M200≥10% + 정배열 + HH: 연 17회")
print()

# 15분봉 가능성
print("💡 15분봉 BB 수축/확장 도입:")
btc_15m = pd.read_csv('btc_15m_ohlcv.csv')
print(f"  15분봉 데이터: {len(btc_15m):,}개")
print(f"  예상 BB 이벤트: 1시간봉의 4배 = 연 ~3,200개")
print(f"  필터 후 예상: 10% 선별 시 연 ~320회")
print()

# 4시간봉 추가
print("4시간봉 추가:")
print("  M50≥15% + 정배열: 연 3회")
print()

print("📊 다중 TF 통합 시:")
print("  1시간봉: 17회")
print("  15분봉: 50~100회 (예상)")
print("  4시간봉: 3회")
print("  ────────────────")
print("  합계: 70~120회/년")
print()

# ═══════════════════════════════════════════════════════════════════
# 방안 4: 다른 엔트리 신호 병행
# ═══════════════════════════════════════════════════════════════════

print("=" * 80)
print("5. 방안 4: 다른 엔트리 신호 추가")
print("=" * 80)
print()

print("BB 수축/확장 외 추가 가능 신호:")
print("  • EMA 크로스 (골든/데스 크로스)")
print("  • 지지/저항선 돌파")
print("  • RSI 과매도/과매수")
print("  • 볼륨 급증")
print("  • MACD 크로스")
print()

# ═══════════════════════════════════════════════════════════════════
# 최종 권장 사항
# ═══════════════════════════════════════════════════════════════════

print("=" * 80)
print("최종 권장 방안")
print("=" * 80)
print()

print("🎯 즉시 적용 가능 (기존 프레임워크 활용):")
print()
print("1. 모멘텀 기준 완화")
print("   현재: M200≥10~20%")
print("   변경: M200≥5% 또는 M200≥0%")
print("   효과: 연 17회 → 50~100회")
print()

print("2. 15분봉 BB 전략 추가")
print("   1시간봉과 동일 로직, 더 짧은 홀딩")
print("   효과: 추가 50~100회/년")
print()

print("3. 복합 필터 최적화")
print("   M200≥5% + EMA정배열")
print("   M100≥5% + HH")
print("   효과: 승률 유지하며 횟수 증가")
print()

print("🔬 중장기 개발:")
print()
print("4. 새로운 엔트리 신호 개발")
print("   EMA 크로스, 지지/저항 돌파 등")
print("   효과: 완전히 다른 기회 포착")
print()

print("=" * 80)
print("✅ 분석 완료")
print("=" * 80)
