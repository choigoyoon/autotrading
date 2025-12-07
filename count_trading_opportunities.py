import pandas as pd

# CSV 읽기
df = pd.read_csv('hlhlhl_full_labeling_analysis.csv')

print(f"{'='*80}")
print(f"🎯 역추세 매매 기회 카운트")
print(f"{'='*80}\n")

total = len(df)

# 1단계: 저점 확인 (LL 패턴) - 이미 모두 LL만 분석함
step1_count = total
print(f"1️⃣ 저점 확인 (LL 패턴): {step1_count}번")

# 2단계: 추세선 돌파 (H3 돌파) - 모든 케이스가 돌파함
step2_count = df['h3_break'].sum()
step2_pct = step2_count / step1_count * 100
print(f"2️⃣ 추세선 돌파 (H3 돌파): {step2_count}번 ({step2_pct:.1f}%)")

# 3단계: 저항선 돌파
# H3 돌파
h3_break_count = df['h3_break'].sum()
h3_break_pct = h3_break_count / step1_count * 100

# H2 돌파
h2_break_count = df['h2_break'].sum()
h2_break_pct = h2_break_count / step1_count * 100

# H1 돌파
h1_break_count = df['h1_break'].sum()
h1_break_pct = h1_break_count / step1_count * 100

print(f"\n3️⃣ 저항선 돌파:")
print(f"   H3 돌파: {h3_break_count}번 ({h3_break_pct:.1f}%)")
print(f"   H2 돌파: {h2_break_count}번 ({h2_break_pct:.1f}%)")
print(f"   H1 돌파: {h1_break_count}번 ({h1_break_pct:.1f}%)")

# 매매 가능 기회
tradable = len(df[df['judgment'] == '매매 가능'])
tradable_pct = tradable / step1_count * 100

conditional = len(df[df['judgment'] == '조건부 가능'])
conditional_pct = conditional / step1_count * 100

not_tradable = len(df[df['judgment'] == '매매 불가'])
not_tradable_pct = not_tradable / step1_count * 100

print(f"\n{'='*80}")
print(f"💰 매매 가능 기회")
print(f"{'='*80}\n")

print(f"✅ 매매 가능 (5점 이상 + H3 돌파): {tradable}번 ({tradable_pct:.1f}%)")
print(f"⚠️ 조건부 가능 (3~4점): {conditional}번 ({conditional_pct:.1f}%)")
print(f"❌ 매매 불가 (3점 미만): {not_tradable}번 ({not_tradable_pct:.1f}%)")

print(f"\n{'='*80}")
print(f"📊 단계별 필터링")
print(f"{'='*80}\n")

print(f"전체 저점(LL): {step1_count}번")
print(f"   ↓ (필터: 추세선 존재)")
print(f"추세선 돌파: {step2_count}번 (-{step1_count - step2_count}번)")
print(f"   ↓ (필터: 힘 5점 이상)")
print(f"매매 가능: {tradable}번 (-{step2_count - tradable}번)")

success_funnel = tradable / step1_count * 100
print(f"\n🎯 전체 → 매매 가능 전환율: {success_funnel:.1f}%")

# 힘 점수별 분포
print(f"\n{'='*80}")
print(f"⚡ 힘 점수별 매매 가능 분포")
print(f"{'='*80}\n")

for score in sorted(df['power_score'].unique()):
    count = len(df[df['power_score'] == score])
    tradable_at_score = len(df[(df['power_score'] == score) & (df['judgment'] == '매매 가능')])
    h2_success = df[df['power_score'] == score]['h2_break'].sum()
    h2_rate = h2_success / count * 100 if count > 0 else 0
    print(f"{score:2d}점: {count:3d}번 | 매매가능 {tradable_at_score:3d}번 | H2돌파 {h2_success:3d}번 ({h2_rate:.1f}%)")

# 최종 요약
print(f"\n{'='*80}")
print(f"✅ 최종 요약")
print(f"{'='*80}\n")

print(f"📅 분석 기간: 2020-03-25 ~ 2025-11-24 (약 5.7년)")
print(f"\n단계별 발생 횟수:")
print(f"   1단계 (저점 확인): {step1_count}번")
print(f"   2단계 (추세선 돌파): {step2_count}번")
print(f"   3단계 (저항선 돌파):")
print(f"      - H3 돌파: {h3_break_count}번")
print(f"      - H2 돌파: {h2_break_count}번")
print(f"      - H1 돌파: {h1_break_count}번")

print(f"\n💰 실제 매매 가능 기회:")
print(f"   매매 가능: {tradable}번")
print(f"   조건부 가능: {conditional}번")
print(f"   합계: {tradable + conditional}번")

# 월평균 계산
years = 5.7
months = years * 12
print(f"\n📈 월평균 기회:")
print(f"   저점(LL) 발생: {step1_count / months:.1f}번/월")
print(f"   추세선 돌파: {step2_count / months:.1f}번/월")
print(f"   매매 가능: {tradable / months:.1f}번/월")
print(f"   매매 가능 + 조건부: {(tradable + conditional) / months:.1f}번/월")

print(f"\n🎯 성공률:")
print(f"   H2 도달률: {h2_break_pct:.1f}%")
print(f"   H1 도달률: {h1_break_pct:.1f}%")

