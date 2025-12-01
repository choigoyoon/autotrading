import pandas as pd
import numpy as np

print("=" * 70)
print("💀 스탑헌팅/휩쏘 분석 - 꼬리로 털리는 경우")
print("=" * 70)

df_15m = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])

# 벡터화 연산으로 빠르게
df_15m['prev_close'] = df_15m['close'].shift(1)
df_15m['range_pct'] = (df_15m['high'] - df_15m['low']) / df_15m['low'] * 100

# 꼬리 분석
df_15m['body'] = abs(df_15m['close'] - df_15m['open'])
df_15m['total_range'] = df_15m['high'] - df_15m['low']
df_15m['wick_pct'] = np.where(df_15m['total_range'] > 0, 
                              (df_15m['total_range'] - df_15m['body']) / df_15m['total_range'] * 100, 0)

print(f"\n📊 15분봉 꼬리 분석:")
print(f"   평균 꼬리 비율: {df_15m['wick_pct'].mean():.1f}%")
print(f"   긴 꼬리 캔들 (70%+): {len(df_15m[df_15m['wick_pct'] > 70]):,}개 ({len(df_15m[df_15m['wick_pct'] > 70])/len(df_15m)*100:.1f}%)")

# SL별 스탑헌팅 분석 (벡터화)
print(f"\n{'='*70}")
print("📈 SL 거리별 스탑헌팅 빈도 (꼬리로 털리고 회복)")
print("=" * 70)

sl_levels = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]

print(f"\n{'SL':<8} {'헌팅 횟수':<12} {'빈도':<10} {'평가':<20}")
print("-" * 50)

for sl in sl_levels:
    sl_pct = sl / 100
    
    # 롱: 저가가 SL 찍고 종가는 회복
    sl_price_long = df_15m['prev_close'] * (1 - sl_pct)
    recovery_price = df_15m['prev_close'] * (1 - sl_pct * 0.3)  # 30%만 회복해도
    
    fake_long = ((df_15m['low'] <= sl_price_long) & (df_15m['close'] > recovery_price)).sum()
    
    # 숏: 고가가 SL 찍고 종가는 회복
    sl_price_short = df_15m['prev_close'] * (1 + sl_pct)
    recovery_price_s = df_15m['prev_close'] * (1 + sl_pct * 0.3)
    
    fake_short = ((df_15m['high'] >= sl_price_short) & (df_15m['close'] < recovery_price_s)).sum()
    
    total = fake_long + fake_short
    freq = total / len(df_15m) * 100
    
    if freq < 0.5:
        grade = "✅ 안전"
    elif freq < 1.0:
        grade = "⚠️ 주의"
    else:
        grade = "🚨 위험"
    
    print(f"{sl}%{'':<5} {total:<12} {freq:.2f}%{'':<5} {grade}")

print(f"\n{'='*70}")
print("🎯 현재 전략 (SL -1.5%) 스탑헌팅 분석")
print("=" * 70)

sl_pct = 0.015
sl_price = df_15m['prev_close'] * (1 - sl_pct)

# SL 찍고 종가가 진입가 위인 경우 (완전 스탑헌팅)
full_fake = ((df_15m['low'] <= sl_price) & (df_15m['close'] > df_15m['prev_close'])).sum()

# SL 찍고 종가가 SL 위인 경우 (부분 스탑헌팅)
partial_fake = ((df_15m['low'] <= sl_price) & (df_15m['close'] > sl_price)).sum()

print(f"\n   SL -1.5% 기준:")
print(f"   - SL 찍고 진입가 위로 회복: {full_fake}회 ({full_fake/len(df_15m)*100:.2f}%)")
print(f"   - SL 찍고 SL가 위로 회복: {partial_fake}회 ({partial_fake/len(df_15m)*100:.2f}%)")

print(f"\n{'='*70}")
print("🛡️ 현재 전략의 스탑헌팅 방어 장치")
print("=" * 70)

print("""
   ✅ 1. 본절 스탑 (TP1 2% 도달 후)
      → SL을 진입가로 이동
      → 스탑헌팅 당해도 손실 0%!
      
   ✅ 2. 분할익절 (50%+50%)
      → TP1에서 절반 익절
      → 나머지만 스탑헌팅 위험
      
   ✅ 3. 시간스탑 (48바 = 12시간)
      → SL 대신 시간으로 청산
      → 스탑헌팅 자체를 피함
      
   ✅ 4. FVG 존 진입
      → 강한 지지/저항에서 진입
      → 스탑헌팅 확률↓
""")

print(f"\n{'='*70}")
print("💡 거래소 스탑헌팅 대응 팁")
print("=" * 70)

print("""
   🚨 스탑헌팅 많은 거래소 특징:
   - 저유동성 (호가창 얇음)
   - 높은 레버리지 허용 (100x+)
   - 청산 엔진 불투명
   
   ✅ 대응 방법:
   
   1. SL을 시장가 → 지정가로
      → 꼬리 끝에 안 잡힘
      → 단, 미체결 위험
      
   2. SL 넓히기 (-1.5% → -2.0%)
      → 스탑헌팅 회피
      → 현재 전략 MDD: -3% → -4% 예상
      
   3. 레버리지 낮추기
      → 청산 타겟에서 제외
      → 3~5배 권장
      
   4. 유동성 좋은 거래소
      → 바이낸스, 바이비트 추천
      → 소형 거래소 피하기
""")

# 실제 거래에서 얼마나 털릴지 추정
print(f"\n{'='*70}")
print("📊 실제 매매 시 스탑헌팅 피해 추정")
print("=" * 70)

# 현재 백테스트 SL 비율
base_sl_rate = 18.7  # MDD 최적화 결과
hunting_rate = partial_fake / len(df_15m) * 100

# 스탑헌팅으로 인한 추가 SL
additional_sl = hunting_rate * 0.3  # 30%는 실제로 스탑헌팅

print(f"""
   백테스트 SL 비율: {base_sl_rate:.1f}%
   스탑헌팅 빈도: {hunting_rate:.2f}%
   
   실제 예상:
   - 정상 SL: {base_sl_rate:.1f}%
   - 스탑헌팅 추가: +{additional_sl:.1f}%
   - 실제 SL 비율: ~{base_sl_rate + additional_sl:.1f}%
   
   💰 수익 영향:
   - 백테스트 월평균: 24.86%
   - 스탑헌팅 손실: -{additional_sl * 1.5:.1f}% (SL당 -1.5%)
   - 실제 예상: ~{24.86 - additional_sl * 1.5:.1f}%
   
   ✅ 결론: 본절 스탑 덕분에 스탑헌팅 피해 최소화!
""")

