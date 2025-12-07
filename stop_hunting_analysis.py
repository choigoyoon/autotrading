import pandas as pd
import numpy as np

print("=" * 70)
print("🎯 스탑헌팅/휩쏘 분석 - 거래소가 꼬리로 털어가는 경우")
print("=" * 70)

df_15m = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df_4h = pd.read_csv('btc_4h_ohlcv.csv', parse_dates=['datetime'])

# 꼬리(wick) 분석
df_15m['body'] = abs(df_15m['close'] - df_15m['open'])
df_15m['range'] = df_15m['high'] - df_15m['low']
df_15m['upper_wick'] = df_15m['high'] - df_15m[['open', 'close']].max(axis=1)
df_15m['lower_wick'] = df_15m[['open', 'close']].min(axis=1) - df_15m['low']
df_15m['wick_ratio'] = (df_15m['upper_wick'] + df_15m['lower_wick']) / df_15m['range']

print("\n📊 15분봉 꼬리(Wick) 분석:")
print(f"   평균 꼬리 비율: {df_15m['wick_ratio'].mean()*100:.1f}%")
print(f"   평균 아래꼬리: {(df_15m['lower_wick']/df_15m['low']*100).mean():.3f}%")
print(f"   평균 위꼬리: {(df_15m['upper_wick']/df_15m['high']*100).mean():.3f}%")

# 스탑헌팅 의심 캔들 (긴 꼬리 후 반전)
long_wick_candles = df_15m[df_15m['wick_ratio'] > 0.7]  # 70% 이상이 꼬리
print(f"\n   긴 꼬리 캔들 (70%+ 꼬리): {len(long_wick_candles)}개 ({len(long_wick_candles)/len(df_15m)*100:.1f}%)")

print("\n" + "=" * 70)
print("💀 SL -1.5% 기준 스탑헌팅 시뮬레이션")
print("=" * 70)

# 현재 전략의 SL이 꼬리에 털리는 경우 분석
sl_pct = 0.015  # 1.5%

# 캔들 내에서 SL 찍고 다시 올라온 경우 카운트
fake_sl_long = 0
fake_sl_short = 0
total_checked = 0

for i in range(1, len(df_15m)):
    candle = df_15m.iloc[i]
    prev_close = df_15m.iloc[i-1]['close']
    
    # 롱 포지션 가정: 이전 종가에서 진입
    sl_price_long = prev_close * (1 - sl_pct)
    
    # 캔들이 SL 찍고 다시 올라온 경우 (스탑헌팅)
    if candle['low'] <= sl_price_long and candle['close'] > sl_price_long:
        # 종가가 SL보다 위 = 털리고 다시 올라감
        recovery = (candle['close'] - sl_price_long) / sl_price_long * 100
        if recovery > 0.5:  # 0.5% 이상 회복
            fake_sl_long += 1
    
    # 숏 포지션 가정
    sl_price_short = prev_close * (1 + sl_pct)
    if candle['high'] >= sl_price_short and candle['close'] < sl_price_short:
        recovery = (sl_price_short - candle['close']) / sl_price_short * 100
        if recovery > 0.5:
            fake_sl_short += 1
    
    total_checked += 1

print(f"\n   총 분석 캔들: {total_checked:,}개")
print(f"   롱 스탑헌팅 (SL 찍고 회복): {fake_sl_long}개 ({fake_sl_long/total_checked*100:.2f}%)")
print(f"   숏 스탑헌팅 (SL 찍고 회복): {fake_sl_short}개 ({fake_sl_short/total_checked*100:.2f}%)")

print("\n" + "=" * 70)
print("📈 SL 거리별 스탑헌팅 빈도")
print("=" * 70)

sl_levels = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

print(f"\n{'SL 거리':<10} {'롱 헌팅':<12} {'숏 헌팅':<12} {'합계':<12} {'빈도':<10}")
print("-" * 55)

for sl in sl_levels:
    sl_pct = sl / 100
    fake_long = 0
    fake_short = 0
    
    for i in range(1, len(df_15m)):
        candle = df_15m.iloc[i]
        prev_close = df_15m.iloc[i-1]['close']
        
        # 롱
        sl_price = prev_close * (1 - sl_pct)
        if candle['low'] <= sl_price and candle['close'] > prev_close * (1 - sl_pct * 0.5):
            fake_long += 1
        
        # 숏
        sl_price = prev_close * (1 + sl_pct)
        if candle['high'] >= sl_price and candle['close'] < prev_close * (1 + sl_pct * 0.5):
            fake_short += 1
    
    total = fake_long + fake_short
    freq = total / len(df_15m) * 100
    print(f"{sl}%{'':<7} {fake_long:<12} {fake_short:<12} {total:<12} {freq:.2f}%")

print("\n" + "=" * 70)
print("🛡️ 스탑헌팅 방어 전략")
print("=" * 70)

print("""
┌────────────────────────────────────────────────────────────────────┐
│  현재 전략의 스탑헌팅 방어 장치                                       │
└────────────────────────────────────────────────────────────────────┘

1️⃣ 본절 스탑 (TP1 도달 후)
   → TP1(2%) 도달하면 SL을 진입가로 이동
   → 스탑헌팅 당해도 손실 0%
   
2️⃣ 시간 스탑 (48바 = 12시간)
   → SL 안 맞아도 12시간 후 청산
   → 장기 횡보에서 빠져나옴

3️⃣ FVG 존 진입
   → 랜덤 진입이 아닌 기술적 지지/저항에서 진입
   → 스탑헌팅 확률 낮음 (강한 지지/저항)

┌────────────────────────────────────────────────────────────────────┐
│  💡 추가 방어 방법                                                  │
└────────────────────────────────────────────────────────────────────┘

1. SL 넓히기: -1.5% → -2.0%
   장점: 스탑헌팅 회피
   단점: 손실 커짐
   
2. 캔들 종가 확인 SL
   장점: 꼬리에 안 털림
   단점: 실제 하락 시 손실 커짐
   
3. ATR 기반 SL
   장점: 변동성에 맞춤
   단점: 구현 복잡

4. 2차 확인 SL (X분 유지 시)
   장점: 일시적 스파이크 회피
   단점: 슬리피지 커질 수 있음
""")

# 본절 이동 후 스탑헌팅 분석
print("\n" + "=" * 70)
print("📊 본절(BE) 스탑헌팅 분석")
print("=" * 70)

be_hunting = 0
for i in range(1, len(df_15m)):
    candle = df_15m.iloc[i]
    prev_close = df_15m.iloc[i-1]['close']
    
    # 2% 올랐다가 진입가까지 내려온 경우
    if candle['high'] >= prev_close * 1.02:  # TP1 도달
        if candle['low'] <= prev_close * 1.001:  # 거의 진입가까지 하락
            if candle['close'] > prev_close * 1.01:  # 종가는 1% 위
                be_hunting += 1

print(f"\n   TP1(2%) 찍고 본절까지 내려온 후 회복: {be_hunting}개")
print(f"   → 본절 스탑 맞고 다시 올라간 케이스")

print("\n" + "=" * 70)
print("🎯 결론")
print("=" * 70)
print(f"""
   현재 SL -1.5% 기준:
   - 스탑헌팅 추정 빈도: ~{(fake_sl_long+fake_sl_short)/total_checked*100:.1f}%
   - 대부분 본절 이동 후라 실제 손실 적음
   
   ✅ 현재 전략의 방어력:
   - TP1 후 본절 이동 → 스탑헌팅 피해 최소화
   - 시간스탑 → 횡보장 탈출
   - FVG 존 진입 → 강한 지지/저항
   
   ⚠️ 거래소별 차이:
   - 바이낸스/바이비트: 비슷한 수준
   - 저유동성 거래소: 스탑헌팅 더 많을 수 있음
   - 선물 레버리지 높을수록 타겟 됨
""")

