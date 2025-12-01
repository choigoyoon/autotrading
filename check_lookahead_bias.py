import pandas as pd
from datetime import timedelta

print("=" * 80)
print("🚨 Look-Ahead Bias 검증")
print("=" * 80)

df_4h = pd.read_csv('btc_4h_ohlcv.csv', parse_dates=['datetime'])
df_15m = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])

print("\n🔍 핵심 질문:")
print("=" * 80)
print("""
00:00 봉 데이터로 FVG 감지 후
00:00부터 즉시 진입 허용하면?

→ 이건 미래 정보 사용! (Look-Ahead Bias) ⚠️

왜냐하면:
  실전에서는 04:00에야 00:00 봉이 완성됨
  하지만 백테스트는 00:00부터 진입 허용
  
  → 4시간 빠른 진입! ⚠️
  → 백테스트 결과가 과대평가됨!
""")

print("\n" + "=" * 80)
print("📌 실제 백테스트 로직 분석")
print("=" * 80)

# FVG 감지 로직 재현
print("\n[STEP 1] FVG 시그널 감지:")
print("-" * 80)

signals = []
for i in range(2, min(20, len(df_4h))):
    current = df_4h.iloc[i]
    prev2 = df_4h.iloc[i-2]
    
    if current['low'] > prev2['high']:
        gap_size = (current['low'] - prev2['high']) / prev2['high'] * 100
        if gap_size >= 0.3:
            signal_time = current['datetime']
            entry_zone = current['low']
            
            signals.append({
                'i': i,
                'signal_time': signal_time,
                'entry_zone': entry_zone,
                'gap': gap_size
            })

if len(signals) > 0:
    first_sig = signals[0]
    print(f"\n첫 FVG 시그널:")
    print(f"  인덱스: i={first_sig['i']}")
    print(f"  시그널 시간: {first_sig['signal_time']}")
    print(f"  진입존: ${first_sig['entry_zone']:.1f}")

    print("\n[STEP 2] 진입 가능 시점:")
    print("-" * 80)
    
    sig_time = first_sig['signal_time']
    
    # 실제 로직: signal_time >= entry_time?
    print(f"\n코드 로직:")
    print(f"  entry_window = df_15m[df_15m['datetime'] >= signal_time]")
    print(f"                                        ^^")
    print(f"  → signal_time 이후부터 진입 가능")
    
    # 진입 가능한 첫 15M 봉
    entry_window = df_15m[df_15m['datetime'] >= sig_time]
    first_entry = entry_window.iloc[0] if len(entry_window) > 0 else None
    
    if first_entry is not None:
        print(f"\n첫 진입 가능 15M 봉:")
        print(f"  시간: {first_entry['datetime']}")
        print(f"  저가: ${first_entry['low']:.1f}")
        
        time_diff = (first_entry['datetime'] - sig_time).total_seconds() / 3600
        
        print(f"\n⚠️ 문제 발견:")
        print(f"  시그널 시간: {sig_time}")
        print(f"  첫 진입 시간: {first_entry['datetime']}")
        print(f"  시간차: {time_diff}시간")
        
        if time_diff == 0:
            print(f"\n  🚨 Look-Ahead Bias 발생! ⚠️")
            print(f"  → 시그널 발생과 동시에 진입 허용")
            print(f"  → 실전에서는 불가능!")
            print(f"\n  실전에서는:")
            print(f"    시그널: {sig_time} (실제로는 +4시간 후 확정)")
            real_confirm = sig_time + timedelta(hours=4)
            print(f"    실제 확정: {real_confirm}")
            print(f"    진입 가능: {real_confirm} 이후")
        else:
            print(f"\n  ✅ Look-Ahead Bias 없음")
            print(f"  → 시그널 발생 {time_diff}시간 후 진입")

print("\n" + "=" * 80)
print("🔍 정확한 백테스트 방법")
print("=" * 80)

print("""
[현재 로직] ❌
━━━━━━━━━━━━━━━━━━━━━━━━━━━
  signal_time = df_4h.iloc[i]['datetime']  # 00:00
  entry_allowed >= signal_time  # 00:00부터 진입 가능
  
  문제: 
    00:00 봉은 04:00에 확정되는데
    00:00부터 진입 허용 → 미래 정보 사용!

[올바른 로직] ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━
  signal_time = df_4h.iloc[i]['datetime']  # 00:00
  confirm_time = signal_time + timedelta(hours=4)  # 04:00
  entry_allowed >= confirm_time  # 04:00부터 진입 가능
  
  또는:
  signal_time = df_4h.iloc[i+1]['datetime']  # 04:00 (다음 봉)
  entry_allowed >= signal_time  # 04:00부터 진입 가능
""")

print("\n" + "=" * 80)
print("🎯 결론")
print("=" * 80)

print("""
현재 백테스트가 Look-Ahead Bias를 가지고 있다면:

1️⃣ 백테스트 성과가 과대평가됨 ⚠️
   - 실전보다 4시간 빠른 진입
   - 더 좋은 진입가 확보
   - 승률과 수익률 상승

2️⃣ 실전 적용 시 성과 하락 예상 📉
   - 진입 타이밍 4시간 지연
   - 진입가 불리해짐
   - 일부 시그널 놓칠 수 있음

3️⃣ 수정 필요! 🔧
   - signal_time에 4시간 추가
   - 또는 다음 봉 시작 시간 사용
   - 재백테스트 필요

정확한 확인이 필요합니다!
코드를 다시 검토해야 합니다! ⚠️
""")
