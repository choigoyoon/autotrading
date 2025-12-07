import pandas as pd
import numpy as np

df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

print("="*80)
print("🔍 승률 차이 분석: 81% vs 51% (30%p 차이)")
print("="*80)

# FVG 탐지
def detect_4h_fvg(df):
    fvgs = []
    for i in range(2, len(df)):
        prev2, prev1, curr = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
        if prev2['high'] < curr['low'] and curr['close'] > curr['open']:
            fvgs.append({
                'datetime': curr['datetime'],
                'fvg_top': curr['low'],
                'fvg_bottom': prev2['high'],
            })
    return fvgs

fvgs_4h = detect_4h_fvg(df_4h)

# 두 로직 비교
results = []

for fvg in fvgs_4h:
    fvg_time = fvg['datetime']
    fvg_top = fvg['fvg_top']
    
    mask = df_15m['datetime'] > fvg_time
    future_15m = df_15m[mask].head(200)
    
    if len(future_15m) < 50:
        continue
    
    for i, (idx, row) in enumerate(future_15m.iterrows()):
        if row['low'] <= fvg_top:
            remaining = future_15m.iloc[i+1:]
            if len(remaining) < 50:
                break
            
            entry_price = remaining.iloc[0]['open']
            
            tp_pct = 1.5
            sl_pct = -0.5
            
            # 기존 로직: max/min만 체크 (순서 무시)
            max_profit = (remaining.iloc[1:101]['high'].max() - entry_price) / entry_price * 100
            max_loss = (remaining.iloc[1:101]['low'].min() - entry_price) / entry_price * 100
            
            old_result = 'WIN' if max_profit >= tp_pct else ('LOSS' if max_loss <= sl_pct else 'TIMEOUT')
            
            # 수정 로직: 순서 체크
            new_result = None
            tp_bar = None
            sl_bar = None
            
            for j, (_, bar) in enumerate(remaining.iloc[1:101].iterrows()):
                high_pct = (bar['high'] - entry_price) / entry_price * 100
                low_pct = (bar['low'] - entry_price) / entry_price * 100
                open_pct = (bar['open'] - entry_price) / entry_price * 100
                
                if high_pct >= tp_pct and low_pct <= sl_pct:
                    new_result = 'WIN' if open_pct >= 0 else 'LOSS'
                    break
                elif high_pct >= tp_pct:
                    new_result = 'WIN'
                    break
                elif low_pct <= sl_pct:
                    new_result = 'LOSS'
                    break
            
            if new_result is None:
                new_result = 'TIMEOUT'
            
            results.append({
                'old_result': old_result,
                'new_result': new_result,
                'max_profit': max_profit,
                'max_loss': max_loss,
            })
            break

df = pd.DataFrame(results)
print(f"\n총 데이터: {len(df)}건")

# 결과 비교
print("\n" + "="*80)
print("📊 기존 vs 수정 로직 결과 비교")
print("="*80)

# 기존 로직 승률
old_wins = len(df[df['old_result'] == 'WIN'])
old_losses = len(df[df['old_result'] == 'LOSS'])
old_decided = old_wins + old_losses
old_wr = old_wins / old_decided * 100

# 수정 로직 승률
new_wins = len(df[df['new_result'] == 'WIN'])
new_losses = len(df[df['new_result'] == 'LOSS'])
new_decided = new_wins + new_losses
new_wr = new_wins / new_decided * 100

print(f"\n기존 로직: {old_wins}승 {old_losses}패 = {old_wr:.1f}%")
print(f"수정 로직: {new_wins}승 {new_losses}패 = {new_wr:.1f}%")
print(f"차이: {old_wr - new_wr:.1f}%p")

# 결과 변화 분석
print("\n" + "="*80)
print("🔍 결과 변화 상세")
print("="*80)

# 교차 분석
cross = pd.crosstab(df['old_result'], df['new_result'])
print("\n[교차표: 기존(행) vs 수정(열)]")
print(cross)

# WIN → LOSS 케이스 분석
win_to_loss = df[(df['old_result'] == 'WIN') & (df['new_result'] == 'LOSS')]
print(f"\n[WIN → LOSS 케이스: {len(win_to_loss)}건]")
print("  = TP도 도달하고 SL도 도달했지만, SL이 먼저 도달한 케이스")
print(f"  평균 max_profit: +{win_to_loss['max_profit'].mean():.2f}%")
print(f"  평균 max_loss: {win_to_loss['max_loss'].mean():.2f}%")

# LOSS → WIN 케이스 분석
loss_to_win = df[(df['old_result'] == 'LOSS') & (df['new_result'] == 'WIN')]
print(f"\n[LOSS → WIN 케이스: {len(loss_to_win)}건]")
if len(loss_to_win) > 0:
    print(f"  평균 max_profit: +{loss_to_win['max_profit'].mean():.2f}%")
    print(f"  평균 max_loss: {loss_to_win['max_loss'].mean():.2f}%")

# 변화 없는 케이스
same = df[df['old_result'] == df['new_result']]
print(f"\n[결과 동일 케이스: {len(same)}건 ({len(same)/len(df)*100:.1f}%)]")

print("\n" + "="*80)
print("💡 결론")
print("="*80)
print(f"""
승률 차이 원인:
- WIN → LOSS 변환: {len(win_to_loss)}건
- LOSS → WIN 변환: {len(loss_to_win)}건
- 순 차이: {len(win_to_loss) - len(loss_to_win)}건

기존 로직은 "TP 도달 여부"만 봄 (max_profit >= 1.5%)
→ SL이 먼저 도달해도 WIN으로 처리

수정 로직은 "먼저 도달한 쪽"으로 판정
→ SL이 먼저 도달하면 LOSS

{len(win_to_loss)}건이 실제로는 SL 먼저 도달 → LOSS
이게 {len(win_to_loss)/old_decided*100:.1f}%p 차이를 만듦
""")

