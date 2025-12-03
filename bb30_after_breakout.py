import pandas as pd
import numpy as np

# 1시간봉 데이터 로드
df = pd.read_csv('analysis_1h.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"1시간봉 데이터: {len(df)}개")

# BB 30 계산
BB_PERIOD = 30
df['bb_mid'] = df['close'].rolling(BB_PERIOD).mean()
df['bb_std'] = df['close'].rolling(BB_PERIOD).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100

# 수축 상태
df['bb_width_min30'] = df['bb_width'].rolling(30).min()
df['is_squeeze'] = df['bb_width'] <= df['bb_width_min30'] * 1.1

# 수축 → 확장 시점 찾기
squeeze_ends = []
in_squeeze = False
squeeze_start = 0

for i in range(60, len(df)):
    if pd.isna(df.iloc[i]['is_squeeze']):
        continue
    if df.iloc[i]['is_squeeze'] and not in_squeeze:
        in_squeeze = True
        squeeze_start = i
    elif not df.iloc[i]['is_squeeze'] and in_squeeze:
        in_squeeze = False
        squeeze_ends.append({
            'idx': i,
            'squeeze_length': i - squeeze_start
        })

print(f"돌파 시점: {len(squeeze_ends)}개")

# 돌파 후 움직임 분석
results = []

for sq in squeeze_ends:
    break_idx = sq['idx']
    
    if break_idx >= len(df) - 50:
        continue
    
    # 돌파 봉 정보
    break_candle = df.iloc[break_idx]
    break_dir = 1 if break_candle['close'] > break_candle['open'] else -1
    entry_price = break_candle['close']
    
    # 돌파 후 N봉 동안의 움직임 추적
    # "추세 확정" = 돌파 방향으로 연속 N봉
    
    # 1봉 후, 2봉 후, 3봉 후... 확정 시점별 분석
    for confirm_bars in [1, 2, 3]:
        # 확정 조건: 돌파 방향으로 연속 양봉/음봉
        confirmed = True
        for j in range(1, confirm_bars + 1):
            if break_idx + j >= len(df):
                confirmed = False
                break
            candle = df.iloc[break_idx + j]
            candle_dir = 1 if candle['close'] > candle['open'] else -1
            if candle_dir != break_dir:
                confirmed = False
                break
        
        if not confirmed:
            continue
        
        # 확정 시점 이후 진입
        entry_idx = break_idx + confirm_bars
        if entry_idx >= len(df) - 30:
            continue
        
        entry_price_after = df.iloc[entry_idx]['close']
        
        # 이후 30봉 동안 추적
        max_profit = 0
        max_loss = 0
        
        for k in range(1, 31):
            if entry_idx + k >= len(df):
                break
            
            price = df.iloc[entry_idx + k]['close']
            
            if break_dir == 1:  # LONG
                profit = (price - entry_price_after) / entry_price_after * 100
            else:  # SHORT
                profit = (entry_price_after - price) / entry_price_after * 100
            
            if profit > max_profit:
                max_profit = profit
            if profit < max_loss:
                max_loss = profit
        
        # SL 체크 (2%, 3%, 5% 기준)
        sl_2_hit = max_loss <= -2
        sl_3_hit = max_loss <= -3
        sl_5_hit = max_loss <= -5
        
        results.append({
            'datetime': df.iloc[entry_idx]['datetime'],
            'break_dir': break_dir,
            'confirm_bars': confirm_bars,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'sl_2_hit': sl_2_hit,
            'sl_3_hit': sl_3_hit,
            'sl_5_hit': sl_5_hit,
            'squeeze_length': sq['squeeze_length']
        })

res_df = pd.DataFrame(results)
print(f"\n분석 케이스: {len(res_df)}개")

print("\n" + "="*80)
print("📊 추세 확정 후 SL 확률")
print("="*80)

for confirm in [1, 2, 3]:
    sub = res_df[res_df['confirm_bars'] == confirm]
    print(f"\n### 돌파 후 {confirm}봉 연속 확인 후 진입")
    print(f"  케이스: {len(sub)}건")
    print(f"  평균 최대수익: +{sub['max_profit'].mean():.2f}%")
    print(f"  평균 최대손실: {sub['max_loss'].mean():.2f}%")
    print(f"  SL -2% 확률: {sub['sl_2_hit'].mean()*100:.1f}%")
    print(f"  SL -3% 확률: {sub['sl_3_hit'].mean()*100:.1f}%")
    print(f"  SL -5% 확률: {sub['sl_5_hit'].mean()*100:.1f}%")
    
    # 손익비
    avg_win = sub[sub['max_profit'] > 0]['max_profit'].mean()
    avg_loss = abs(sub['max_loss'].mean())
    print(f"  손익비 (최대수익/최대손실): {avg_win/avg_loss:.2f}")

print("\n" + "="*80)
print("📊 방향별 분석")
print("="*80)

for confirm in [2, 3]:
    sub = res_df[res_df['confirm_bars'] == confirm]
    print(f"\n### {confirm}봉 확인 후")
    for d in [1, -1]:
        dir_label = 'LONG' if d == 1 else 'SHORT'
        sub_dir = sub[sub['break_dir'] == d]
        if len(sub_dir) >= 20:
            print(f"  {dir_label}: {len(sub_dir)}건")
            print(f"    → 최대수익: +{sub_dir['max_profit'].mean():.2f}%, 최대손실: {sub_dir['max_loss'].mean():.2f}%")
            print(f"    → SL -2%: {sub_dir['sl_2_hit'].mean()*100:.1f}%, SL -3%: {sub_dir['sl_3_hit'].mean()*100:.1f}%")

print("\n" + "="*80)
print("📊 수축 길이별 (추세 확정 후)")
print("="*80)

sub = res_df[res_df['confirm_bars'] == 2]  # 2봉 확인 기준
for low, high, label in [(5, 10, '짧은수축(5-10h)'), (10, 20, '보통수축(10-20h)'), (20, 50, '긴수축(20h+)')]:
    sub_len = sub[(sub['squeeze_length'] >= low) & (sub['squeeze_length'] < high)]
    if len(sub_len) >= 20:
        print(f"\n  {label}: {len(sub_len)}건")
        print(f"    → 최대수익: +{sub_len['max_profit'].mean():.2f}%")
        print(f"    → 최대손실: {sub_len['max_loss'].mean():.2f}%")
        print(f"    → SL -2%: {sub_len['sl_2_hit'].mean()*100:.1f}%")
        print(f"    → SL -3%: {sub_len['sl_3_hit'].mean()*100:.1f}%")

# 실제 전략 시뮬레이션
print("\n" + "="*80)
print("💰 전략 시뮬레이션: 추세 확정 후 진입")
print("="*80)

for confirm in [2, 3]:
    sub = res_df[res_df['confirm_bars'] == confirm]
    
    print(f"\n### {confirm}봉 확인 후 진입, SL -3%, TP +5%")
    
    wins = 0
    losses = 0
    total_pnl = 0
    
    for _, row in sub.iterrows():
        if row['max_profit'] >= 5:  # TP 먼저 도달
            if row['max_loss'] > -3:  # SL 안맞고 TP
                wins += 1
                total_pnl += 5
            else:  # 뭐가 먼저인지 모름 - 보수적으로 SL
                losses += 1
                total_pnl -= 3
        elif row['max_loss'] <= -3:  # SL
            losses += 1
            total_pnl -= 3
        else:  # 둘 다 안닿음
            total_pnl += row['max_profit']  # 30봉 후 청산
            if row['max_profit'] > 0:
                wins += 1
            else:
                losses += 1
    
    total = wins + losses
    print(f"  승: {wins}, 패: {losses}, 승률: {wins/total*100:.1f}%")
    print(f"  총 수익: {total_pnl:.1f}% ({len(sub)}회 거래)")
    print(f"  회당 평균: {total_pnl/len(sub):.2f}%")

res_df.to_csv('bb30_after_breakout.csv', index=False)
print("\n\n결과 저장: bb30_after_breakout.csv")
