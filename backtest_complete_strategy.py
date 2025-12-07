import pandas as pd
import numpy as np

print(f"{'='*80}")
print(f"🎯 역추세 전략 완전 백테스트")
print(f"{'='*80}\n")

# CSV 읽기
df = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df = df.rename(columns={'datetime': 'timestamp'})
df = df.sort_values('timestamp').reset_index(drop=True)

# BB 계산
def calculate_bb(df, window=20, num_std=2):
    df['bb_middle'] = df['close'].rolling(window=window).mean()
    df['bb_std'] = df['close'].rolling(window=window).std()
    df['bb_upper'] = df['bb_middle'] + (num_std * df['bb_std'])
    df['bb_lower'] = df['bb_middle'] - (num_std * df['bb_std'])
    return df

df = calculate_bb(df)

# Swing High/Low 찾기
def find_swing_highs(df, window=10):
    highs = []
    for i in range(window, len(df) - window):
        if df.loc[i, 'high'] == df.loc[i-window:i+window+1, 'high'].max():
            highs.append({
                'index': i,
                'time': df.loc[i, 'timestamp'],
                'price': df.loc[i, 'high'],
                'type': 'H'
            })
    return highs

def find_swing_lows(df, window=10):
    lows = []
    for i in range(window, len(df) - window):
        if df.loc[i, 'low'] == df.loc[i-window:i+window+1, 'low'].min():
            lows.append({
                'index': i,
                'time': df.loc[i, 'timestamp'],
                'price': df.loc[i, 'low'],
                'type': 'L'
            })
    return lows

highs = find_swing_highs(df)
lows = find_swing_lows(df)

# HLHLHL 라벨링
all_points = highs + lows
all_points = sorted(all_points, key=lambda x: x['index'])

labeled_points = []
for i, point in enumerate(all_points):
    if i == 0:
        labeled_points.append({**point, 'pattern': 'start'})
        continue
    
    prev_same_type = None
    for j in range(i-1, -1, -1):
        if all_points[j]['type'] == point['type']:
            prev_same_type = all_points[j]
            break
    
    if prev_same_type:
        if point['type'] == 'L':
            if point['price'] < prev_same_type['price']:
                pattern = 'LL'
            else:
                pattern = 'HL'
        else:
            if point['price'] < prev_same_type['price']:
                pattern = 'LH'
            else:
                pattern = 'HH'
    else:
        pattern = 'first'
    
    labeled_points.append({**point, 'pattern': pattern})

print(f"📊 데이터 준비 완료")
print(f"   총 캔들: {len(df):,}개")
print(f"   HL 포인트: {len(labeled_points):,}개\n")

# 백테스트 실행
trades = []

for i in range(len(labeled_points)):
    point = labeled_points[i]
    
    # LL 패턴만
    if point['type'] != 'L' or point['pattern'] != 'LL':
        continue
    
    point_idx = point['index']
    point_price = point['price']
    
    # 이전 3개 H값으로 추세선
    prev_highs = [p for p in labeled_points[:i] if p['type'] == 'H'][-3:]
    if len(prev_highs) < 3:
        continue
    
    # 하락 추세선 확인
    if not (prev_highs[0]['price'] > prev_highs[1]['price'] > prev_highs[2]['price']):
        continue
    
    h1, h2, h3 = prev_highs[0], prev_highs[1], prev_highs[2]
    
    # 추세선 돌파 찾기
    breakout_idx = None
    for idx in range(point_idx + 1, min(point_idx + 100, len(df))):
        if df.loc[idx, 'close'] > h3['price']:
            breakout_idx = idx
            break
    
    if not breakout_idx:
        continue
    
    # 돌파 힘 확인
    tb_idx = breakout_idx
    power_score = 0
    
    # BB 상단 찢기
    if not pd.isna(df.loc[tb_idx, 'bb_upper']):
        if df.loc[tb_idx, 'high'] > df.loc[tb_idx, 'bb_upper']:
            power_score += 3
    
    # FVG
    if tb_idx >= 2:
        if df.loc[tb_idx-2, 'high'] < df.loc[tb_idx, 'low']:
            power_score += 2
    
    # OB
    body_pct = (df.loc[tb_idx, 'close'] / df.loc[tb_idx, 'open'] - 1) * 100
    if body_pct > 1:
        power_score += 2
    
    # 강양봉
    if body_pct > 2:
        power_score += 3
    
    # 5점 미만이면 스킵
    if power_score < 5:
        continue
    
    # 리테스트 확인
    retest_confirmed = False
    retest_idx = None
    
    for idx in range(breakout_idx + 1, min(breakout_idx + 51, len(df))):
        if abs(df.loc[idx, 'low'] - h3['price']) / h3['price'] < 0.01:
            next_5 = df.loc[idx+1:idx+6]
            if len(next_5) >= 5 and next_5['close'].mean() > h3['price']:
                retest_confirmed = True
                retest_idx = idx
                break
    
    if not retest_confirmed:
        continue
    
    # 진입
    entry_idx = retest_idx + 1
    entry_price = df.loc[entry_idx, 'open']
    entry_time = df.loc[entry_idx, 'timestamp']
    
    # SL/TP 설정
    sl_price = h3['price'] * 0.995
    tp1_price = h2['price']
    tp2_price = h1['price']
    
    # 추적
    position_size = 100  # 초기 포지션 100%
    total_pnl = 0
    exit_reason = 'Open'
    
    for idx in range(entry_idx, min(entry_idx + 200, len(df))):
        high = df.loc[idx, 'high']
        low = df.loc[idx, 'low']
        
        # SL 체크
        if low < sl_price and position_size > 0:
            pnl = (sl_price / entry_price - 1) * 100 * (position_size / 100)
            total_pnl += pnl
            exit_reason = 'SL'
            position_size = 0
            break
        
        # TP1 체크
        if high >= tp1_price and position_size == 100:
            # 50% 익절
            pnl = (tp1_price / entry_price - 1) * 100 * 0.5
            total_pnl += pnl
            position_size = 50
            sl_price = entry_price  # SL을 본전으로 이동
            exit_reason = 'TP1_Partial'
        
        # TP2 체크
        if high >= tp2_price and position_size == 50:
            # 나머지 50% 익절
            pnl = (tp2_price / entry_price - 1) * 100 * 0.5
            total_pnl += pnl
            exit_reason = 'TP2_Full'
            position_size = 0
            break
        
        # 본전 SL 체크 (TP1 후)
        if position_size == 50 and low < sl_price:
            # 나머지 50%는 본전 청산
            exit_reason = 'TP1_Breakeven'
            position_size = 0
            break
    
    # R:R 계산
    risk = abs(entry_price - (h3['price'] * 0.995))
    reward_tp1 = abs(tp1_price - entry_price)
    reward_tp2 = abs(tp2_price - entry_price)
    
    rr_tp1 = reward_tp1 / risk if risk > 0 else 0
    rr_tp2 = reward_tp2 / risk if risk > 0 else 0
    
    trades.append({
        'entry_time': entry_time,
        'entry_price': entry_price,
        'h1_price': h1['price'],
        'h2_price': h2['price'],
        'h3_price': h3['price'],
        'sl_price': h3['price'] * 0.995,
        'tp1_price': tp1_price,
        'tp2_price': tp2_price,
        'power_score': power_score,
        'exit_reason': exit_reason,
        'pnl_pct': total_pnl,
        'rr_tp1': rr_tp1,
        'rr_tp2': rr_tp2
    })

print(f"✅ 백테스트 완료!\n")

# DataFrame 생성
df_trades = pd.DataFrame(trades)

if len(df_trades) == 0:
    print("❌ 거래 없음!")
    exit()

# 통계
print(f"{'='*80}")
print(f"📊 백테스트 결과")
print(f"{'='*80}\n")

total_trades = len(df_trades)
print(f"총 거래 수: {total_trades}개\n")

# 결과별 분포
print(f"📈 청산 사유별 분포:")
exit_counts = df_trades['exit_reason'].value_counts()
for reason, count in exit_counts.items():
    pct = count / total_trades * 100
    avg_pnl = df_trades[df_trades['exit_reason'] == reason]['pnl_pct'].mean()
    print(f"   {reason:20s}: {count:3d}개 ({pct:5.1f}%) | 평균 수익: {avg_pnl:+.2f}%")

# 승률 계산
wins = len(df_trades[df_trades['pnl_pct'] > 0])
losses = len(df_trades[df_trades['pnl_pct'] < 0])
breakeven = len(df_trades[df_trades['pnl_pct'] == 0])

win_rate = wins / total_trades * 100

print(f"\n💰 손익 통계:")
print(f"   승리: {wins}개 ({wins/total_trades*100:.1f}%)")
print(f"   손실: {losses}개 ({losses/total_trades*100:.1f}%)")
print(f"   본전: {breakeven}개 ({breakeven/total_trades*100:.1f}%)")
print(f"   승률: {win_rate:.1f}%")

# 수익률
total_pnl = df_trades['pnl_pct'].sum()
avg_pnl = df_trades['pnl_pct'].mean()
avg_win = df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].mean() if wins > 0 else 0
avg_loss = df_trades[df_trades['pnl_pct'] < 0]['pnl_pct'].mean() if losses > 0 else 0

print(f"\n📊 수익률:")
print(f"   총 수익: {total_pnl:+.2f}%")
print(f"   평균 수익: {avg_pnl:+.2f}%")
print(f"   평균 승리: {avg_win:+.2f}%")
print(f"   평균 손실: {avg_loss:+.2f}%")

# R:R
avg_rr_tp1 = df_trades['rr_tp1'].mean()
avg_rr_tp2 = df_trades['rr_tp2'].mean()

print(f"\n📏 평균 R:R 비율:")
print(f"   TP1 R:R: 1:{avg_rr_tp1:.2f}")
print(f"   TP2 R:R: 1:{avg_rr_tp2:.2f}")

# 기간별 통계
df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
df_trades['year'] = df_trades['entry_time'].dt.year

print(f"\n📅 연도별 통계:")
for year in sorted(df_trades['year'].unique()):
    year_trades = df_trades[df_trades['year'] == year]
    year_count = len(year_trades)
    year_pnl = year_trades['pnl_pct'].sum()
    year_avg = year_trades['pnl_pct'].mean()
    year_wins = len(year_trades[year_trades['pnl_pct'] > 0])
    year_winrate = year_wins / year_count * 100
    print(f"   {year}: {year_count:2d}개 | 총수익 {year_pnl:+6.2f}% | 평균 {year_avg:+.2f}% | 승률 {year_winrate:.1f}%")

# Top 10 수익 거래
print(f"\n🔥 Top 10 수익 거래:")
top_trades = df_trades.nlargest(10, 'pnl_pct')
for idx, row in top_trades.iterrows():
    print(f"\n{row['entry_time']}")
    print(f"   진입: ${row['entry_price']:,.0f}")
    print(f"   H3: ${row['h3_price']:,.0f} | H2: ${row['h2_price']:,.0f} | H1: ${row['h1_price']:,.0f}")
    print(f"   힘: {row['power_score']}점 | 청산: {row['exit_reason']}")
    print(f"   수익: {row['pnl_pct']:+.2f}%")

# CSV 저장
df_trades.to_csv('backtest_complete_results.csv', index=False)
print(f"\n✅ 결과 저장: backtest_complete_results.csv")

# 최종 요약
print(f"\n{'='*80}")
print(f"✅ 최종 요약")
print(f"{'='*80}\n")

print(f"📅 기간: 2020-03-25 ~ 2025-11-24 (5.7년)")
print(f"💰 총 거래: {total_trades}개")
print(f"📈 승률: {win_rate:.1f}%")
print(f"💵 총 수익: {total_pnl:+.2f}%")
print(f"📊 평균 수익: {avg_pnl:+.2f}%")
print(f"📏 평균 R:R: 1:{avg_rr_tp2:.2f}")

# 월평균
months = 5.7 * 12
print(f"\n📈 월평균:")
print(f"   거래 횟수: {total_trades / months:.1f}번/월")
print(f"   월 수익: {total_pnl / months:+.2f}%/월")

# 전략 효과
print(f"\n💡 전략 효과:")
tp1_partial = len(df_trades[df_trades['exit_reason'] == 'TP1_Partial'])
tp2_full = len(df_trades[df_trades['exit_reason'] == 'TP2_Full'])
tp1_breakeven = len(df_trades[df_trades['exit_reason'] == 'TP1_Breakeven'])
sl_hit = len(df_trades[df_trades['exit_reason'] == 'SL'])

print(f"   TP1 후 본전 확보: {tp1_partial + tp1_breakeven}번 ({(tp1_partial + tp1_breakeven)/total_trades*100:.1f}%)")
print(f"   TP2 완전 달성: {tp2_full}번 ({tp2_full/total_trades*100:.1f}%)")
print(f"   초기 SL: {sl_hit}번 ({sl_hit/total_trades*100:.1f}%)")

