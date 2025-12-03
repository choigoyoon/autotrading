import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv('analysis_1h.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

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

# 수축 → 확장 시점 찾기 + 3봉 확인
trade_signals = []
in_squeeze = False
squeeze_start = 0

for i in range(60, len(df) - 35):
    if pd.isna(df.iloc[i]['is_squeeze']):
        continue
    if df.iloc[i]['is_squeeze'] and not in_squeeze:
        in_squeeze = True
        squeeze_start = i
    elif not df.iloc[i]['is_squeeze'] and in_squeeze:
        in_squeeze = False
        break_idx = i
        
        # 돌파 방향
        break_candle = df.iloc[break_idx]
        break_dir = 1 if break_candle['close'] > break_candle['open'] else -1
        
        # 3봉 연속 확인
        confirmed = True
        for j in range(1, 4):
            candle = df.iloc[break_idx + j]
            candle_dir = 1 if candle['close'] > candle['open'] else -1
            if candle_dir != break_dir:
                confirmed = False
                break
        
        if confirmed:
            entry_idx = break_idx + 3
            entry_price = df.iloc[entry_idx]['close']
            
            # 30봉 후 결과
            max_profit = 0
            max_loss = 0
            exit_idx = entry_idx + 30
            
            for k in range(1, 31):
                price = df.iloc[entry_idx + k]['close']
                if break_dir == 1:
                    profit = (price - entry_price) / entry_price * 100
                else:
                    profit = (entry_price - price) / entry_price * 100
                if profit > max_profit:
                    max_profit = profit
                if profit < max_loss:
                    max_loss = profit
            
            trade_signals.append({
                'squeeze_start': squeeze_start,
                'break_idx': break_idx,
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'direction': break_dir,
                'entry_price': entry_price,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'win': max_profit > abs(max_loss)
            })

print(f"총 매매 신호: {len(trade_signals)}개")

# 최근 성공/실패 케이스 각각 선택
wins = [t for t in trade_signals if t['win']]
losses = [t for t in trade_signals if not t['win']]

# 좋은 예시 3개 선택 (최근 것들)
examples = []
if len(wins) >= 2:
    examples.extend(wins[-2:])  # 최근 성공 2개
if len(losses) >= 1:
    examples.append(losses[-1])  # 최근 실패 1개

# 시간순 정렬
examples = sorted(examples, key=lambda x: x['entry_idx'])[-3:]

# 차트 그리기
fig, axes = plt.subplots(len(examples), 1, figsize=(14, 5*len(examples)))
if len(examples) == 1:
    axes = [axes]

for idx, (ax, trade) in enumerate(zip(axes, examples)):
    # 범위 설정 (수축 시작 10봉 전 ~ 진입 후 35봉)
    start = max(0, trade['squeeze_start'] - 10)
    end = min(len(df), trade['exit_idx'] + 5)
    
    plot_df = df.iloc[start:end].copy()
    plot_df = plot_df.reset_index(drop=True)
    
    # 인덱스 조정
    sq_start_plot = trade['squeeze_start'] - start
    break_plot = trade['break_idx'] - start
    entry_plot = trade['entry_idx'] - start
    exit_plot = trade['exit_idx'] - start
    
    # 캔들 차트
    x = range(len(plot_df))
    
    # BB 밴드
    ax.fill_between(x, plot_df['bb_upper'], plot_df['bb_lower'], alpha=0.2, color='blue', label='BB Band')
    ax.plot(x, plot_df['bb_mid'], 'b--', alpha=0.5, linewidth=1, label='BB Mid')
    ax.plot(x, plot_df['bb_upper'], 'b-', alpha=0.3, linewidth=1)
    ax.plot(x, plot_df['bb_lower'], 'b-', alpha=0.3, linewidth=1)
    
    # 캔들
    for i in range(len(plot_df)):
        row = plot_df.iloc[i]
        color = 'green' if row['close'] >= row['open'] else 'red'
        
        # 몸통
        ax.bar(i, abs(row['close'] - row['open']), 
               bottom=min(row['open'], row['close']),
               color=color, width=0.6, edgecolor=color)
        # 꼬리
        ax.vlines(i, row['low'], row['high'], color=color, linewidth=1)
    
    # 수축 구간 표시
    ax.axvspan(sq_start_plot, break_plot, alpha=0.1, color='yellow', label='Squeeze')
    
    # 돌파 시점
    ax.axvline(break_plot, color='purple', linestyle='--', linewidth=2, label='Breakout')
    
    # 3봉 확인 구간
    ax.axvspan(break_plot, entry_plot, alpha=0.2, color='orange', label='3-bar Confirm')
    
    # 진입 시점
    entry_price = trade['entry_price']
    direction = 'LONG' if trade['direction'] == 1 else 'SHORT'
    marker = '^' if trade['direction'] == 1 else 'v'
    color = 'green' if trade['direction'] == 1 else 'red'
    ax.scatter(entry_plot, entry_price, marker=marker, s=200, color=color, zorder=5, label=f'Entry ({direction})')
    
    # 결과 표시
    result = 'WIN' if trade['win'] else 'LOSS'
    result_color = 'green' if trade['win'] else 'red'
    
    # 타이틀
    date_str = df.iloc[trade['entry_idx']]['datetime'].strftime('%Y-%m-%d %H:%M')
    ax.set_title(f"Trade {idx+1}: {date_str} | {direction} | {result} | Max Profit: +{trade['max_profit']:.1f}% | Max Loss: {trade['max_loss']:.1f}%",
                 fontsize=12, fontweight='bold', color=result_color)
    
    ax.set_xlabel('Bars (1H)')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('trade_examples.png', dpi=150, bbox_inches='tight')
print("저장: trade_examples.png")

# 추가: 전체 수익 곡선
fig2, ax2 = plt.subplots(figsize=(12, 6))

cumulative_pnl = []
running_pnl = 0
dates = []

for trade in trade_signals:
    # SL -3%, TP +5% 기준
    if trade['max_profit'] >= 5 and trade['max_loss'] > -3:
        pnl = 5
    elif trade['max_loss'] <= -3:
        pnl = -3
    else:
        pnl = trade['max_profit'] if trade['max_profit'] > 0 else trade['max_loss']
    
    running_pnl += pnl
    cumulative_pnl.append(running_pnl)
    dates.append(df.iloc[trade['entry_idx']]['datetime'])

ax2.plot(dates, cumulative_pnl, 'b-', linewidth=2)
ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
ax2.fill_between(dates, cumulative_pnl, 0, where=[p > 0 for p in cumulative_pnl], alpha=0.3, color='green')
ax2.fill_between(dates, cumulative_pnl, 0, where=[p <= 0 for p in cumulative_pnl], alpha=0.3, color='red')

ax2.set_title('Cumulative PnL: BB30 Squeeze + 3-Bar Confirmation Strategy', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Cumulative Return (%)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('equity_curve.png', dpi=150, bbox_inches='tight')
print("저장: equity_curve.png")

plt.close('all')
