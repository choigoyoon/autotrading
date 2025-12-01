import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
trades_df = pd.read_csv('backtest_HL_strategy_results.csv')
candles_df = pd.read_csv('btc_15m_ohlcv.csv')

# 타임스탬프 변환
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
trades_df['hold_hours'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])

print(f"Total trades: {len(trades_df)}")
print(f"Total candles: {len(candles_df)}")

# 다양한 케이스 선택
selected_trades = []

# 1. Very Strong HL (5%+) - Win
extreme_win = trades_df[(trades_df['strength_group'] == '극강(5%+)') & 
                        (trades_df['exit_reason'] == 'TP2_Full')].head(1)
if len(extreme_win) > 0:
    selected_trades.append(('Case 1: Very Strong HL (>5%) - TP2 Win', extreme_win.iloc[0]))

# 2. Strong HL (2-5%) - Win  
strong_win = trades_df[(trades_df['strength_group'] == '매우강함(2-5%)') & 
                       (trades_df['exit_reason'] == 'TP2_Full')].head(1)
if len(strong_win) > 0:
    selected_trades.append(('Case 2: Strong HL (2-5%) - TP2 Win', strong_win.iloc[0]))

# 3. Medium HL (1-2%) - Win
medium_win = trades_df[(trades_df['strength_group'] == '강함(1-2%)') & 
                       (trades_df['exit_reason'] == 'TP2_Full')].head(1)
if len(medium_win) > 0:
    selected_trades.append(('Case 3: Medium HL (1-2%) - TP2 Win', medium_win.iloc[0]))

# 4. Weak HL (0.5-1%) - Win
weak_win = trades_df[(trades_df['strength_group'] == '보통(0.5-1%)') & 
                     (trades_df['exit_reason'] == 'TP2_Full')].head(1)
if len(weak_win) > 0:
    selected_trades.append(('Case 4: Weak HL (0.5-1%) - TP2 Win', weak_win.iloc[0]))

# 5. Weak HL - Loss
weak_loss = trades_df[(trades_df['strength_group'] == '보통(0.5-1%)') & 
                      (trades_df['exit_reason'] == 'SL')].head(1)
if len(weak_loss) > 0:
    selected_trades.append(('Case 5: Weak HL (0.5-1%) - SL Loss', weak_loss.iloc[0]))

# 6. Strong HL but SL Loss (worst case)
strong_loss = trades_df[(trades_df['strength_group'] == '극강(5%+)') & 
                        (trades_df['exit_reason'] == 'SL')].head(1)
if len(strong_loss) > 0:
    selected_trades.append(('Case 6: Very Strong HL (>5%) - SL Loss', strong_loss.iloc[0]))

print(f"\nSelected {len(selected_trades)} trades for visualization\n")

# 각 거래를 차트로 시각화
fig = plt.figure(figsize=(22, 5.5 * len(selected_trades)))

for idx, (title, trade) in enumerate(selected_trades, 1):
    print(f"Processing {title}")
    print(f"  Entry: {trade['entry_time']} | Exit: {trade['exit_time']}")
    print(f"  PNL: {trade['pnl_pct']:.2f}% | HL Strength: {trade['HL_strength']:.2f}%\n")
    
    # 차트 데이터 추출
    entry_time = trade['entry_time']
    exit_time = trade['exit_time']
    
    # 진입 시점 찾기
    entry_idx = candles_df[candles_df['datetime'] <= entry_time].index[-1] if len(candles_df[candles_df['datetime'] <= entry_time]) > 0 else 0
    
    # 차트 범위 설정 (진입 전 60캔들 ~ 청산 후 30캔들)
    start_idx = max(0, entry_idx - 60)
    exit_idx = candles_df[candles_df['datetime'] <= exit_time].index[-1] if len(candles_df[candles_df['datetime'] <= exit_time]) > 0 else entry_idx + 30
    end_idx = min(len(candles_df) - 1, exit_idx + 30)
    
    chart_data = candles_df.iloc[start_idx:end_idx].copy()
    
    # 서브플롯 생성
    ax = plt.subplot(len(selected_trades), 1, idx)
    
    # 캔들스틱 그리기
    for i, candle in chart_data.iterrows():
        color = '#26a69a' if candle['close'] >= candle['open'] else '#ef5350'  # 초록/빨강
        # 몸통
        ax.plot([candle['datetime'], candle['datetime']], 
               [candle['open'], candle['close']], 
               color=color, linewidth=4, solid_capstyle='round', alpha=0.9)
        # 꼬리
        ax.plot([candle['datetime'], candle['datetime']], 
               [candle['low'], candle['high']], 
               color=color, linewidth=1.2, alpha=0.7)
    
    # HL 가격 계산
    hl_price = trade['entry_price'] / (1 + trade['HL_strength'] / 100)
    sl_price = hl_price * 0.99
    
    # TP 설정
    if trade['HL_strength'] >= 5:
        tp1_pct, tp2_pct = 2.0, 4.0
    elif trade['HL_strength'] >= 2:
        tp1_pct, tp2_pct = 1.5, 3.0
    elif trade['HL_strength'] >= 1:
        tp1_pct, tp2_pct = 1.0, 2.0
    else:
        tp1_pct, tp2_pct = 0.7, 1.5
    
    tp1_price = trade['entry_price'] * (1 + tp1_pct / 100)
    tp2_price = trade['entry_price'] * (1 + tp2_pct / 100)
    
    # 라인 그리기
    ax.axhline(y=hl_price, color='#ff9800', linestyle=':', linewidth=2.5, alpha=0.9, 
              label=f'HL Price: ${hl_price:.2f}', zorder=3)
    ax.axhline(y=sl_price, color='#f44336', linestyle='--', linewidth=2.5, alpha=0.8, 
              label=f'Stop Loss: ${sl_price:.2f} (-1% from HL)', zorder=3)
    ax.axhline(y=tp1_price, color='#8bc34a', linestyle='--', linewidth=1.8, alpha=0.7, 
              label=f'TP1: ${tp1_price:.2f} (+{tp1_pct}%)', zorder=3)
    ax.axhline(y=tp2_price, color='#4caf50', linestyle='--', linewidth=2.5, alpha=0.9, 
              label=f'TP2: ${tp2_price:.2f} (+{tp2_pct}%)', zorder=3)
    
    # 진입/청산 지점 표시
    ax.scatter(entry_time, trade['entry_price'], color='#2196f3', s=400, marker='^', 
              label=f"ENTRY: ${trade['entry_price']:.2f}", zorder=6, edgecolors='white', linewidth=3)
    
    exit_color = '#4caf50' if trade['pnl_pct'] > 0 else '#f44336'
    ax.scatter(exit_time, trade['exit_price'], color=exit_color, s=400, marker='v', 
              label=f"EXIT: ${trade['exit_price']:.2f} ({trade['exit_reason']})", 
              zorder=6, edgecolors='white', linewidth=3)
    
    # 진입/청산 시점 수직선
    ax.axvline(x=entry_time, color='#2196f3', linestyle=':', linewidth=2.5, alpha=0.6, zorder=2)
    ax.axvline(x=exit_time, color=exit_color, linestyle=':', linewidth=2.5, alpha=0.6, zorder=2)
    
    # 제목
    result_text = f"PROFIT: +{trade['pnl_pct']:.2f}%" if trade['pnl_pct'] > 0 else f"LOSS: {trade['pnl_pct']:.2f}%"
    result_color = 'green' if trade['pnl_pct'] > 0 else 'red'
    
    ax.set_title(f'{title}\n'
                f'Entry: {entry_time.strftime("%Y-%m-%d %H:%M")} → '
                f'Exit: {exit_time.strftime("%Y-%m-%d %H:%M")} ({trade["hold_hours"]:.1f}h) | '
                f'{result_text} | '
                f'HL Strength: {trade["HL_strength"]:.2f}%',
                fontsize=14, fontweight='bold', pad=18, color=result_color)
    
    ax.set_ylabel('BTC/USDT Price', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    
    # y축 범위 조정
    y_min = chart_data['low'].min() * 0.998
    y_max = chart_data['high'].max() * 1.002
    ax.set_ylim([y_min, y_max])
    
    # x축 날짜 포맷
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=8))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
    
    # 배경색
    ax.set_facecolor('#fafafa')

plt.tight_layout()
plt.savefig('actual_trades_visualization.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n✅ Chart saved: actual_trades_visualization.png")

# 통계 정보 출력
print("\n" + "="*90)
print("DETAILED TRADES SUMMARY")
print("="*90)
for title, trade in selected_trades:
    print(f"\n{title}")
    print(f"  Entry Time:    {trade['entry_time']}")
    print(f"  Entry Price:   ${trade['entry_price']:.2f}")
    print(f"  HL Price:      ${trade['entry_price'] / (1 + trade['HL_strength'] / 100):.2f} (Reference)")
    print(f"  Exit Time:     {trade['exit_time']}")
    print(f"  Exit Price:    ${trade['exit_price']:.2f}")
    print(f"  Exit Reason:   {trade['exit_reason']}")
    print(f"  PNL:           {trade['pnl_pct']:.2f}%")
    print(f"  HL Strength:   {trade['HL_strength']:.2f}%")
    print(f"  Hold Duration: {trade['hold_hours']:.2f} hours")
    print("-" * 90)

