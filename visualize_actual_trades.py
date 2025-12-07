import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np

# 한글 폰트 설정
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

# 다양한 케이스 선택 (승리/패배, 강도별)
selected_trades = []

# 1. 극강 HL 승리 케이스
extreme_win = trades_df[(trades_df['strength_group'] == '극강 (5%+)') & 
                        (trades_df['exit_reason'] == 'TP2_Full')].head(1)
if len(extreme_win) > 0:
    selected_trades.append(('극강 HL 승리 (TP2)', extreme_win.iloc[0]))

# 2. 강함 HL 승리 케이스
strong_win = trades_df[(trades_df['strength_group'] == '강함 (2-5%)') & 
                       (trades_df['exit_reason'] == 'TP2_Full')].head(1)
if len(strong_win) > 0:
    selected_trades.append(('강함 HL 승리 (TP2)', strong_win.iloc[0]))

# 3. 보통 HL 승리 케이스
normal_win = trades_df[(trades_df['strength_group'] == '보통 (1-2%)') & 
                       (trades_df['exit_reason'] == 'TP2_Full')].head(1)
if len(normal_win) > 0:
    selected_trades.append(('보통 HL 승리 (TP2)', normal_win.iloc[0]))

# 4. 보통 HL 손절 케이스
normal_loss = trades_df[(trades_df['strength_group'] == '보통 (1-2%)') & 
                        (trades_df['exit_reason'] == 'SL')].head(1)
if len(normal_loss) > 0:
    selected_trades.append(('보통 HL 손절 (SL)', normal_loss.iloc[0]))

# 5. 약함 HL 손절 케이스
weak_loss = trades_df[(trades_df['strength_group'] == '약함 (0.5-1%)') & 
                      (trades_df['exit_reason'] == 'SL')].head(1)
if len(weak_loss) > 0:
    selected_trades.append(('약함 HL 손절 (SL)', weak_loss.iloc[0]))

# 6. 큰 손실 케이스
big_loss = trades_df[trades_df['exit_reason'] == 'SL'].nsmallest(1, 'pnl_pct')
if len(big_loss) > 0:
    selected_trades.append(('최대 손실', big_loss.iloc[0]))

print(f"\nSelected {len(selected_trades)} trades for visualization")

# 각 거래를 차트로 시각화
fig = plt.figure(figsize=(20, 5 * len(selected_trades)))

for idx, (title, trade) in enumerate(selected_trades, 1):
    print(f"\nProcessing trade {idx}: {title}")
    print(f"  Entry: {trade['entry_time']}")
    print(f"  Exit: {trade['exit_time']}")
    print(f"  PNL: {trade['pnl_pct']:.2f}%")
    print(f"  HL Strength: {trade['HL_strength']:.2f}%")
    
    # 차트 데이터 추출 (진입 전 50캔들 ~ 청산 후 20캔들)
    entry_time = trade['entry_time']
    exit_time = trade['exit_time']
    
    # 진입 시점 찾기
    entry_idx = candles_df[candles_df['datetime'] <= entry_time].index[-1] if len(candles_df[candles_df['datetime'] <= entry_time]) > 0 else 0
    
    # 차트 범위 설정
    start_idx = max(0, entry_idx - 50)
    exit_idx = candles_df[candles_df['datetime'] <= exit_time].index[-1] if len(candles_df[candles_df['datetime'] <= exit_time]) > 0 else entry_idx + 20
    end_idx = min(len(candles_df) - 1, exit_idx + 20)
    
    chart_data = candles_df.iloc[start_idx:end_idx].copy()
    
    # 서브플롯 생성
    ax = plt.subplot(len(selected_trades), 1, idx)
    
    # 캔들스틱 그리기
    for i, candle in chart_data.iterrows():
        color = 'green' if candle['close'] >= candle['open'] else 'red'
        # 몸통
        ax.plot([candle['datetime'], candle['datetime']], 
               [candle['open'], candle['close']], 
               color=color, linewidth=3, solid_capstyle='round')
        # 꼬리
        ax.plot([candle['datetime'], candle['datetime']], 
               [candle['low'], candle['high']], 
               color=color, linewidth=1)
    
    # HL 가격 표시 (SL 라인)
    hl_price = trade['entry_price'] / (1 + trade['HL_strength'] / 100)
    sl_price = hl_price * 0.99
    
    # TP 라인 표시
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
    
    # HL, SL, TP 라인 먼저 그리기
    ax.axhline(y=hl_price, color='orange', linestyle=':', linewidth=2, alpha=0.8, label=f'HL: ${hl_price:.2f}', zorder=3)
    ax.axhline(y=sl_price, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'SL: ${sl_price:.2f}', zorder=3)
    ax.axhline(y=tp1_price, color='lightgreen', linestyle='--', linewidth=1.5, alpha=0.6, label=f'TP1 ({tp1_pct}%): ${tp1_price:.2f}', zorder=3)
    ax.axhline(y=tp2_price, color='darkgreen', linestyle='--', linewidth=2, alpha=0.8, label=f'TP2 ({tp2_pct}%): ${tp2_price:.2f}', zorder=3)
    
    # 진입/청산 지점 표시
    ax.scatter(entry_time, trade['entry_price'], color='blue', s=300, marker='^', 
              label=f"Entry: ${trade['entry_price']:.2f}", zorder=5, edgecolors='black', linewidth=2)
    
    exit_color = 'green' if trade['pnl_pct'] > 0 else 'red'
    ax.scatter(exit_time, trade['exit_price'], color=exit_color, s=300, marker='v', 
              label=f"Exit: ${trade['exit_price']:.2f} ({trade['exit_reason']})", zorder=5, edgecolors='black', linewidth=2)
    
    # 진입/청산 시점 수직선
    ax.axvline(x=entry_time, color='blue', linestyle=':', linewidth=2, alpha=0.5, zorder=2)
    ax.axvline(x=exit_time, color=exit_color, linestyle=':', linewidth=2, alpha=0.5, zorder=2)
    
    # 제목 및 레이블
    pnl_color = 'green' if trade['pnl_pct'] > 0 else 'red'
    ax.set_title(f'{title}\n'
                f'Entry: {entry_time.strftime("%Y-%m-%d %H:%M")} | '
                f'Exit: {exit_time.strftime("%Y-%m-%d %H:%M")} ({trade["hold_hours"]:.1f}h) | '
                f'PNL: {trade["pnl_pct"]:.2f}% | '
                f'HL Strength: {trade["HL_strength"]:.2f}%',
                fontsize=13, fontweight='bold', pad=15)
    
    ax.set_ylabel('Price (USDT)', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # x축 날짜 포맷
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('actual_trades_visualization.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Chart saved: actual_trades_visualization.png")

# 통계 정보 출력
print("\n" + "="*80)
print("SELECTED TRADES SUMMARY")
print("="*80)
for title, trade in selected_trades:
    print(f"\n{title}:")
    print(f"  Entry Time: {trade['entry_time']}")
    print(f"  Entry Price: ${trade['entry_price']:.2f}")
    print(f"  Exit Time: {trade['exit_time']}")
    print(f"  Exit Price: ${trade['exit_price']:.2f}")
    print(f"  Exit Reason: {trade['exit_reason']}")
    print(f"  PNL: {trade['pnl_pct']:.2f}%")
    print(f"  HL Strength: {trade['HL_strength']:.2f}%")
    print(f"  Hold Time: {trade['hold_hours']:.2f} hours")

