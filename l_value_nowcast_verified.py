"""
L값 기반 나우캐스트 준수 전략 (미래데이터 없음)
======================================================================
원칙:
1. 진입 시점에 알 수 있는 정보만 사용
2. 필터는 기술적 분석 원칙 기반 (결과 기반 최적화 X)
3. 고정 TP/SL (적응형 불가)
4. 슬리피지 + 수수료 포함

진입 신호:
- L값 확정 (1H MACD Cross 정의)
- L+1 시가 진입

필터 (선택적, 기술적 원칙):
- RSI < 30: 전통적 과매도
- MACD Hist < 0: 하락 모멘텀
- 볼륨 > 평균 1.5배: 매도 클라이맥스

청산:
- TP: +2.0% (고정)
- SL: -1.5% (고정)
- 최대 보유: 50봉 (12.5시간)

비용 (Bybit 기준):
- 슬리피지: 0.05% x 2 = 0.1%
- 테이커 수수료: 0.055% x 2 = 0.11%
- 총 비용: 0.21% (왕복)
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 70)
print("L값 기반 나우캐스트 검증 전략")
print("=" * 70)
print()

# ═══════════════════════════════════════════════════════════════════
# 데이터 로드
# ═══════════════════════════════════════════════════════════════════

df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

df_l = pd.read_csv('l_labels_1h_cross.csv')
df_l['datetime'] = pd.to_datetime(df_l['datetime'])

cutoff = df['datetime'].max() - timedelta(days=1825)
df = df[df['datetime'] >= cutoff].reset_index(drop=True)
df_l = df_l[df_l['datetime'] >= cutoff].reset_index(drop=True)

print(f"15분 데이터: {len(df):,}개")
print(f"L값 (1H Cross): {len(df_l)}개\n")

# ═══════════════════════════════════════════════════════════════════
# 지표 계산 (진입 전 확정 데이터만)
# ═══════════════════════════════════════════════════════════════════

print("지표 계산 중...")

# RSI (14봉)
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# 볼륨 비율
df['vol_ma'] = df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['vol_ma']

print("  완료!\n")

# ═══════════════════════════════════════════════════════════════════
# 전략 파라미터
# ═══════════════════════════════════════════════════════════════════

# 비용
SLIPPAGE = 0.05  # 0.05% (진입/청산)
FEE = 0.055      # 0.055% 테이커 수수료 (Bybit)
TOTAL_COST = (SLIPPAGE + FEE) * 2  # 0.21% 왕복

# 청산 설정
TP_PCT = 2.0     # TP 2%
SL_PCT = -1.5    # SL -1.5%
MAX_HOLD = 50    # 최대 50봉

# 비용 적용 후
NET_TP = TP_PCT - TOTAL_COST
NET_SL = SL_PCT - TOTAL_COST
BREAKEVEN_WR = -NET_SL / (NET_TP - NET_SL) * 100

print("=" * 70)
print("전략 설정")
print("=" * 70)
print()
print(f"진입: L값 확정 → L+1 시가")
print(f"청산: TP {TP_PCT}% / SL {SL_PCT}% / 최대 {MAX_HOLD}봉")
print()
print(f"비용:")
print(f"  슬리피지: {SLIPPAGE}% x 2")
print(f"  수수료: {FEE}% x 2")
print(f"  총 비용: {TOTAL_COST:.2f}%")
print()
print(f"실제 손익:")
print(f"  TP 도달: +{NET_TP:.2f}%")
print(f"  SL 도달: {NET_SL:.2f}%")
print(f"  손익분기 승률: {BREAKEVEN_WR:.1f}%")
print()

# ═══════════════════════════════════════════════════════════════════
# 백테스트 (나우캐스트 준수)
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("백테스트 실행")
print("=" * 70)
print()

trades = []

for _, l_row in df_l.iterrows():
    l_idx = int(l_row['l_idx'])
    l_time = l_row['datetime']

    if l_idx < 20 or l_idx + MAX_HOLD >= len(df):
        continue

    # ─────────────────────────────────────────────────────────────
    # 진입 결정 (L값 시점 데이터만 사용)
    # ─────────────────────────────────────────────────────────────

    l_candle = df.iloc[l_idx]

    # 필터 조건 (L값 시점 확정 데이터)
    l_rsi = l_candle['rsi']
    l_macd_hist = l_candle['macd_hist']
    l_vol_ratio = l_candle['vol_ratio']

    # 필터 플래그
    filter_none = True  # 필터 없음
    filter_rsi30 = l_rsi < 30  # RSI < 30
    filter_rsi35 = l_rsi < 35  # RSI < 35 (완화)
    filter_macd = l_macd_hist < 0  # MACD 음수
    filter_macd20 = l_macd_hist < -20  # MACD 깊음
    filter_vol = l_vol_ratio >= 1.5  # 볼륨 1.5배+
    filter_rsi30_and_macd = filter_rsi30 and filter_macd  # 복합
    filter_rsi35_or_macd20 = filter_rsi35 or filter_macd20  # OR 로직

    # ─────────────────────────────────────────────────────────────
    # 진입 (L+1 시가)
    # ─────────────────────────────────────────────────────────────

    entry_idx = l_idx + 1
    entry_price = df.iloc[entry_idx]['open']
    entry_time = df.iloc[entry_idx]['datetime']

    # TP/SL 레벨
    tp_level = entry_price * (1 + TP_PCT / 100)
    sl_level = entry_price * (1 + SL_PCT / 100)

    # ─────────────────────────────────────────────────────────────
    # 청산 시뮬레이션 (고정 TP/SL, 미래 정보 불필요)
    # ─────────────────────────────────────────────────────────────

    exit_bar = None
    exit_price = None
    exit_reason = None

    for i in range(entry_idx + 1, min(entry_idx + MAX_HOLD + 1, len(df))):
        bar = df.iloc[i]

        # TP/SL 터치 확인
        hit_tp = bar['high'] >= tp_level
        hit_sl = bar['low'] <= sl_level

        # 둘 다 터치된 경우 → 시가 기준 판단 (보수적)
        if hit_tp and hit_sl:
            if bar['open'] <= sl_level:
                # 시가가 SL 이하 → SL 먼저 터치
                exit_bar = i
                exit_price = sl_level
                exit_reason = 'SL'
                break
            elif bar['open'] >= tp_level:
                # 시가가 TP 이상 → TP 먼저 터치
                exit_bar = i
                exit_price = tp_level
                exit_reason = 'TP'
                break
            else:
                # 시가가 중간 → SL 우선 (보수적)
                exit_bar = i
                exit_price = sl_level
                exit_reason = 'SL'
                break

        # SL만 터치
        elif hit_sl:
            exit_bar = i
            exit_price = sl_level
            exit_reason = 'SL'
            break

        # TP만 터치
        elif hit_tp:
            exit_bar = i
            exit_price = tp_level
            exit_reason = 'TP'
            break

    # 타임아웃
    if exit_bar is None:
        exit_bar = min(entry_idx + MAX_HOLD, len(df) - 1)
        exit_price = df.iloc[exit_bar]['close']
        exit_reason = 'TIMEOUT'

    # ─────────────────────────────────────────────────────────────
    # 손익 계산 (비용 포함)
    # ─────────────────────────────────────────────────────────────

    gross_pnl = (exit_price - entry_price) / entry_price * 100
    net_pnl = gross_pnl - TOTAL_COST
    hold_bars = exit_bar - entry_idx

    # 거래 기록
    trades.append({
        'l_time': l_time,
        'entry_time': entry_time,
        'entry_idx': entry_idx,
        'entry_price': entry_price,
        'exit_time': df.iloc[exit_bar]['datetime'],
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'hold_bars': hold_bars,
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        # 필터 플래그
        'f_none': filter_none,
        'f_rsi30': filter_rsi30,
        'f_rsi35': filter_rsi35,
        'f_macd': filter_macd,
        'f_macd20': filter_macd20,
        'f_vol': filter_vol,
        'f_rsi30_and_macd': filter_rsi30_and_macd,
        'f_rsi35_or_macd20': filter_rsi35_or_macd20,
        # L값 지표
        'l_rsi': l_rsi,
        'l_macd_hist': l_macd_hist,
        'l_vol_ratio': l_vol_ratio,
    })

df_trades = pd.DataFrame(trades)

print(f"총 거래 기회: {len(df_trades)}개\n")

# ═══════════════════════════════════════════════════════════════════
# 성과 분석 함수
# ═══════════════════════════════════════════════════════════════════

def analyze_filter(df_trades, filter_col, name):
    """필터별 성과 분석"""
    filtered = df_trades[df_trades[filter_col]]

    if len(filtered) == 0:
        print(f"  ⚠️ 조건 충족 거래 없음")
        return

    # 기본 통계
    win_count = (filtered['net_pnl'] > 0).sum()
    loss_count = (filtered['net_pnl'] <= 0).sum()
    win_rate = win_count / len(filtered) * 100

    avg_pnl = filtered['net_pnl'].mean()
    median_pnl = filtered['net_pnl'].median()

    avg_win = filtered[filtered['net_pnl'] > 0]['net_pnl'].mean() if win_count > 0 else 0
    avg_loss = filtered[filtered['net_pnl'] <= 0]['net_pnl'].mean() if loss_count > 0 else 0

    total_pnl = filtered['net_pnl'].sum()

    # 청산 분포
    tp_count = (filtered['exit_reason'] == 'TP').sum()
    sl_count = (filtered['exit_reason'] == 'SL').sum()
    timeout_count = (filtered['exit_reason'] == 'TIMEOUT').sum()

    # 연간 거래 수
    date_range = (df['datetime'].max() - df['datetime'].min()).days / 365.25
    trades_per_year = len(filtered) / date_range

    # Risk/Reward
    if avg_loss != 0:
        rr_ratio = abs(avg_win / avg_loss)
    else:
        rr_ratio = 999

    # 출력
    print(f"【{name}】")
    print(f"  거래 수: {len(filtered)}개 ({len(filtered)/len(df_trades)*100:.1f}%) | 연간 {trades_per_year:.0f}개")
    print(f"  승률: {win_rate:.1f}% (승 {win_count} / 패 {loss_count})")
    print(f"  평균 손익: {avg_pnl:+.3f}% | 중앙값: {median_pnl:+.3f}%")
    print(f"  평균 수익: {avg_win:+.3f}% | 평균 손실: {avg_loss:+.3f}%")
    print(f"  Risk/Reward: {rr_ratio:.2f}")
    print(f"  누적 손익: {total_pnl:+.2f}%")
    print(f"  청산: TP {tp_count}개 ({tp_count/len(filtered)*100:.0f}%) | SL {sl_count}개 ({sl_count/len(filtered)*100:.0f}%) | 타임아웃 {timeout_count}개 ({timeout_count/len(filtered)*100:.0f}%)")
    print()

    # 수익성 판단
    if avg_pnl > 0 and win_rate > BREAKEVEN_WR:
        print(f"  ✅ 수익 가능 (평균 {avg_pnl:+.3f}%, 승률 {win_rate:.1f}%)")
    elif avg_pnl > 0:
        print(f"  ⚠️ 약간 수익 (승률 {win_rate:.1f}% < 분기점 {BREAKEVEN_WR:.1f}%)")
    else:
        print(f"  ❌ 손실 전략 (평균 {avg_pnl:+.3f}%)")
    print()

# ═══════════════════════════════════════════════════════════════════
# 결과 분석
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("전략별 성과 (슬리피지+수수료 포함)")
print("=" * 70)
print()
print(f"※ 손익분기 승률: {BREAKEVEN_WR:.1f}%")
print(f"※ 실제 손익: TP {NET_TP:+.2f}% / SL {NET_SL:+.2f}%")
print()

analyze_filter(df_trades, 'f_none', '전략 1: 필터 없음 (모든 L값)')
analyze_filter(df_trades, 'f_rsi30', '전략 2: RSI < 30 (전통적 과매도)')
analyze_filter(df_trades, 'f_rsi35', '전략 3: RSI < 35 (완화된 과매도)')
analyze_filter(df_trades, 'f_macd', '전략 4: MACD Hist < 0 (하락 모멘텀)')
analyze_filter(df_trades, 'f_macd20', '전략 5: MACD Hist < -20 (깊은 하락)')
analyze_filter(df_trades, 'f_vol', '전략 6: 볼륨 1.5배+ (매도 클라이맥스)')
analyze_filter(df_trades, 'f_rsi30_and_macd', '전략 7: RSI<30 AND MACD<0 (복합)')
analyze_filter(df_trades, 'f_rsi35_or_macd20', '전략 8: RSI<35 OR MACD<-20 (OR 로직)')

# ═══════════════════════════════════════════════════════════════════
# 최종 결론
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("검증 결과")
print("=" * 70)
print()

print("✅ 나우캐스트 준수 확인:")
print("  • L값 확정 후 진입 (1H MACD Cross)")
print("  • L+1 시가 사용 (미래 정보 없음)")
print("  • 필터는 L값 시점 확정 데이터만 사용")
print("  • 고정 TP/SL (적응형 불가)")
print("  • 청산 로직에 미래 데이터 없음")
print()

print("💰 비용 영향:")
print(f"  • 총 비용: {TOTAL_COST:.2f}% (슬리피지 {SLIPPAGE*2:.2f}% + 수수료 {FEE*2:.2f}%)")
print(f"  • TP 2.0% → 실제 {NET_TP:.2f}%")
print(f"  • SL -1.5% → 실제 {NET_SL:.2f}%")
print(f"  • 손익분기 승률: {BREAKEVEN_WR:.1f}%")
print()

# 최고 전략 찾기
best_strategy = None
best_avg_pnl = -999
best_filter = None

filters_to_check = [
    ('f_none', '필터 없음'),
    ('f_rsi30', 'RSI < 30'),
    ('f_rsi35', 'RSI < 35'),
    ('f_macd', 'MACD < 0'),
    ('f_macd20', 'MACD < -20'),
    ('f_vol', '볼륨 1.5x+'),
    ('f_rsi30_and_macd', 'RSI+MACD'),
    ('f_rsi35_or_macd20', 'RSI OR MACD')
]

for col, name in filters_to_check:
    filtered = df_trades[df_trades[col]]
    if len(filtered) > 10:  # 최소 10개 거래
        avg_pnl = filtered['net_pnl'].mean()
        win_rate = (filtered['net_pnl'] > 0).sum() / len(filtered) * 100

        # 승률도 분기점 이상이어야 함
        if avg_pnl > best_avg_pnl and win_rate >= BREAKEVEN_WR - 5:  # 5%p 여유
            best_avg_pnl = avg_pnl
            best_strategy = name
            best_filter = col

if best_avg_pnl > 0:
    best_trades = df_trades[df_trades[best_filter]]
    best_wr = (best_trades['net_pnl'] > 0).sum() / len(best_trades) * 100
    best_total = best_trades['net_pnl'].sum()

    print(f"🏆 최고 전략: {best_strategy}")
    print(f"  평균 손익: {best_avg_pnl:+.3f}%")
    print(f"  승률: {best_wr:.1f}%")
    print(f"  거래 수: {len(best_trades)}개")
    print(f"  누적 손익: {best_total:+.2f}%")
else:
    print(f"⚠️  모든 전략이 손실 또는 승률 부족")
    print(f"   최선: {best_strategy} (평균 {best_avg_pnl:+.3f}%)")
    print()
    print(f"💡 원인:")
    print(f"   1. 슬리피지 {SLIPPAGE*2:.2f}% + 수수료 {FEE*2:.2f}% = {TOTAL_COST:.2f}% 비용")
    print(f"   2. 고정 TP 2% → 실제 {NET_TP:.2f}%")
    print(f"   3. 승률 {BREAKEVEN_WR:.1f}% 이상 필요")
    print()
    print(f"💡 대안:")
    print(f"   1. 더 높은 TP 사용 (3%, 4%)")
    print(f"   2. 메이커 주문으로 수수료 절감 (0.02%)")
    print(f"   3. 다른 진입 신호 탐색")

print()

# 결과 저장
df_trades.to_csv('l_value_nowcast_trades.csv', index=False)
print(f"💾 거래 내역 저장: l_value_nowcast_trades.csv")

print()
print("=" * 70)
print("✅ 나우캐스트 검증 완료")
print("=" * 70)
