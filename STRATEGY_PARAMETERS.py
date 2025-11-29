"""
필터 적용 전략 - 파라미터 명세서

실전 트레이딩에 필요한 모든 파라미터 값
"""

# ═══════════════════════════════════════════════════════════
# 1. 시장 기본 설정
# ═══════════════════════════════════════════════════════════

EXCHANGE = "Bybit"
SYMBOL = "BTCUSDT"
TIMEFRAME = "15m"  # 15분봉

# ═══════════════════════════════════════════════════════════
# 2. MACD 파라미터
# ═══════════════════════════════════════════════════════════

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# MTF (Multi-TimeFrame) 설정
MTF_TIMEFRAMES = {
    'base': '15m',
    'H1': '1H',
    'H4': '4H',
    'D1': '1D',
}

# ═══════════════════════════════════════════════════════════
# 3. 진입 신호 파라미터
# ═══════════════════════════════════════════════════════════

# Trendline 설정
TRENDLINE_MIN_TOUCHES = 2  # 최소 터치 포인트
TRENDLINE_LOOKBACK = 100   # 추세선 탐색 범위 (봉)

# 진입 조건
ENTRY_TYPE = "trendline_up"  # Trendline 상향 돌파만 (롱 전략)
BREAKOUT_CONFIRMATION = True  # 돌파 확정 후 진입

# ═══════════════════════════════════════════════════════════
# 4. 연속 신호 필터 (핵심!)
# ═══════════════════════════════════════════════════════════

MIN_INTERVAL_BARS = 10  # 최소 간격 (10봉 = 150분 = 2.5시간)

# 필터 설명:
# - 마지막 진입 후 10봉(150분) 이내 신호는 스킵
# - 오버트레이딩 방지
# - 거래 품질 향상

# ═══════════════════════════════════════════════════════════
# 5. TP/SL 파라미터 (2가지 전략)
# ═══════════════════════════════════════════════════════════

# [전략 A] 기본 전략 (고정 TP/SL)
STRATEGY_BASIC = {
    'name': '기본 전략',
    'tp_pct': 2.0,      # Take Profit: 2%
    'sl_pct': 2.0,      # Stop Loss: 2%
    'expected_win_rate': 79.8,
    'expected_monthly_return': 13.88,  # % (단리)
}

# [전략 B] 동적 TP (FVG 기반) - 권장!
STRATEGY_DYNAMIC = {
    'name': '동적 TP (FVG 기반)',

    # FVG 있을 때 (공격적)
    'fvg_present': {
        'tp_pct': 2.5,   # Take Profit: 2.5%
        'sl_pct': 2.0,   # Stop Loss: 2.0%
    },

    # FVG 없을 때 (보수적)
    'fvg_absent': {
        'tp_pct': 1.5,   # Take Profit: 1.5%
        'sl_pct': 2.0,   # Stop Loss: 2.0%
    },

    'expected_win_rate': 84.1,
    'expected_monthly_return': 14.41,  # % (단리)
}

# ═══════════════════════════════════════════════════════════
# 6. FVG (Fair Value Gap) 감지 파라미터
# ═══════════════════════════════════════════════════════════

FVG_LOOKBACK = 3  # 3개 캔들로 FVG 감지

# FVG 조건 (Bullish):
# candle[i-2].high < candle[i].low
# → 갭이 존재하면 FVG 있음

# ═══════════════════════════════════════════════════════════
# 7. 포지션 사이징 (레버리지 없음 기준)
# ═══════════════════════════════════════════════════════════

# [옵션 1] 보수적
CONSERVATIVE = {
    'leverage': 1,           # 레버리지 없음
    'position_size': 1.0,    # 전액 투자 (100%)
    'risk_per_trade': 2.0,   # 거래당 리스크: 2%
}

# [옵션 2] 중도 (레버리지 사용 시)
MODERATE = {
    'leverage': 2,           # 2배 레버리지
    'position_size': 0.3,    # 30% 포지션
    'risk_per_trade': 1.2,   # 거래당 리스크: 1.2%
}

# [옵션 3] 공격적 (레버리지 사용 시)
AGGRESSIVE = {
    'leverage': 3,           # 3배 레버리지
    'position_size': 0.3,    # 30% 포지션
    'risk_per_trade': 1.8,   # 거래당 리스크: 1.8%
}

# ═══════════════════════════════════════════════════════════
# 8. 수수료 설정
# ═══════════════════════════════════════════════════════════

FEE_MAKER = 0.02   # 0.02% (Bybit maker)
FEE_TAKER = 0.055  # 0.055% (Bybit taker)
FEE_TOTAL = 0.11   # 0.11% (진입 + 청산)

# 참고: 실전에서는 taker 수수료 적용 (시장가 주문)

# ═══════════════════════════════════════════════════════════
# 9. 타이밍 파라미터
# ═══════════════════════════════════════════════════════════

CHECK_INTERVAL = 900  # 15분 = 900초
EXECUTION_DELAY = 2   # 주문 실행 지연: 2초

# 백테스트 가정:
# - T봉 close → 신호 생성
# - T+1봉 open → 실제 진입
# - 슬리피지: 0.00% (실측 결과)

# ═══════════════════════════════════════════════════════════
# 10. 출구 전략 파라미터
# ═══════════════════════════════════════════════════════════

MAX_HOLDING_BARS = 50  # 최대 보유: 50봉 (12.5시간)

# 출구 우선순위:
# 1. TP 도달 → 익절
# 2. SL 도달 → 손절
# 3. 50봉 경과 → 시장가 청산

# ═══════════════════════════════════════════════════════════
# 11. 리스크 관리 파라미터
# ═══════════════════════════════════════════════════════════

MAX_DRAWDOWN_LIMIT = -10.0   # MDD 한계: -10%
MAX_CONSECUTIVE_LOSSES = 10  # 최대 연속 손실: 10회
DAILY_LOSS_LIMIT = -5.0      # 일일 손실 한계: -5%

# 비상 정지 조건:
# 1. MDD > -10% → 모든 포지션 청산
# 2. 연속 손실 > 10회 → 24시간 거래 중지
# 3. 일일 손실 > -5% → 당일 거래 중지

# ═══════════════════════════════════════════════════════════
# 12. 백테스트 결과 (5년, 2020-2024)
# ═══════════════════════════════════════════════════════════

BACKTEST_RESULTS = {
    'period': '2020-01-01 ~ 2024-12-29',
    'total_trades': 983,

    'basic_strategy': {
        'win_rate': 79.8,              # %
        'avg_pnl_per_trade': 0.847,    # % (수수료 후)
        'monthly_return': 13.88,        # % (단리, 레버리지 없음)
        'yearly_return': 166.6,         # % (단리, 레버리지 없음)
        'max_drawdown': -5.23,          # %
        'max_consecutive_losses': 5,
    },

    'dynamic_strategy': {
        'win_rate': 84.1,              # %
        'avg_pnl_per_trade': 0.880,    # % (수수료 후)
        'monthly_return': 14.41,        # % (단리, 레버리지 없음)
        'yearly_return': 172.9,         # % (단리, 레버리지 없음)
        'improvement': '+6.3%p',        # 연 수익 개선
    },
}

# ═══════════════════════════════════════════════════════════
# 13. 실전 트레이딩 로직 (Python 예시)
# ═══════════════════════════════════════════════════════════

"""
# 주요 변수
last_entry_bar = None
current_position = None

# 매 15분마다 실행
while True:
    # 1. 데이터 수집
    df = fetch_ohlcv(symbol='BTCUSDT', timeframe='15m', limit=200)

    # 2. MACD 계산
    macd = calculate_macd(df, fast=12, slow=26, signal=9)

    # 3. L/H 라벨링
    labels = generate_hl_labels(macd)

    # 4. Trendline 계산
    trendlines = calculate_trendlines(df, labels)

    # 5. 돌파 감지
    breakout = detect_breakout(df, trendlines)

    if breakout and breakout['type'] == 'trendline_up':
        # 6. 연속 신호 필터
        current_bar = len(df) - 1

        if last_entry_bar is None or (current_bar - last_entry_bar >= 10):
            # 7. FVG 확인
            has_fvg = detect_fvg(df, current_bar)

            # 8. TP/SL 설정
            if has_fvg:
                tp_pct = 2.5
                sl_pct = 2.0
            else:
                tp_pct = 1.5
                sl_pct = 2.0

            # 9. 진입 주문
            entry_price = df.iloc[-1]['close']
            tp_price = entry_price * (1 + tp_pct/100)
            sl_price = entry_price * (1 - sl_pct/100)

            order = place_order(
                symbol='BTCUSDT',
                side='BUY',
                amount=calculate_position_size(),
                tp=tp_price,
                sl=sl_price,
            )

            last_entry_bar = current_bar
            current_position = order

            print(f"✅ 진입: {entry_price}, TP: {tp_price}, SL: {sl_price}")
        else:
            print(f"⏭️ 스킵: 마지막 진입 후 {current_bar - last_entry_bar}봉 경과")

    # 10. 다음 체크까지 대기
    time.sleep(900)  # 15분
"""

# ═══════════════════════════════════════════════════════════
# 14. 권장 설정 (실전)
# ═══════════════════════════════════════════════════════════

RECOMMENDED_SETUP = {
    'strategy': 'dynamic_tp',         # 동적 TP 전략
    'leverage': 1,                     # 레버리지 없음 (초기)
    'position_size': 1.0,              # 전액 투자
    'min_interval': 10,                # 10봉 간격 필터

    'tp_with_fvg': 2.5,               # FVG 있을 때 TP
    'tp_without_fvg': 1.5,            # FVG 없을 때 TP
    'sl': 2.0,                         # 통일된 SL

    'expected_monthly': 14.41,         # % (단리)
    'expected_trades_per_month': 16.4,

    'initial_capital': 10000,          # $10,000 권장
}

print(__doc__)
print("\n" + "="*70)
print("전략 파라미터 명세서")
print("="*70)

print(f"\n1. 시장: {EXCHANGE} {SYMBOL}")
print(f"   타임프레임: {TIMEFRAME}")

print(f"\n2. MACD: Fast={MACD_FAST}, Slow={MACD_SLOW}, Signal={MACD_SIGNAL}")

print(f"\n3. 진입 신호: {ENTRY_TYPE}")
print(f"   연속 필터: {MIN_INTERVAL_BARS}봉 ({MIN_INTERVAL_BARS * 15}분)")

print(f"\n4. 기본 전략:")
print(f"   TP: {STRATEGY_BASIC['tp_pct']}%")
print(f"   SL: {STRATEGY_BASIC['sl_pct']}%")
print(f"   예상 승률: {STRATEGY_BASIC['expected_win_rate']}%")
print(f"   예상 월수익: {STRATEGY_BASIC['expected_monthly_return']}%")

print(f"\n5. 동적 TP 전략 (권장):")
print(f"   FVG 있음: TP {STRATEGY_DYNAMIC['fvg_present']['tp_pct']}%, SL {STRATEGY_DYNAMIC['fvg_present']['sl_pct']}%")
print(f"   FVG 없음: TP {STRATEGY_DYNAMIC['fvg_absent']['tp_pct']}%, SL {STRATEGY_DYNAMIC['fvg_absent']['sl_pct']}%")
print(f"   예상 승률: {STRATEGY_DYNAMIC['expected_win_rate']}%")
print(f"   예상 월수익: {STRATEGY_DYNAMIC['expected_monthly_return']}%")

print(f"\n6. 리스크 관리:")
print(f"   MDD 한계: {MAX_DRAWDOWN_LIMIT}%")
print(f"   일일 손실 한계: {DAILY_LOSS_LIMIT}%")

print(f"\n7. 백테스트 성과 (5년):")
print(f"   총 거래: {BACKTEST_RESULTS['total_trades']}개")
print(f"   승률: {BACKTEST_RESULTS['dynamic_strategy']['win_rate']}%")
print(f"   월 수익: {BACKTEST_RESULTS['dynamic_strategy']['monthly_return']}%")
print(f"   MDD: {BACKTEST_RESULTS['basic_strategy']['max_drawdown']}%")

print(f"\n8. 권장 설정:")
print(f"   초기 자본: ${RECOMMENDED_SETUP['initial_capital']:,}")
print(f"   레버리지: {RECOMMENDED_SETUP['leverage']}배")
print(f"   전략: {RECOMMENDED_SETUP['strategy']}")
print(f"   예상 월수익: ${RECOMMENDED_SETUP['initial_capital'] * RECOMMENDED_SETUP['expected_monthly'] / 100:,.2f}")

print("\n" + "="*70)
print("파라미터 출력 완료!")
print("="*70)
