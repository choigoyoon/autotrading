#!/usr/bin/env python3
"""
MTF (Multi-Time Frame) 필터 전략

핵심: 큰 시간대 추세 확인 후에만 작은 시간대에서 진입
주봉 상승추세 깨지면 → Long 금지!
"""

import pandas as pd
import numpy as np

def analyze_trend(df, period_name=''):
    """
    추세 분석: Higher High/Lower High, Higher Low/Lower Low
    """
    
    # Swing High/Low 찾기 (간단 버전: 좌우 5개 확인)
    highs = []
    lows = []
    
    for i in range(5, len(df) - 5):
        # Swing High
        if df.iloc[i]['high'] == df.iloc[i-5:i+6]['high'].max():
            highs.append({
                'datetime': df.iloc[i]['datetime'],
                'price': df.iloc[i]['high']
            })
        
        # Swing Low
        if df.iloc[i]['low'] == df.iloc[i-5:i+6]['low'].min():
            lows.append({
                'datetime': df.iloc[i]['datetime'],
                'price': df.iloc[i]['low']
            })
    
    if len(highs) < 3 or len(lows) < 3:
        return 'unknown', {}
    
    # 최근 3개 고점/저점
    recent_highs = [h['price'] for h in highs[-3:]]
    recent_lows = [l['price'] for l in lows[-3:]]
    
    # 추세 판단
    h1, h2, h3 = recent_highs
    l1, l2, l3 = recent_lows
    
    # 상승 추세: HH + HL
    if h3 > h2 > h1 and l3 > l2 > l1:
        trend = 'uptrend'
    
    # 하락 추세: LH + LL
    elif h3 < h2 < h1 and l3 < l2 < l1:
        trend = 'downtrend'
    
    # 상승추세 깨짐: HH → LH (고점 낮아짐)
    elif h2 > h1 and h3 < h2:
        trend = 'uptrend_broken'
    
    # 하락추세 깨짐: LL → HL (저점 높아짐)
    elif l2 < l1 and l3 > l2:
        trend = 'downtrend_broken'
    
    else:
        trend = 'sideways'
    
    return trend, {
        'recent_highs': recent_highs,
        'recent_lows': recent_lows,
        'high_structure': 'HH' if h3 > h2 else 'LH',
        'low_structure': 'HL' if l3 > l2 else 'LL'
    }


def check_mtf_alignment(weekly_trend, daily_trend, h4_trend):
    """
    MTF 정렬 확인
    
    매매 허용 조건:
    1. 주봉 상승 + 일봉 상승 → Long 허용
    2. 주봉 하락 + 일봉 하락 → Short 허용
    3. 주봉 상승추세 깨짐 → Long 금지, 초단타만
    """
    
    # Long 허용 조건
    allow_long = False
    allow_short = False
    trade_type = 'none'
    
    # 완벽한 상승 정렬
    if weekly_trend == 'uptrend' and daily_trend in ['uptrend', 'sideways']:
        allow_long = True
        trade_type = 'swing_long'  # 스윙 가능
    
    # 주봉 상승이지만 일봉 조정
    elif weekly_trend == 'uptrend' and daily_trend == 'downtrend':
        allow_long = True
        trade_type = 'scalp_long'  # 짧먹만
    
    # 주봉 상승추세 깨짐 → 초단타만
    elif weekly_trend == 'uptrend_broken':
        allow_long = True
        trade_type = 'ultra_scalp_long'  # 초단타만 (1:1)
    
    # 완벽한 하락 정렬
    elif weekly_trend == 'downtrend' and daily_trend in ['downtrend', 'sideways']:
        allow_short = True
        trade_type = 'swing_short'
    
    # 주봉 하락이지만 일봉 반등
    elif weekly_trend == 'downtrend' and daily_trend == 'uptrend':
        allow_short = True
        trade_type = 'scalp_short'
    
    # 주봉 하락추세 깨짐 → 초단타만
    elif weekly_trend == 'downtrend_broken':
        allow_short = True
        trade_type = 'ultra_scalp_short'
    
    return {
        'allow_long': allow_long,
        'allow_short': allow_short,
        'trade_type': trade_type
    }


def run_mtf_backtest():
    """
    MTF 필터 적용 백테스트
    """
    
    print("="*80)
    print("MTF (Multi-Time Frame) 필터 전략")
    print("="*80)
    
    # 데이터 로드
    df_15m = pd.read_csv('btc_15m_ohlcv.csv')
    df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
    df_15m = df_15m[df_15m['datetime'] >= '2024-01-01'].copy()
    
    df_1h = pd.read_csv('btc_1h_ohlcv.csv')
    df_1h['datetime'] = pd.to_datetime(df_1h['datetime'])
    df_1h = df_1h[df_1h['datetime'] >= '2024-01-01'].copy()
    
    df_4h = pd.read_csv('btc_4h_ohlcv.csv')
    df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])
    df_4h = df_4h[df_4h['datetime'] >= '2024-01-01'].copy()
    
    # 주봉 생성
    df_weekly = df_1h.copy()
    df_weekly = df_weekly.set_index('datetime')
    df_weekly = df_weekly.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    
    print(f"\n데이터 기간: {df_15m['datetime'].min()} ~ {df_15m['datetime'].max()}")
    print(f"15분봉: {len(df_15m):,}개")
    print(f"1시간봉: {len(df_1h):,}개")
    print(f"4시간봉: {len(df_4h):,}개")
    print(f"주봉: {len(df_weekly):,}개")
    
    # 추세 분석
    print("\n" + "="*80)
    print("현재 MTF 추세 분석")
    print("="*80)
    
    weekly_trend, weekly_info = analyze_trend(df_weekly, '주봉')
    print(f"\n주봉 추세: {weekly_trend}")
    print(f"  고점 구조: {weekly_info.get('high_structure', 'N/A')}")
    print(f"  저점 구조: {weekly_info.get('low_structure', 'N/A')}")
    print(f"  최근 고점: {weekly_info.get('recent_highs', [])}")
    print(f"  최근 저점: {weekly_info.get('recent_lows', [])}")
    
    daily_trend, daily_info = analyze_trend(df_4h, '일봉(4시간)')
    print(f"\n4시간봉 추세: {daily_trend}")
    print(f"  고점 구조: {daily_info.get('high_structure', 'N/A')}")
    print(f"  저점 구조: {daily_info.get('low_structure', 'N/A')}")
    
    h1_trend, h1_info = analyze_trend(df_1h, '1시간')
    print(f"\n1시간봉 추세: {h1_trend}")
    print(f"  고점 구조: {h1_info.get('high_structure', 'N/A')}")
    print(f"  저점 구조: {h1_info.get('low_structure', 'N/A')}")
    
    # MTF 정렬 확인
    mtf_status = check_mtf_alignment(weekly_trend, daily_trend, h1_trend)
    
    print("\n" + "="*80)
    print("MTF 매매 허용 상태")
    print("="*80)
    print(f"Long 허용: {mtf_status['allow_long']}")
    print(f"Short 허용: {mtf_status['allow_short']}")
    print(f"매매 타입: {mtf_status['trade_type']}")
    
    # 매매 타입별 설명
    trade_type_desc = {
        'swing_long': '스윙 Long 가능 (주봉 상승 + 일봉 상승)',
        'scalp_long': '짧먹 Long만 (주봉 상승 + 일봉 조정)',
        'ultra_scalp_long': '초단타만! (주봉 상승추세 깨짐)',
        'swing_short': '스윙 Short 가능 (주봉 하락 + 일봉 하락)',
        'scalp_short': '짧먹 Short만 (주봉 하락 + 일봉 반등)',
        'ultra_scalp_short': '초단타만! (주봉 하락추세 깨짐)',
        'none': '매매 금지 (추세 불명확)'
    }
    
    print(f"\n설명: {trade_type_desc.get(mtf_status['trade_type'], '알 수 없음')}")
    
    # 현재 상황 판단
    print("\n" + "="*80)
    print("현재 상황 종합 판단")
    print("="*80)
    
    latest_price = df_15m.iloc[-1]['close']
    print(f"\n현재가: ${latest_price:,.2f}")
    
    if weekly_trend == 'uptrend_broken':
        print("\n⚠️ 주봉 상승추세 깨짐!")
        print("   → 리테스트 할지, 더 내릴지 모르는 자리")
        print("   → 명확한 신호 나올 때까지 관망 또는 초단타만")
        print("   → Long 진입 시 TP 1:1, 빠른 청산 필수")
    
    elif weekly_trend == 'downtrend':
        print("\n🔴 주봉 하락 추세")
        print("   → Long 진입 금지!")
        print("   → Short 위주 전략")
    
    elif weekly_trend == 'uptrend':
        print("\n🟢 주봉 상승 추세 유지")
        print("   → Long 진입 가능")
        print("   → 일봉 조정 시 매수 기회")
    
    # 저장
    result = {
        'analysis_time': df_15m.iloc[-1]['datetime'],
        'current_price': latest_price,
        'weekly_trend': weekly_trend,
        'daily_trend': daily_trend,
        'h1_trend': h1_trend,
        'allow_long': mtf_status['allow_long'],
        'allow_short': mtf_status['allow_short'],
        'trade_type': mtf_status['trade_type']
    }
    
    pd.DataFrame([result]).to_csv('mtf_analysis_current.csv', index=False)
    
    print("\n" + "="*80)
    print("✅ 분석 완료: mtf_analysis_current.csv")
    print("="*80)


if __name__ == "__main__":
    run_mtf_backtest()
