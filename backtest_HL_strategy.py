#!/usr/bin/env python3
"""
HL 패턴 전략 백테스트 (미래 데이터 누락 방지)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 80)
print("HL 패턴 기반 백테스트 (No Look-Ahead Bias)")
print("=" * 80)

# 데이터 로드
print("\n📊 데이터 로딩...")
ohlcv = pd.read_csv('btc_15m_ohlcv.csv')
ohlcv['datetime'] = pd.to_datetime(ohlcv['datetime'])
all_L = pd.read_csv('all_L_values.csv')
all_L['datetime'] = pd.to_datetime(all_L['datetime'])

print(f"✅ OHLCV: {len(ohlcv):,}개 캔들")
print(f"✅ L값: {len(all_L):,}개")

# L값에 인덱스 추가 (빠른 검색용)
ohlcv = ohlcv.reset_index(drop=True)
all_L = all_L.reset_index(drop=True)

print("\n" + "=" * 80)
print("HL 감지 시스템 구축")
print("=" * 80)

# HL 이벤트 생성 (미래 데이터 없음)
HL_events = []

for i in range(1, len(all_L)):
    prev_L = all_L.iloc[i-1]['L_value']
    curr_L = all_L.iloc[i]['L_value']
    
    # HL 발생 확인
    if curr_L > prev_L:
        strength = (curr_L - prev_L) / prev_L * 100
        
        HL_events.append({
            'datetime': all_L.iloc[i]['datetime'],
            'prev_L': prev_L,
            'curr_L': curr_L,
            'strength': strength,
            'L_num': all_L.iloc[i]['L_num']
        })

HL_df = pd.DataFrame(HL_events)
print(f"\n✅ 감지된 HL: {len(HL_df):,}개")
print(f"   강도 분포:")
print(f"   • 극강 (5%+): {(HL_df['strength'] >= 5).sum()}개")
print(f"   • 매우강함 (2-5%): {((HL_df['strength'] >= 2) & (HL_df['strength'] < 5)).sum()}개")
print(f"   • 강함 (1-2%): {((HL_df['strength'] >= 1) & (HL_df['strength'] < 2)).sum()}개")
print(f"   • 보통 (0.5-1%): {((HL_df['strength'] >= 0.5) & (HL_df['strength'] < 1)).sum()}개")
print(f"   • 약함 (0-0.5%): {(HL_df['strength'] < 0.5).sum()}개")

print("\n" + "=" * 80)
print("보조지표 계산 (각 캔들마다)")
print("=" * 80)

# RSI 계산
def calculate_RSI(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# MACD 계산
def calculate_MACD(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return hist

# 볼린저밴드 계산
def calculate_BB_position(series, period=20):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    position = (series - lower) / (upper - lower)
    return position

print("계산 중...")
ohlcv['RSI'] = calculate_RSI(ohlcv['close'])
ohlcv['MACD_hist'] = calculate_MACD(ohlcv['close'])
ohlcv['BB_position'] = calculate_BB_position(ohlcv['close'])
ohlcv['volume_ratio'] = ohlcv['volume'] / ohlcv['volume'].rolling(20).mean()

print("✅ 보조지표 계산 완료")

print("\n" + "=" * 80)
print("HL 전략 백테스트 실행")
print("=" * 80)

class HL_Strategy_Backtest:
    def __init__(self, ohlcv, HL_events, all_L):
        self.ohlcv = ohlcv
        self.HL_events = HL_events
        self.all_L = all_L
        self.trades = []
        self.current_position = None
        
    def get_current_L_values(self, current_time):
        """현재 시점까지의 L값만 사용 (미래 데이터 차단)"""
        past_L = self.all_L[self.all_L['datetime'] <= current_time].copy()
        if len(past_L) < 3:
            return None, None, None
        
        # 가장 최근 3개 L값
        recent_L = past_L.tail(3)
        L1 = recent_L.iloc[-1]['L_value']  # 가장 최근 (H1)
        L2 = recent_L.iloc[-2]['L_value'] if len(recent_L) >= 2 else L1  # H2
        L3 = recent_L.iloc[-3]['L_value'] if len(recent_L) >= 3 else L2  # H3
        
        return L1, L2, L3
    
    def check_HL_recently(self, current_time, lookback_hours=24):
        """최근 HL 발생 확인 (미래 데이터 차단)"""
        cutoff_time = current_time - timedelta(hours=lookback_hours)
        recent_HLs = self.HL_events[
            (self.HL_events['datetime'] > cutoff_time) & 
            (self.HL_events['datetime'] <= current_time)
        ]
        
        if len(recent_HLs) == 0:
            return None
        
        # 가장 최근 HL
        return recent_HLs.iloc[-1]
    
    def check_entry_conditions(self, candle, HL_event):
        """진입 조건 체크 (현재 시점 데이터만 사용)"""
        # 1. HL 강도 체크
        if HL_event['strength'] < 0.5:
            return False, "HL too weak"
        
        # 2. RSI 체크
        if pd.isna(candle['RSI']):
            return False, "RSI not available"
        if candle['RSI'] < 30 or candle['RSI'] > 50:
            return False, f"RSI {candle['RSI']:.1f} out of range"
        
        # 3. MACD 체크
        if pd.isna(candle['MACD_hist']):
            return False, "MACD not available"
        if candle['MACD_hist'] > 0:
            return False, f"MACD positive {candle['MACD_hist']:.2f}"
        
        # 4. BB Position 체크
        if pd.isna(candle['BB_position']):
            return False, "BB not available"
        if candle['BB_position'] > 0.5:
            return False, f"BB position {candle['BB_position']:.2f} too high"
        
        # 5. HL 발생 후 시간 체크
        hours_since_HL = (candle['datetime'] - HL_event['datetime']).total_seconds() / 3600
        
        # HL 강도별 진입 윈도우
        if HL_event['strength'] >= 5:
            # 극강: 즉시 ~ 2시간
            if hours_since_HL > 2:
                return False, f"Too late for extreme HL ({hours_since_HL:.1f}h)"
        elif HL_event['strength'] >= 2:
            # 매우강함: 0.5 ~ 3시간
            if hours_since_HL < 0.5 or hours_since_HL > 3:
                return False, f"Outside window for very strong HL ({hours_since_HL:.1f}h)"
        elif HL_event['strength'] >= 1:
            # 강함: 2 ~ 6시간
            if hours_since_HL < 2 or hours_since_HL > 6:
                return False, f"Outside window for strong HL ({hours_since_HL:.1f}h)"
        elif HL_event['strength'] >= 0.5:
            # 보통: 3 ~ 8시간
            if hours_since_HL < 3 or hours_since_HL > 8:
                return False, f"Outside window for medium HL ({hours_since_HL:.1f}h)"
        else:
            # 약함: 패스
            return False, "HL too weak"
        
        # 6. H3 돌파 체크 (확정 조건)
        L1, L2, L3 = self.get_current_L_values(candle['datetime'])
        if L3 is None:
            return False, "L values not available"
        
        if candle['close'] <= L3:
            return False, f"Price {candle['close']:.0f} below H3 {L3:.0f}"
        
        # 모든 조건 통과
        return True, "All conditions met"
    
    def calculate_targets(self, entry_price, HL_event):
        """TP/SL 계산"""
        strength = HL_event['strength']
        
        # HL 강도별 목표
        if strength >= 5:
            tp1_pct = 2.0
            tp2_pct = 4.0
        elif strength >= 2:
            tp1_pct = 1.5
            tp2_pct = 3.0
        elif strength >= 1:
            tp1_pct = 1.0
            tp2_pct = 2.0
        else:
            tp1_pct = 0.7
            tp2_pct = 1.5
        
        # SL: HL 가격 아래 1%
        sl_price = HL_event['curr_L'] * 0.99
        
        return {
            'TP1': entry_price * (1 + tp1_pct / 100),
            'TP2': entry_price * (1 + tp2_pct / 100),
            'SL': sl_price
        }
    
    def run_backtest(self):
        """백테스트 실행"""
        print("\n백테스트 진행 중...")
        
        for idx, candle in self.ohlcv.iterrows():
            current_time = candle['datetime']
            
            # 포지션 관리
            if self.current_position is not None:
                # 기존 포지션 체크
                position = self.current_position
                
                # SL 체크
                if candle['low'] <= position['SL']:
                    exit_price = position['SL']
                    pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                    
                    self.trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'exit_reason': 'SL',
                        'pnl_pct': pnl_pct,
                        'HL_strength': position['HL_strength'],
                        'hours_held': (current_time - position['entry_time']).total_seconds() / 3600
                    })
                    
                    self.current_position = None
                    continue
                
                # TP1 체크 (60% 익절)
                if not position.get('TP1_hit', False) and candle['high'] >= position['TP1']:
                    position['TP1_hit'] = True
                    position['SL'] = position['entry_price']  # 손익분기 이동
                
                # TP2 체크 (전체 청산)
                if candle['high'] >= position['TP2']:
                    # TP1 먼저 맞고 TP2 도달
                    if position.get('TP1_hit', False):
                        # 60%는 TP1에서, 40%는 TP2에서
                        avg_exit = position['TP1'] * 0.6 + position['TP2'] * 0.4
                        exit_reason = 'TP2_Full'
                    else:
                        # TP1 건너뛰고 바로 TP2
                        avg_exit = position['TP2']
                        exit_reason = 'TP2_Direct'
                    
                    pnl_pct = (avg_exit - position['entry_price']) / position['entry_price'] * 100
                    
                    self.trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': avg_exit,
                        'exit_reason': exit_reason,
                        'pnl_pct': pnl_pct,
                        'HL_strength': position['HL_strength'],
                        'hours_held': (current_time - position['entry_time']).total_seconds() / 3600
                    })
                    
                    self.current_position = None
                    continue
                
                # TP1 맞고 손익분기로 청산
                if position.get('TP1_hit', False) and candle['low'] <= position['SL']:
                    # 60%는 TP1, 40%는 손익분기
                    avg_exit = position['TP1'] * 0.6 + position['entry_price'] * 0.4
                    pnl_pct = (avg_exit - position['entry_price']) / position['entry_price'] * 100
                    
                    self.trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': avg_exit,
                        'exit_reason': 'TP1_Breakeven',
                        'pnl_pct': pnl_pct,
                        'HL_strength': position['HL_strength'],
                        'hours_held': (current_time - position['entry_time']).total_seconds() / 3600
                    })
                    
                    self.current_position = None
                    continue
            
            # 신규 진입 체크
            if self.current_position is None:
                # 최근 HL 확인
                recent_HL = self.check_HL_recently(current_time, lookback_hours=24)
                
                if recent_HL is not None:
                    # 진입 조건 체크
                    can_enter, reason = self.check_entry_conditions(candle, recent_HL)
                    
                    if can_enter:
                        # 진입!
                        targets = self.calculate_targets(candle['close'], recent_HL)
                        
                        self.current_position = {
                            'entry_time': current_time,
                            'entry_price': candle['close'],
                            'HL_strength': recent_HL['strength'],
                            'HL_time': recent_HL['datetime'],
                            'TP1': targets['TP1'],
                            'TP2': targets['TP2'],
                            'SL': targets['SL'],
                            'TP1_hit': False
                        }
        
        # 미청산 포지션 처리
        if self.current_position is not None:
            last_candle = self.ohlcv.iloc[-1]
            pnl_pct = (last_candle['close'] - self.current_position['entry_price']) / self.current_position['entry_price'] * 100
            
            self.trades.append({
                'entry_time': self.current_position['entry_time'],
                'exit_time': last_candle['datetime'],
                'entry_price': self.current_position['entry_price'],
                'exit_price': last_candle['close'],
                'exit_reason': 'Open',
                'pnl_pct': pnl_pct,
                'HL_strength': self.current_position['HL_strength'],
                'hours_held': (last_candle['datetime'] - self.current_position['entry_time']).total_seconds() / 3600
            })
        
        return pd.DataFrame(self.trades)

# 백테스트 실행
backtester = HL_Strategy_Backtest(ohlcv, HL_df, all_L)
results = backtester.run_backtest()

print(f"\n✅ 백테스트 완료!")
print(f"   총 거래: {len(results)}건")

if len(results) > 0:
    print("\n" + "=" * 80)
    print("백테스트 결과 분석")
    print("=" * 80)
    
    # 전체 성과
    total_pnl = results['pnl_pct'].sum()
    avg_pnl = results['pnl_pct'].mean()
    win_rate = (results['pnl_pct'] > 0).sum() / len(results) * 100
    
    print(f"\n📊 전체 성과:")
    print(f"   • 총 PNL: {total_pnl:.2f}%")
    print(f"   • 평균 PNL: {avg_pnl:.3f}%")
    print(f"   • 승률: {win_rate:.2f}%")
    print(f"   • 최대 수익: {results['pnl_pct'].max():.2f}%")
    print(f"   • 최대 손실: {results['pnl_pct'].min():.2f}%")
    print(f"   • 평균 보유 시간: {results['hours_held'].mean():.1f}시간")
    
    # 청산 사유별
    print(f"\n📊 청산 사유별:")
    exit_stats = results.groupby('exit_reason').agg({
        'pnl_pct': ['count', 'mean', 'sum']
    }).round(3)
    
    for exit_reason in results['exit_reason'].unique():
        reason_data = results[results['exit_reason'] == exit_reason]
        count = len(reason_data)
        avg = reason_data['pnl_pct'].mean()
        total = reason_data['pnl_pct'].sum()
        pct = count / len(results) * 100
        
        print(f"   • {exit_reason}: {count}건 ({pct:.1f}%), 평균 {avg:.3f}%, 합계 {total:.2f}%")
    
    # HL 강도별
    print(f"\n📊 HL 강도별 성과:")
    results['strength_group'] = pd.cut(
        results['HL_strength'],
        bins=[0, 0.5, 1, 2, 5, 100],
        labels=['약함(0-0.5%)', '보통(0.5-1%)', '강함(1-2%)', '매우강함(2-5%)', '극강(5%+)']
    )
    
    for group in ['약함(0-0.5%)', '보통(0.5-1%)', '강함(1-2%)', '매우강함(2-5%)', '극강(5%+)']:
        group_data = results[results['strength_group'] == group]
        if len(group_data) == 0:
            continue
        
        count = len(group_data)
        avg = group_data['pnl_pct'].mean()
        win_rate_group = (group_data['pnl_pct'] > 0).sum() / count * 100
        tp2_rate = (group_data['exit_reason'] == 'TP2_Full').sum() / count * 100
        
        print(f"   • {group}: {count}건, 평균 {avg:.3f}%, 승률 {win_rate_group:.1f}%, TP2 {tp2_rate:.1f}%")
    
    # 연도별
    results['year'] = pd.to_datetime(results['entry_time']).dt.year
    print(f"\n📊 연도별 성과:")
    for year in sorted(results['year'].unique()):
        year_data = results[results['year'] == year]
        count = len(year_data)
        avg = year_data['pnl_pct'].mean()
        total = year_data['pnl_pct'].sum()
        win_rate_year = (year_data['pnl_pct'] > 0).sum() / count * 100
        
        print(f"   • {year}: {count}건, 평균 {avg:.3f}%, 합계 {total:.2f}%, 승률 {win_rate_year:.1f}%")
    
    # 결과 저장
    results.to_csv('backtest_HL_strategy_results.csv', index=False)
    print(f"\n💾 결과 저장: backtest_HL_strategy_results.csv")
    
    # 샘플 거래 출력
    print(f"\n📋 샘플 거래 (처음 10건):")
    print(results[['entry_time', 'exit_reason', 'HL_strength', 'pnl_pct', 'hours_held']].head(10).to_string(index=False))
    
else:
    print("\n❌ 거래 없음")

print("\n" + "=" * 80)
print("백테스트 완료!")
print("=" * 80)
