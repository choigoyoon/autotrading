#!/usr/bin/env python3
"""
HL 패턴 전략 백테스트 - 최적화 버전
필터 강화 및 타이밍 조정
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 80)
print("HL 패턴 최적화 전략 백테스트")
print("=" * 80)

# 데이터 로드
print("\n📊 데이터 로딩...")
ohlcv = pd.read_csv('btc_15m_ohlcv.csv')
ohlcv['datetime'] = pd.to_datetime(ohlcv['datetime'])
all_L = pd.read_csv('all_L_values.csv')
all_L['datetime'] = pd.to_datetime(all_L['datetime'])

print(f"✅ OHLCV: {len(ohlcv):,}개 캔들")
print(f"✅ L값: {len(all_L):,}개")

ohlcv = ohlcv.reset_index(drop=True)
all_L = all_L.reset_index(drop=True)

# HL 이벤트 생성
HL_events = []
for i in range(1, len(all_L)):
    prev_L = all_L.iloc[i-1]['L_value']
    curr_L = all_L.iloc[i]['L_value']
    
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

# 보조지표 계산
def calculate_RSI(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_MACD(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return hist

def calculate_BB_position(series, period=20):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    position = (series - lower) / (upper - lower)
    return position

print("\n보조지표 계산 중...")
ohlcv['RSI'] = calculate_RSI(ohlcv['close'])
ohlcv['MACD_hist'] = calculate_MACD(ohlcv['close'])
ohlcv['BB_position'] = calculate_BB_position(ohlcv['close'])
ohlcv['volume_ratio'] = ohlcv['volume'] / ohlcv['volume'].rolling(20).mean()
print("✅ 완료")

class HL_Strategy_Optimized:
    def __init__(self, ohlcv, HL_events, all_L):
        self.ohlcv = ohlcv
        self.HL_events = HL_events
        self.all_L = all_L
        self.trades = []
        self.current_position = None
        
    def get_current_L_values(self, current_time):
        """현재 시점까지의 L값만 사용"""
        past_L = self.all_L[self.all_L['datetime'] <= current_time].copy()
        if len(past_L) < 3:
            return None, None, None
        
        recent_L = past_L.tail(3)
        L1 = recent_L.iloc[-1]['L_value']
        L2 = recent_L.iloc[-2]['L_value'] if len(recent_L) >= 2 else L1
        L3 = recent_L.iloc[-3]['L_value'] if len(recent_L) >= 3 else L2
        
        return L1, L2, L3
    
    def check_HL_recently(self, current_time, lookback_hours=24):
        """최근 HL 발생 확인"""
        cutoff_time = current_time - timedelta(hours=lookback_hours)
        recent_HLs = self.HL_events[
            (self.HL_events['datetime'] > cutoff_time) & 
            (self.HL_events['datetime'] <= current_time)
        ]
        
        if len(recent_HLs) == 0:
            return None
        
        return recent_HLs.iloc[-1]
    
    def get_recent_candles(self, current_idx, count=5):
        """최근 캔들 가져오기"""
        start_idx = max(0, current_idx - count)
        return self.ohlcv.iloc[start_idx:current_idx]
    
    def check_entry_conditions_optimized(self, candle, candle_idx, HL_event):
        """최적화된 진입 조건 (필터 강화)"""
        
        # 1. HL 강도 필터 (약한 HL 제외)
        if HL_event['strength'] < 0.7:  # 0.5 → 0.7로 상향
            return False, "HL too weak (<0.7%)"
        
        # 2. RSI 필터 (범위 축소)
        if pd.isna(candle['RSI']):
            return False, "RSI N/A"
        if candle['RSI'] < 35 or candle['RSI'] > 48:  # 30-50 → 35-48로 축소
            return False, f"RSI {candle['RSI']:.1f} out of range"
        
        # 3. MACD 필터 (더 엄격하게)
        if pd.isna(candle['MACD_hist']):
            return False, "MACD N/A"
        if candle['MACD_hist'] > -10:  # 0 → -10으로 강화
            return False, f"MACD {candle['MACD_hist']:.2f} not negative enough"
        
        # 4. BB Position 필터
        if pd.isna(candle['BB_position']):
            return False, "BB N/A"
        if candle['BB_position'] > 0.4:  # 0.5 → 0.4로 강화
            return False, f"BB {candle['BB_position']:.2f} too high"
        
        # 5. 거래량 필터 (감소 확인)
        recent_candles = self.get_recent_candles(candle_idx, 5)
        if len(recent_candles) >= 5:
            volume_change = (recent_candles['volume'].iloc[-1] / recent_candles['volume'].mean() - 1) * 100
            if volume_change > 0:  # 거래량이 증가하면 거부
                return False, f"Volume increasing {volume_change:.1f}%"
        
        # 6. 최근 가격 움직임 필터 (하락 확인)
        if len(recent_candles) >= 5:
            price_change_5 = (recent_candles['close'].iloc[-1] / recent_candles['close'].iloc[0] - 1) * 100
            if price_change_5 > 0:  # 이미 상승 중이면 거부
                return False, f"Already rising {price_change_5:.2f}%"
        
        # 7. HL 후 타이밍 (더 보수적으로)
        hours_since_HL = (candle['datetime'] - HL_event['datetime']).total_seconds() / 3600
        
        if HL_event['strength'] >= 5:
            # 극강: 1 ~ 3시간 (0~2 → 1~3)
            if hours_since_HL < 1 or hours_since_HL > 3:
                return False, f"Outside window for extreme HL ({hours_since_HL:.1f}h)"
        elif HL_event['strength'] >= 2:
            # 매우강함: 2 ~ 5시간 (0.5~3 → 2~5)
            if hours_since_HL < 2 or hours_since_HL > 5:
                return False, f"Outside window for very strong HL ({hours_since_HL:.1f}h)"
        elif HL_event['strength'] >= 1:
            # 강함: 3 ~ 7시간 (2~6 → 3~7)
            if hours_since_HL < 3 or hours_since_HL > 7:
                return False, f"Outside window for strong HL ({hours_since_HL:.1f}h)"
        else:
            # 보통: 4 ~ 10시간 (3~8 → 4~10)
            if hours_since_HL < 4 or hours_since_HL > 10:
                return False, f"Outside window for medium HL ({hours_since_HL:.1f}h)"
        
        # 8. H3 돌파 + 여유 확인
        L1, L2, L3 = self.get_current_L_values(candle['datetime'])
        if L3 is None:
            return False, "L values N/A"
        
        # H3보다 0.5% 이상 위에 있어야 함
        if candle['close'] < L3 * 1.005:
            return False, f"Price {candle['close']:.0f} not above H3 {L3:.0f} +0.5%"
        
        # 9. 모멘텀 개선 확인 (신규 필터)
        if len(recent_candles) >= 3:
            # 최근 3캔들이 상승 추세여야 함
            last_3_closes = recent_candles['close'].tail(3).values
            if not (last_3_closes[-1] > last_3_closes[-2] and last_3_closes[-2] > last_3_closes[-3]):
                return False, "No momentum improvement"
        
        # 모든 조건 통과
        return True, "All conditions met"
    
    def calculate_targets(self, entry_price, HL_event):
        """TP/SL 계산"""
        strength = HL_event['strength']
        
        # HL 강도별 목표 (더 보수적으로)
        if strength >= 5:
            tp1_pct = 1.5  # 2.0 → 1.5
            tp2_pct = 3.0  # 4.0 → 3.0
        elif strength >= 2:
            tp1_pct = 1.2  # 1.5 → 1.2
            tp2_pct = 2.5  # 3.0 → 2.5
        elif strength >= 1:
            tp1_pct = 0.8  # 1.0 → 0.8
            tp2_pct = 1.8  # 2.0 → 1.8
        else:
            tp1_pct = 0.6  # 0.7 → 0.6
            tp2_pct = 1.3  # 1.5 → 1.3
        
        # SL: HL 가격 아래 0.8% (1% → 0.8%)
        sl_price = HL_event['curr_L'] * 0.992
        
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
                
                # TP1 체크 (70% 익절)
                if not position.get('TP1_hit', False) and candle['high'] >= position['TP1']:
                    position['TP1_hit'] = True
                    position['SL'] = position['entry_price']  # 손익분기 이동
                
                # TP2 체크
                if candle['high'] >= position['TP2']:
                    if position.get('TP1_hit', False):
                        avg_exit = position['TP1'] * 0.7 + position['TP2'] * 0.3
                        exit_reason = 'TP2_Full'
                    else:
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
                    avg_exit = position['TP1'] * 0.7 + position['entry_price'] * 0.3
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
                recent_HL = self.check_HL_recently(current_time, lookback_hours=24)
                
                if recent_HL is not None:
                    can_enter, reason = self.check_entry_conditions_optimized(candle, idx, recent_HL)
                    
                    if can_enter:
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
print("\n" + "=" * 80)
print("최적화된 HL 전략 실행")
print("=" * 80)

backtester = HL_Strategy_Optimized(ohlcv, HL_df, all_L)
results = backtester.run_backtest()

print(f"\n✅ 백테스트 완료!")
print(f"   총 거래: {len(results)}건")

if len(results) > 0:
    print("\n" + "=" * 80)
    print("최적화 전략 결과")
    print("=" * 80)
    
    total_pnl = results['pnl_pct'].sum()
    avg_pnl = results['pnl_pct'].mean()
    win_rate = (results['pnl_pct'] > 0).sum() / len(results) * 100
    
    print(f"\n📊 전체 성과:")
    print(f"   • 총 PNL: {total_pnl:.2f}%")
    print(f"   • 평균 PNL: {avg_pnl:.3f}%")
    print(f"   • 승률: {win_rate:.2f}%")
    print(f"   • 최대 수익: {results['pnl_pct'].max():.2f}%")
    print(f"   • 최대 손실: {results['pnl_pct'].min():.2f}%")
    print(f"   • 평균 보유: {results['hours_held'].mean():.1f}시간")
    
    print(f"\n📊 청산 사유별:")
    for exit_reason in results['exit_reason'].unique():
        reason_data = results[results['exit_reason'] == exit_reason]
        count = len(reason_data)
        avg = reason_data['pnl_pct'].mean()
        total = reason_data['pnl_pct'].sum()
        pct = count / len(results) * 100
        
        print(f"   • {exit_reason}: {count}건 ({pct:.1f}%), 평균 {avg:.3f}%, 합계 {total:.2f}%")
    
    print(f"\n📊 HL 강도별:")
    results['strength_group'] = pd.cut(
        results['HL_strength'],
        bins=[0, 0.5, 1, 2, 5, 100],
        labels=['약함', '보통', '강함', '매우강함', '극강']
    )
    
    for group in ['보통', '강함', '매우강함', '극강']:
        group_data = results[results['strength_group'] == group]
        if len(group_data) == 0:
            continue
        
        count = len(group_data)
        avg = group_data['pnl_pct'].mean()
        win_rate_group = (group_data['pnl_pct'] > 0).sum() / count * 100
        tp2_rate = (group_data['exit_reason'] == 'TP2_Full').sum() / count * 100
        
        print(f"   • {group}: {count}건, 평균 {avg:.3f}%, 승률 {win_rate_group:.1f}%, TP2 {tp2_rate:.1f}%")
    
    results.to_csv('backtest_HL_optimized_results.csv', index=False)
    print(f"\n💾 결과 저장: backtest_HL_optimized_results.csv")
    
    print(f"\n📋 샘플 (처음 10건):")
    print(results[['entry_time', 'exit_reason', 'HL_strength', 'pnl_pct']].head(10).to_string(index=False))

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
