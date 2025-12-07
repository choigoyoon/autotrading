#!/usr/bin/env python3
"""
순차적 백테스트 (Sequential Backtest)

핵심: 차트를 앞에서부터 읽으면서, 이전 매매 결과를 다음 판단에 적용
"""

import pandas as pd
import numpy as np
from datetime import datetime

class TradingState:
    """매매 상태 관리"""
    
    def __init__(self):
        self.position = None  # 현재 포지션
        self.trades_history = []  # 전체 매매 내역
        self.recent_trades = []  # 최근 10개 매매
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        
    def update_after_trade(self, trade_result):
        """매매 종료 후 상태 업데이트"""
        self.trades_history.append(trade_result)
        self.recent_trades.append(trade_result)
        
        # 최근 10개만 유지
        if len(self.recent_trades) > 10:
            self.recent_trades.pop(0)
        
        # PNL 업데이트
        pnl = trade_result['pnl_pct']
        self.total_pnl += pnl
        
        # 연속 승/패 카운트
        if pnl > 0:
            self.win_count += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.loss_count += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
    
    def get_win_rate(self):
        """최근 승률 계산"""
        if len(self.recent_trades) == 0:
            return 0.5
        
        wins = sum(1 for t in self.recent_trades if t['pnl_pct'] > 0)
        return wins / len(self.recent_trades)
    
    def should_use_strategy(self, strategy_name):
        """전략 사용 여부 판단 (이전 결과 기반)"""
        
        # 연속 3번 손실 → 보수적 전략만
        if self.consecutive_losses >= 3:
            return strategy_name == 'conservative'
        
        # 연속 3번 승리 → 공격적 전략 사용 가능
        if self.consecutive_wins >= 3:
            return strategy_name in ['aggressive', 'momentum']
        
        # 최근 승률 > 60% → 모든 전략 사용
        if self.get_win_rate() > 0.6:
            return True
        
        # 최근 승률 < 40% → 보수적 전략만
        if self.get_win_rate() < 0.4:
            return strategy_name == 'conservative'
        
        # 기본: 모든 전략 사용
        return True


class MultiStrategyBacktest:
    """여러 전략을 OR로 조합한 순차적 백테스트"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.state = TradingState()
        
        # BB 계산
        self.df['bb_middle'] = self.df['close'].rolling(window=20).mean()
        self.df['bb_std'] = self.df['close'].rolling(window=20).std()
        self.df['bb_upper'] = self.df['bb_middle'] + (self.df['bb_std'] * 2)
        self.df['bb_lower'] = self.df['bb_middle'] - (self.df['bb_std'] * 2)
        
        # RSI 계산
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))
    
    def check_conservative_signal(self, i):
        """보수적 전략: 높은 승률 중심"""
        
        if i < 1:
            return None
        
        prev = self.df.iloc[i-1]
        curr = self.df.iloc[i]
        
        # 신호 1: BB 하단 + RSI 과매도
        if (prev['close'] < prev['bb_lower'] and 
            curr['close'] > curr['open'] and
            curr['close'] > prev['close'] and
            curr['rsi'] < 30):
            
            return {
                'type': 'conservative',
                'signal': 'BB 하단 + RSI 과매도',
                'entry_price': curr['close'],
                'tp_pct': 1.0,  # 1:1
                'sl_pct': 0.8
            }
        
        return None
    
    def check_aggressive_signal(self, i):
        """공격적 전략: 큰 수익 추구"""
        
        if i < 1:
            return None
        
        prev = self.df.iloc[i-1]
        curr = self.df.iloc[i]
        
        # 신호 1: BB 상단 돌파 + RSI > 50
        if (prev['close'] < prev['bb_upper'] and 
            curr['close'] > curr['bb_upper'] and
            curr['rsi'] > 50):
            
            return {
                'type': 'aggressive',
                'signal': 'BB 상단 돌파 + RSI 강세',
                'entry_price': curr['close'],
                'tp_pct': 2.0,  # 1:2
                'sl_pct': 1.0
            }
        
        return None
    
    def check_momentum_signal(self, i):
        """모멘텀 전략: 강한 힘 포착"""
        
        if i < 1:
            return None
        
        prev = self.df.iloc[i-1]
        curr = self.df.iloc[i]
        
        # 신호 1: BB 찢고 복귀 + 큰 양봉
        candle_size = (curr['close'] - curr['open']) / curr['open'] * 100
        
        if (curr['low'] < curr['bb_lower'] and 
            curr['close'] > curr['bb_lower'] and
            curr['close'] > curr['open'] and
            candle_size > 0.3):
            
            return {
                'type': 'momentum',
                'signal': 'BB 찢고 큰 양봉',
                'entry_price': curr['close'],
                'tp_pct': 1.5,  # 1:1.5
                'sl_pct': 1.0
            }
        
        return None
    
    def manage_position(self, i):
        """현재 포지션 관리 (TP/SL 체크)"""
        
        if not self.state.position:
            return None
        
        curr = self.df.iloc[i]
        pos = self.state.position
        
        # TP 도달
        if curr['high'] >= pos['tp_price']:
            return {
                'entry_time': pos['entry_time'],
                'entry_price': pos['entry_price'],
                'exit_time': curr['datetime'],
                'exit_price': pos['tp_price'],
                'exit_type': 'TP',
                'strategy': pos['strategy'],
                'signal': pos['signal'],
                'pnl_pct': (pos['tp_price'] - pos['entry_price']) / pos['entry_price'] * 100
            }
        
        # SL 도달
        if curr['low'] <= pos['sl_price']:
            return {
                'entry_time': pos['entry_time'],
                'entry_price': pos['entry_price'],
                'exit_time': curr['datetime'],
                'exit_price': pos['sl_price'],
                'exit_type': 'SL',
                'strategy': pos['strategy'],
                'signal': pos['signal'],
                'pnl_pct': (pos['sl_price'] - pos['entry_price']) / pos['entry_price'] * 100
            }
        
        # Time Stop (24시간)
        hours_in_trade = (curr['datetime'] - pos['entry_time']).total_seconds() / 3600
        if hours_in_trade >= 24:
            return {
                'entry_time': pos['entry_time'],
                'entry_price': pos['entry_price'],
                'exit_time': curr['datetime'],
                'exit_price': curr['close'],
                'exit_type': 'Time Stop',
                'strategy': pos['strategy'],
                'signal': pos['signal'],
                'pnl_pct': (curr['close'] - pos['entry_price']) / pos['entry_price'] * 100
            }
        
        return None
    
    def run(self):
        """순차적 백테스트 실행"""
        
        print("="*80)
        print("순차적 백테스트 (Sequential Backtest)")
        print("="*80)
        print("\n차트를 앞에서부터 읽으면서 매매 실행...")
        print("이전 매매 결과를 다음 판단에 적용...\n")
        
        for i in range(20, len(self.df)):
            curr = self.df.iloc[i]
            
            # 1. 기존 포지션 관리
            if self.state.position:
                exit_result = self.manage_position(i)
                
                if exit_result:
                    # 포지션 청산
                    self.state.update_after_trade(exit_result)
                    self.state.position = None
                    
                    # 진행 상황 출력 (100개마다)
                    if len(self.state.trades_history) % 100 == 0:
                        print(f"[{curr['datetime']}] 거래 {len(self.state.trades_history)}개 완료")
                        print(f"  승률: {self.state.get_win_rate()*100:.1f}%")
                        print(f"  연속 승: {self.state.consecutive_wins}, 연속 패: {self.state.consecutive_losses}")
                        print(f"  누적 PNL: {self.state.total_pnl:.2f}%\n")
                
                continue
            
            # 2. 신호 감지 (여러 전략 OR 조합)
            signal = None
            
            # 보수적 전략 체크
            if self.state.should_use_strategy('conservative'):
                signal = self.check_conservative_signal(i)
            
            # 공격적 전략 체크 (보수적 신호 없을 때만)
            if not signal and self.state.should_use_strategy('aggressive'):
                signal = self.check_aggressive_signal(i)
            
            # 모멘텀 전략 체크 (다른 신호 없을 때만)
            if not signal and self.state.should_use_strategy('momentum'):
                signal = self.check_momentum_signal(i)
            
            # 3. 진입
            if signal:
                self.state.position = {
                    'entry_time': curr['datetime'],
                    'entry_price': signal['entry_price'],
                    'strategy': signal['type'],
                    'signal': signal['signal'],
                    'tp_price': signal['entry_price'] * (1 + signal['tp_pct'] / 100),
                    'sl_price': signal['entry_price'] * (1 - signal['sl_pct'] / 100)
                }
        
        # 결과 반환
        return pd.DataFrame(self.state.trades_history)


def main():
    # 데이터 로드 (최근 2년)
    df = pd.read_csv('btc_15m_ohlcv.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['datetime'] >= '2023-01-01'].copy().reset_index(drop=True)
    
    print(f"데이터 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"총 캔들 수: {len(df):,}\n")
    
    # 순차적 백테스트 실행
    backtest = MultiStrategyBacktest(df)
    trades_df = backtest.run()
    
    # 결과 분석
    print("\n" + "="*80)
    print("최종 결과")
    print("="*80)
    
    if len(trades_df) > 0:
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
        win_rate = winning_trades / total_trades * 100
        avg_pnl = trades_df['pnl_pct'].mean()
        total_pnl = trades_df['pnl_pct'].sum()
        
        # 월간 거래 수
        start_date = df['datetime'].min()
        end_date = df['datetime'].max()
        months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        monthly_trades = total_trades / months if months > 0 else 0
        
        print(f"\n총 거래 수: {total_trades}")
        print(f"월간 거래 수: {monthly_trades:.1f}")
        print(f"승률: {win_rate:.1f}%")
        print(f"평균 수익: {avg_pnl:.2f}%")
        print(f"누적 수익: {total_pnl:.2f}%")
        
        print("\n전략별 성과:")
        for strategy in trades_df['strategy'].unique():
            strat_trades = trades_df[trades_df['strategy'] == strategy]
            strat_wins = len(strat_trades[strat_trades['pnl_pct'] > 0])
            strat_wr = strat_wins / len(strat_trades) * 100 if len(strat_trades) > 0 else 0
            strat_avg = strat_trades['pnl_pct'].mean()
            
            print(f"\n{strategy}:")
            print(f"  거래 수: {len(strat_trades)}")
            print(f"  승률: {strat_wr:.1f}%")
            print(f"  평균 수익: {strat_avg:.2f}%")
        
        print("\n청산 타입별:")
        print(trades_df['exit_type'].value_counts())
        
        # TOP 10 수익 트레이드
        print("\n" + "="*80)
        print("TOP 10 수익 트레이드")
        print("="*80)
        
        top10 = trades_df.nlargest(10, 'pnl_pct')
        for i, (idx, trade) in enumerate(top10.iterrows(), 1):
            print(f"\n#{i}: {trade['strategy']} - {trade['signal']}")
            print(f"  진입: {trade['entry_time']} @ ${trade['entry_price']:.2f}")
            print(f"  청산: {trade['exit_time']} @ ${trade['exit_price']:.2f}")
            print(f"  수익: {trade['pnl_pct']:.2f}% ({trade['exit_type']})")
        
        # 저장
        trades_df.to_csv('sequential_backtest_trades.csv', index=False)
        print("\n" + "="*80)
        print("✅ 결과 저장: sequential_backtest_trades.csv")
        print("="*80)
    
    else:
        print("\n거래 없음")


if __name__ == "__main__":
    main()
