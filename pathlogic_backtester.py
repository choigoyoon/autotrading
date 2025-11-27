"""
PathLogic 백테스트 엔진
포지션 관리 의사결정의 성능을 측정합니다.
"""

import pandas as pd
import numpy as np
from pathlogic_input_generator import PathLogicInputGenerator, HLStatusTracker
from pathlogic_engine import PathLogicEngine
from datetime import datetime
import json


class Position:
    """포지션 상태 추적"""

    def __init__(self, entry_idx, entry_price, entry_time, stop_price, low_idx, target_R=3.0):
        self.entry_idx = entry_idx
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.initial_stop_price = stop_price
        self.current_stop_price = stop_price
        self.low_idx = low_idx
        self.target_R = target_R

        self.initial_size = 1.0
        self.current_size = 1.0

        self.risk = entry_price - stop_price
        self.max_R_achieved = 0.0
        self.closed = False
        self.exit_reason = None
        self.exit_price = None
        self.exit_time = None
        self.realized_R = 0.0

        self.decisions_log = []  # PathLogic 결정 로그

    def update_stop(self, new_stop_R):
        """손절가 업데이트"""
        new_stop_price = self.entry_price + (new_stop_R * self.risk)
        self.current_stop_price = max(self.current_stop_price, new_stop_price)  # 손절은 올리기만

    def partial_exit(self, exit_ratio, exit_price, exit_time):
        """부분 청산"""
        exit_size = self.current_size * exit_ratio
        realized_R_partial = ((exit_price - self.entry_price) / self.risk) * exit_size

        self.realized_R += realized_R_partial
        self.current_size -= exit_size

        return exit_size, realized_R_partial

    def full_exit(self, exit_price, exit_time, reason):
        """전체 청산"""
        final_R = ((exit_price - self.entry_price) / self.risk) * self.current_size
        self.realized_R += final_R

        self.closed = True
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason

        return self.realized_R

    def check_stop_hit(self, current_low):
        """손절가 터치 확인"""
        return current_low <= self.current_stop_price

    def check_target_hit(self, current_high):
        """목표가 달성 확인"""
        target_price = self.entry_price + (self.target_R * self.risk)
        return current_high >= target_price


class PathLogicBacktester:
    """PathLogic 백테스트 시스템"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.generator = PathLogicInputGenerator(data_path)
        self.engine = PathLogicEngine()
        self.positions = []
        self.closed_positions = []

    def run_backtest(self, max_positions=20, start_idx=100):
        """백테스트 실행"""

        print("=" * 80)
        print("PathLogic 백테스트 시작")
        print("=" * 80)

        # 데이터 로드
        self.generator.load_and_prepare_data()

        # 진입 신호 찾기
        entries = self.generator.find_entry_signals(start_idx=start_idx, max_positions=max_positions)

        if not entries:
            print("진입 신호를 찾을 수 없습니다.")
            return

        print(f"\n진입 신호: {len(entries)}개")

        # 각 포지션 추적
        for entry_data in entries:
            position = Position(
                entry_idx=entry_data['entry_idx'],
                entry_price=entry_data['entry_price'],
                entry_time=entry_data['entry_time'],
                stop_price=entry_data['stop_price'],
                low_idx=entry_data['low_idx'],
                target_R=entry_data['target_R']
            )

            print(f"\n{'='*80}")
            print(f"포지션 진입: {position.entry_time}")
            print(f"  진입가: ${position.entry_price:.2f}")
            print(f"  손절가: ${position.current_stop_price:.2f}")
            print(f"  리스크: ${position.risk:.2f}")
            print(f"  목표: {position.target_R}R")

            # 포지션 관리 시뮬레이션
            self._manage_position(position)

            self.closed_positions.append(position)

        # 결과 분석
        self._analyze_results()

    def _manage_position(self, position):
        """단일 포지션 관리"""

        df = self.generator.df
        entry_idx = position.entry_idx

        # 진입 후 최대 200 캔들까지 추적 (50시간)
        for offset in range(0, min(200, len(df) - entry_idx)):
            current_idx = entry_idx + offset

            current_price = df['close'].iloc[current_idx]
            current_high = df['high'].iloc[current_idx]
            current_low = df['low'].iloc[current_idx]
            current_time = df['datetime'].iloc[current_idx]

            # 현재 R 계산
            current_R = (current_price - position.entry_price) / position.risk
            position.max_R_achieved = max(position.max_R_achieved, current_R)

            # 손절 확인
            if position.check_stop_hit(current_low):
                final_R = position.full_exit(position.current_stop_price, current_time, "stop_loss")
                print(f"  [{current_time}] 손절: ${position.current_stop_price:.2f}, R={final_R:.2f}")
                break

            # PathLogic 입력 생성 (매 15분마다)
            json_input = self.generator.generate_json_for_position(
                {
                    'entry_idx': position.entry_idx,
                    'entry_price': position.entry_price,
                    'entry_time': position.entry_time,
                    'stop_price': position.initial_stop_price,
                    'current_stop_price': position.current_stop_price,
                    'low_idx': position.low_idx,
                    'target_R': position.target_R,
                    'current_size': position.current_size
                },
                current_idx
            )

            # PathLogic 결정
            decision = self.engine.decide_action(json_input)
            position.decisions_log.append({
                'time': current_time,
                'decision': decision,
                'current_R': current_R
            })

            # 결정 실행
            if decision['action'] == 'full_exit':
                final_R = position.full_exit(current_price, current_time, "pathlogic_decision")
                print(f"  [{current_time}] PathLogic 청산: ${current_price:.2f}, R={final_R:.2f}")
                print(f"    사유: {decision['comment']}")
                break

            elif decision['action'] == 'tighten_sl':
                if decision['new_stop_R'] is not None:
                    position.update_stop(decision['new_stop_R'])
                    print(f"  [{current_time}] 손절 상향: {decision['new_stop_R']:.2f}R (${position.current_stop_price:.2f})")

            elif decision['action'] == 'partial_tp':
                if decision['partial_size'] is not None:
                    exit_size, partial_R = position.partial_exit(
                        decision['partial_size'],
                        current_price,
                        current_time
                    )
                    print(f"  [{current_time}] 부분익절: {decision['partial_size']*100:.0f}%, R={partial_R:.2f}")

                    if position.current_size <= 0.01:  # 거의 전부 청산
                        position.closed = True
                        position.exit_price = current_price
                        position.exit_time = current_time
                        position.exit_reason = "full_partial_exit"
                        break

        # 200 캔들 후에도 미청산이면 강제 청산
        if not position.closed:
            final_idx = min(entry_idx + 200, len(df) - 1)
            final_price = df['close'].iloc[final_idx]
            final_time = df['datetime'].iloc[final_idx]
            final_R = position.full_exit(final_price, final_time, "timeout")
            print(f"  [{final_time}] 시간 만료 청산: ${final_price:.2f}, R={final_R:.2f}")

    def _analyze_results(self):
        """백테스트 결과 분석"""

        print("\n" + "=" * 80)
        print("백테스트 결과 분석")
        print("=" * 80)

        if not self.closed_positions:
            print("청산된 포지션이 없습니다.")
            return

        # 기본 통계
        total_positions = len(self.closed_positions)
        total_R = sum(p.realized_R for p in self.closed_positions)
        avg_R = total_R / total_positions

        winners = [p for p in self.closed_positions if p.realized_R > 0]
        losers = [p for p in self.closed_positions if p.realized_R <= 0]

        win_rate = len(winners) / total_positions * 100
        avg_win = np.mean([p.realized_R for p in winners]) if winners else 0
        avg_loss = np.mean([p.realized_R for p in losers]) if losers else 0

        print(f"\n총 포지션: {total_positions}")
        print(f"승리: {len(winners)}, 패배: {len(losers)}")
        print(f"승률: {win_rate:.1f}%")
        print(f"평균 수익: {avg_win:.2f}R")
        print(f"평균 손실: {avg_loss:.2f}R")
        print(f"총 수익: {total_R:.2f}R")
        print(f"평균 R: {avg_R:.2f}R")

        # Exit 사유 분석
        exit_reasons = {}
        for p in self.closed_positions:
            reason = p.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = {'count': 0, 'total_R': 0}
            exit_reasons[reason]['count'] += 1
            exit_reasons[reason]['total_R'] += p.realized_R

        print("\n청산 사유별 분석:")
        for reason, stats in exit_reasons.items():
            avg = stats['total_R'] / stats['count']
            print(f"  {reason}: {stats['count']}건, 평균 {avg:.2f}R")

        # 상세 결과 저장
        self._save_detailed_results()

    def _save_detailed_results(self):
        """상세 결과를 CSV로 저장"""

        results = []
        for i, p in enumerate(self.closed_positions):
            results.append({
                'Position_ID': i + 1,
                'Entry_Time': p.entry_time,
                'Entry_Price': p.entry_price,
                'Exit_Time': p.exit_time,
                'Exit_Price': p.exit_price,
                'Exit_Reason': p.exit_reason,
                'Realized_R': p.realized_R,
                'Max_R_Achieved': p.max_R_achieved,
                'Duration_Hours': (p.exit_time - p.entry_time).total_seconds() / 3600,
                'Num_Decisions': len(p.decisions_log)
            })

        df_results = pd.DataFrame(results)
        output_path = "pathlogic_backtest_results.csv"
        df_results.to_csv(output_path, index=False)

        print(f"\n상세 결과 저장: {output_path}")

        # 성과 요약 저장
        summary = {
            'Total_Positions': len(self.closed_positions),
            'Winners': len([p for p in self.closed_positions if p.realized_R > 0]),
            'Losers': len([p for p in self.closed_positions if p.realized_R <= 0]),
            'Win_Rate_Pct': len([p for p in self.closed_positions if p.realized_R > 0]) / len(self.closed_positions) * 100,
            'Total_R': sum(p.realized_R for p in self.closed_positions),
            'Avg_R': np.mean([p.realized_R for p in self.closed_positions]),
            'Avg_Win_R': np.mean([p.realized_R for p in self.closed_positions if p.realized_R > 0]) if any(p.realized_R > 0 for p in self.closed_positions) else 0,
            'Avg_Loss_R': np.mean([p.realized_R for p in self.closed_positions if p.realized_R <= 0]) if any(p.realized_R <= 0 for p in self.closed_positions) else 0,
            'Max_R': max(p.realized_R for p in self.closed_positions),
            'Min_R': min(p.realized_R for p in self.closed_positions)
        }

        summary_df = pd.DataFrame([summary])
        summary_path = "pathlogic_backtest_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print(f"성과 요약 저장: {summary_path}")


if __name__ == "__main__":
    import glob

    # 최신 BTC 데이터 파일
    data_files = glob.glob("data/BTC_USDT_USDT_15m_*.csv")

    if not data_files:
        print("BTC 데이터 파일을 찾을 수 없습니다.")
    else:
        latest_file = max(data_files, key=lambda x: x.split('_')[-1])

        print(f"데이터 파일: {latest_file}\n")

        backtester = PathLogicBacktester(latest_file)
        backtester.run_backtest(max_positions=20, start_idx=100)
