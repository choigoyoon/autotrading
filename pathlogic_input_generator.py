"""
PathLogic 입력 JSON 생성기
BTC 15분봉 데이터로부터 PathLogic에 필요한 입력 JSON을 생성합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json


class HeikinAshiCalculator:
    """헤이킨아시 캔들 계산 및 추세 판단"""

    @staticmethod
    def calculate(df):
        """헤이킨아시 캔들 계산"""
        ha_df = df.copy()

        ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        ha_df['ha_open'] = 0.0
        ha_df.loc[0, 'ha_open'] = (df.loc[0, 'open'] + df.loc[0, 'close']) / 2

        for i in range(1, len(ha_df)):
            ha_df.loc[i, 'ha_open'] = (ha_df.loc[i-1, 'ha_open'] + ha_df.loc[i-1, 'ha_close']) / 2

        ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)
        ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)

        return ha_df

    @staticmethod
    def get_trend(ha_df, lookback=4):
        """헤이킨아시 추세 판단 (최근 N개 캔들 기준)"""
        recent = ha_df.tail(lookback)

        # 연속 상승/하락 캔들 수
        up_candles = (recent['ha_close'] > recent['ha_open']).sum()
        down_candles = (recent['ha_close'] < recent['ha_open']).sum()

        # 몸통 크기 평균
        body_sizes = abs(recent['ha_close'] - recent['ha_open'])
        avg_body = body_sizes.mean()

        # 추세 판단
        if up_candles >= 3:
            if avg_body > recent['ha_close'].iloc[-1] * 0.003:  # 0.3% 이상
                return "strong_up"
            else:
                return "weak_up"
        elif down_candles >= 3:
            if avg_body > recent['ha_close'].iloc[-1] * 0.003:
                return "strong_down"
            else:
                return "weak_down"
        else:
            return "side"


class MACDCalculator:
    """MACD 계산 및 H/L 라벨링"""

    @staticmethod
    def calculate(df, fast=12, slow=26, signal=9):
        """MACD 계산"""
        result = df.copy()

        # EMA 계산
        ema_fast = result['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = result['close'].ewm(span=slow, adjust=False).mean()

        result['macd'] = ema_fast - ema_slow
        result['macd_signal'] = result['macd'].ewm(span=signal, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']

        return result

    @staticmethod
    def label_hl(df):
        """MACD Histogram 기반 H/L 라벨링"""
        df = df.copy()
        df['hl_label'] = None

        for i in range(1, len(df) - 1):
            prev_hist = df['macd_hist'].iloc[i - 1]
            curr_hist = df['macd_hist'].iloc[i]
            next_hist = df['macd_hist'].iloc[i + 1]

            # High: +에서 -로 전환
            if prev_hist > 0 and curr_hist < 0:
                df.loc[df.index[i], 'hl_label'] = 'H'

            # Low: -에서 +로 전환
            elif prev_hist < 0 and curr_hist > 0:
                df.loc[df.index[i], 'hl_label'] = 'L'

        return df


class PathHintCalculator:
    """가격 움직임 패턴 판단"""

    @staticmethod
    def calculate(df, lookback=8):
        """최근 가격 움직임 패턴 분석"""
        recent = df.tail(lookback)

        # 가격 변화율
        price_change_pct = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0] * 100

        # 변동성 (표준편차)
        volatility = recent['close'].pct_change().std() * 100

        # 패턴 판단
        if abs(price_change_pct) < 1.0:  # 1% 미만 움직임
            return "no_follow"
        elif price_change_pct > 3.0 and volatility > 0.5:  # 급등
            return "fast_spike"
        elif price_change_pct < -3.0 and volatility > 0.5:  # 급락
            return "waterfall"
        else:  # 완만한 상승/하락
            return "slow_grind"


class HLStatusTracker:
    """HL 상태 추적 (intact/broken_shallow/broken_deep)"""

    @staticmethod
    def get_status(df, last_hl_idx, current_idx, position_direction='long'):
        """
        현재 HL 상태 판단

        Args:
            df: 데이터프레임
            last_hl_idx: 마지막 L(진입 기준) 인덱스
            current_idx: 현재 인덱스
            position_direction: 'long' 또는 'short'
        """
        if last_hl_idx is None:
            return "intact"

        last_hl_price = df['low'].iloc[last_hl_idx] if position_direction == 'long' else df['high'].iloc[last_hl_idx]
        current_price = df['close'].iloc[current_idx]

        if position_direction == 'long':
            # Long 포지션: Low 기준
            if current_price > last_hl_price:
                return "intact"
            else:
                # 하락 정도 측정
                break_pct = (last_hl_price - current_price) / last_hl_price * 100
                if break_pct < 0.5:  # 0.5% 미만
                    return "broken_shallow"
                else:
                    return "broken_deep"
        else:
            # Short 포지션: High 기준
            if current_price < last_hl_price:
                return "intact"
            else:
                break_pct = (current_price - last_hl_price) / last_hl_price * 100
                if break_pct < 0.5:
                    return "broken_shallow"
                else:
                    return "broken_deep"


class PathLogicInputGenerator:
    """PathLogic 입력 JSON 생성"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.positions = []  # 활성 포지션 리스트

    def load_and_prepare_data(self):
        """데이터 로드 및 전처리"""
        print("데이터 로드 중...")
        self.df = pd.read_csv(self.data_path)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])

        # MACD 계산
        print("MACD 계산 중...")
        self.df = MACDCalculator.calculate(self.df)
        self.df = MACDCalculator.label_hl(self.df)

        # 헤이킨아시 계산
        print("헤이킨아시 계산 중...")
        self.df = HeikinAshiCalculator.calculate(self.df)

        print(f"데이터 준비 완료: {len(self.df)} 캔들")
        print(f"기간: {self.df['datetime'].min()} ~ {self.df['datetime'].max()}")

    def find_entry_signals(self, start_idx=100, max_positions=10):
        """
        진입 신호 탐색 (간단한 예시)
        실제로는 더 복잡한 조건 필요:
        - 상방 추세 역전 후 HL 형성
        - H1(1시간) 돌파
        - zone 근접
        - MACD 후기 하락
        """
        print("\n진입 신호 탐색 중...")
        entries = []

        for i in range(start_idx, len(self.df) - 100):  # 미래 100캔들 확보
            # 간단한 예시: Low 발생 후 MACD hist가 상승 전환 + 가격 상승
            if self.df['hl_label'].iloc[i] == 'L':
                # Low 발생 후 4~8 캔들 내 상승 확인
                for offset in range(4, 12):
                    if i + offset >= len(self.df):
                        break

                    future_idx = i + offset
                    if (self.df['macd_hist'].iloc[future_idx] > self.df['macd_hist'].iloc[i] * 1.5 and
                        self.df['close'].iloc[future_idx] > self.df['close'].iloc[i] * 1.005):  # 0.5% 상승

                        entry_price = self.df['close'].iloc[future_idx]
                        entry_time = self.df['datetime'].iloc[future_idx]

                        # 손절가 설정: Low 가격의 -0.5%
                        stop_price = self.df['low'].iloc[i] * 0.995

                        entries.append({
                            'entry_idx': future_idx,
                            'entry_price': entry_price,
                            'entry_time': entry_time,
                            'stop_price': stop_price,
                            'low_idx': i,  # HL 추적용
                            'target_R': 3.0
                        })

                        if len(entries) >= max_positions:
                            print(f"\n발견된 진입 신호: {len(entries)}개")
                            return entries
                        break

        print(f"\n발견된 진입 신호: {len(entries)}개")
        return entries

    def generate_json_for_position(self, position, current_idx):
        """특정 포지션의 PathLogic 입력 JSON 생성"""

        # Context
        current_price = self.df['close'].iloc[current_idx]
        current_time = self.df['datetime'].iloc[current_idx]
        current_macd_hist = self.df['macd_hist'].iloc[current_idx]

        # HL 상태
        hl_status = HLStatusTracker.get_status(
            self.df,
            position['low_idx'],
            current_idx,
            'long'
        )

        # 헤이킨아시 추세
        ha_trend = HeikinAshiCalculator.get_trend(self.df.iloc[:current_idx + 1])

        # Path hint
        path_hint = PathHintCalculator.calculate(self.df.iloc[:current_idx + 1])

        # Entry
        entry_price = position['entry_price']
        stop_price = position['current_stop_price']
        risk = entry_price - stop_price

        # State
        position_size = position['current_size']
        unrealized_pnl_R = (current_price - entry_price) / risk if risk > 0 else 0
        elapsed_hours = (current_time - position['entry_time']).total_seconds() / 3600

        # History bucket (간단한 예시 - 실제로는 DB에서 조회)
        similar_patterns = [
            {
                "date": "2023-11-20",
                "outcome": "stopped_out",
                "max_R_achieved": 1.8,
                "duration_hours": 18
            },
            {
                "date": "2023-10-05",
                "outcome": "target_hit",
                "max_R_achieved": 3.2,
                "duration_hours": 48
            }
        ]

        json_input = {
            "context": {
                "current_time": current_time.isoformat(),
                "current_price": float(current_price),
                "current_macd_hist": float(current_macd_hist),
                "hl_status": hl_status,
                "heikin_ashi_trend": ha_trend,
                "path_hint": path_hint
            },
            "entry": {
                "entry_price": float(entry_price),
                "entry_time": position['entry_time'].isoformat(),
                "initial_stop_price": float(position['stop_price']),
                "target_R": position['target_R']
            },
            "state": {
                "current_position_size": float(position_size),
                "current_stop_price": float(stop_price),
                "unrealized_pnl_R": float(unrealized_pnl_R),
                "elapsed_hours": float(elapsed_hours)
            },
            "history_bucket": {
                "similar_patterns": similar_patterns
            }
        }

        return json_input

    def save_sample_inputs(self, output_path="sample_pathlogic_inputs.jsonl"):
        """샘플 입력 JSON 생성 및 저장"""

        # 진입 신호 찾기
        entries = self.find_entry_signals(max_positions=5)

        if not entries:
            print("진입 신호를 찾을 수 없습니다.")
            return

        # JSONL 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                # 포지션 초기화
                position = {
                    'entry_idx': entry['entry_idx'],
                    'entry_price': entry['entry_price'],
                    'entry_time': entry['entry_time'],
                    'stop_price': entry['stop_price'],
                    'current_stop_price': entry['stop_price'],
                    'low_idx': entry['low_idx'],
                    'target_R': entry['target_R'],
                    'current_size': 1.0
                }

                # 진입 후 10개 캔들마다 JSON 생성 (샘플)
                for offset in range(0, 100, 10):
                    current_idx = entry['entry_idx'] + offset
                    if current_idx >= len(self.df):
                        break

                    json_input = self.generate_json_for_position(position, current_idx)
                    f.write(json.dumps(json_input, ensure_ascii=False) + '\n')

        print(f"\n샘플 입력 저장 완료: {output_path}")
        print(f"총 {entries}개 포지션, 캔들당 샘플 생성")


if __name__ == "__main__":
    import glob

    # 최신 BTC 데이터 파일 찾기
    data_files = glob.glob("data/BTC_USDT_USDT_15m_*.csv")

    if not data_files:
        print("BTC 데이터 파일을 찾을 수 없습니다.")
    else:
        latest_file = max(data_files, key=lambda x: x.split('_')[-1])

        print(f"데이터 파일: {latest_file}\n")

        generator = PathLogicInputGenerator(latest_file)
        generator.load_and_prepare_data()
        generator.save_sample_inputs()
