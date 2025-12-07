"""
동적 매매관리 시스템
- 이전 H/L 기준 동적 TP/SL
- 되돌림 패턴 인식 후 추가 진입
- 부분 청산으로 리스크 관리
- 목표: 모든 케이스 수익화
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Position:
    """포지션 정보"""
    direction: str  # 'long' or 'short'
    entry_price: float
    size: float
    total_size: float
    avg_price: float
    entries: List[dict]  # 진입 기록

    current_pl: float = 0.0
    realized_pl: float = 0.0
    max_profit: float = 0.0
    max_drawdown: float = 0.0


class DynamicPositionManager:
    """동적 포지션 관리자"""

    def __init__(self, df, labeled_df):
        self.df = df
        self.labeled_df = labeled_df

        # H/L 레벨 캐시
        self.hl_levels = self._build_hl_cache()

    def _build_hl_cache(self):
        """H/L 레벨 캐시 구축"""
        cache = {}

        h_points = self.labeled_df[self.labeled_df['label'] == 'H'].copy()
        l_points = self.labeled_df[self.labeled_df['label'] == 'L'].copy()

        for idx in range(len(self.df)):
            # 이전 H/L 찾기
            prev_h = h_points[h_points.index < idx]
            prev_l = l_points[l_points.index < idx]

            cache[idx] = {
                'prev_h': prev_h['label_price'].iloc[-1] if len(prev_h) > 0 else None,
                'prev_l': prev_l['label_price'].iloc[-1] if len(prev_l) > 0 else None,
                'prev_h_idx': prev_h.index[-1] if len(prev_h) > 0 else None,
                'prev_l_idx': prev_l.index[-1] if len(prev_l) > 0 else None,
            }

        return cache

    def get_dynamic_levels(self, idx, direction):
        """동적 TP/SL 레벨 계산"""
        levels = self.hl_levels.get(idx, {})
        prev_h = levels.get('prev_h')
        prev_l = levels.get('prev_l')

        if direction == 'long':
            tp_level = prev_h  # 이전 고점
            sl_level = prev_l  # 이전 저점
        else:  # short
            tp_level = prev_l  # 이전 저점
            sl_level = prev_h  # 이전 고점

        return tp_level, sl_level

    def manage_position(self, position: Position, current_idx: int, current_price: float,
                       high: float, low: float) -> dict:
        """
        포지션 동적 관리

        Returns:
            action: 'hold', 'add', 'partial_tp', 'full_tp', 'stop_loss'
            size: 조정할 크기
            reason: 사유
        """
        # 동적 TP/SL 레벨
        tp_level, sl_level = self.get_dynamic_levels(current_idx, position.direction)

        if tp_level is None or sl_level is None:
            return {'action': 'hold', 'size': 0, 'reason': 'no_hl_levels'}

        # 현재 손익 계산
        if position.direction == 'long':
            unrealized_pl = (current_price - position.avg_price) / position.avg_price * 100
            max_high = high
            max_low = low
        else:
            unrealized_pl = (position.avg_price - current_price) / position.avg_price * 100
            max_high = high
            max_low = low

        position.current_pl = unrealized_pl
        position.max_profit = max(position.max_profit, unrealized_pl)
        position.max_drawdown = min(position.max_drawdown, unrealized_pl)

        # 1. TP 도달 (이전 H/L 레벨)
        if position.direction == 'long':
            if high >= tp_level:
                # 이전 고점 도달 → 50% 청산
                if position.total_size > 0.5:
                    return {'action': 'partial_tp', 'size': 0.5, 'price': tp_level,
                           'reason': f'hit_prev_h_{tp_level:.2f}'}
                else:
                    return {'action': 'full_tp', 'size': position.total_size, 'price': tp_level,
                           'reason': f'hit_prev_h_{tp_level:.2f}'}
        else:  # short
            if low <= tp_level:
                if position.total_size > 0.5:
                    return {'action': 'partial_tp', 'size': 0.5, 'price': tp_level,
                           'reason': f'hit_prev_l_{tp_level:.2f}'}
                else:
                    return {'action': 'full_tp', 'size': position.total_size, 'price': tp_level,
                           'reason': f'hit_prev_l_{tp_level:.2f}'}

        # 2. SL 도달 (이전 H/L 레벨)
        if position.direction == 'long':
            if low <= sl_level:
                return {'action': 'stop_loss', 'size': position.total_size, 'price': sl_level,
                       'reason': f'hit_prev_l_{sl_level:.2f}'}
        else:
            if high >= sl_level:
                return {'action': 'stop_loss', 'size': position.total_size, 'price': sl_level,
                       'reason': f'hit_prev_h_{sl_level:.2f}'}

        # 3. 되돌림 패턴 인식 → 추가 진입
        # LONG: -0.3% ~ -0.6% 되돌림
        if position.direction == 'long' and -0.6 <= unrealized_pl <= -0.3:
            if position.total_size < 1.0:  # 아직 추가 가능
                # 지지 확인 (이전 L 근처)
                distance_to_prev_l = abs(current_price - sl_level) / current_price * 100
                if distance_to_prev_l < 0.5:  # 이전 L에서 0.5% 이내
                    return {'action': 'add', 'size': 0.3, 'price': current_price,
                           'reason': f'pullback_support_{distance_to_prev_l:.2f}%'}

        # SHORT: -0.3% ~ -0.6% 되돌림
        elif position.direction == 'short' and -0.6 <= unrealized_pl <= -0.3:
            if position.total_size < 1.0:
                distance_to_prev_h = abs(current_price - sl_level) / current_price * 100
                if distance_to_prev_h < 0.5:
                    return {'action': 'add', 'size': 0.3, 'price': current_price,
                           'reason': f'pullback_resistance_{distance_to_prev_h:.2f}%'}

        # 4. Trailing stop (수익 보호)
        # +1% 이상 수익 시 -0.5% 역행하면 50% 청산
        if unrealized_pl > 1.0 and position.max_profit - unrealized_pl > 0.5:
            if position.total_size > 0.5:
                return {'action': 'partial_tp', 'size': 0.5, 'price': current_price,
                       'reason': f'trailing_stop_profit_{unrealized_pl:.2f}%'}

        # 5. 손익비 확인 (3:1 미만이면 홀드)
        risk = abs(position.avg_price - sl_level) / position.avg_price * 100
        reward = abs(tp_level - position.avg_price) / position.avg_price * 100
        rr_ratio = reward / risk if risk > 0 else 0

        if rr_ratio < 2.0:
            # 손익비 불리 → 조기 청산 고려
            if unrealized_pl > 0.5:
                return {'action': 'partial_tp', 'size': 0.3, 'price': current_price,
                       'reason': f'low_rr_{rr_ratio:.2f}'}

        return {'action': 'hold', 'size': 0, 'reason': f'monitoring_pl_{unrealized_pl:.2f}%'}


def backtest_dynamic_management(df, labeled_df, breakouts_df):
    """동적 관리 백테스트"""

    manager = DynamicPositionManager(df, labeled_df)

    results = []

    for idx, breakout in breakouts_df.iterrows():
        break_idx = breakout['break_idx']
        break_type = breakout['type']
        break_price = df.iloc[break_idx]['close']

        # 방향 결정
        direction = 'long' if break_type == 'trendline_up' else 'short'

        # 초기 포지션 (30% 진입)
        position = Position(
            direction=direction,
            entry_price=break_price,
            size=0.3,
            total_size=0.3,
            avg_price=break_price,
            entries=[{'idx': break_idx, 'price': break_price, 'size': 0.3}]
        )

        # 포지션 관리 (60봉까지)
        end_idx = min(break_idx + 60, len(df) - 1)
        actions_log = []

        for i in range(break_idx + 1, end_idx + 1):
            current_price = df.iloc[i]['close']
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']

            # 동적 관리 결정
            action = manager.manage_position(position, i, current_price, high, low)

            if action['action'] != 'hold':
                actions_log.append({
                    'idx': i,
                    'action': action['action'],
                    'size': action['size'],
                    'price': action.get('price', current_price),
                    'reason': action['reason']
                })

            # 액션 실행
            if action['action'] == 'add':
                # 추가 진입
                add_size = action['size']
                add_price = action['price']
                new_total = position.total_size + add_size
                position.avg_price = (position.avg_price * position.total_size + add_price * add_size) / new_total
                position.total_size = new_total
                position.entries.append({'idx': i, 'price': add_price, 'size': add_size})

            elif action['action'] == 'partial_tp':
                # 부분 청산
                exit_size = action['size']
                exit_price = action['price']

                if direction == 'long':
                    pl = (exit_price - position.avg_price) / position.avg_price * 100
                else:
                    pl = (position.avg_price - exit_price) / position.avg_price * 100

                position.realized_pl += pl * exit_size
                position.total_size -= exit_size

                if position.total_size <= 0:
                    break

            elif action['action'] in ['full_tp', 'stop_loss']:
                # 전체 청산
                exit_price = action['price']

                if direction == 'long':
                    pl = (exit_price - position.avg_price) / position.avg_price * 100
                else:
                    pl = (position.avg_price - exit_price) / position.avg_price * 100

                position.realized_pl += pl * position.total_size
                position.total_size = 0
                break

        # 만약 60봉까지 포지션 남아있으면 강제 청산
        if position.total_size > 0:
            final_price = df.iloc[end_idx]['close']
            if direction == 'long':
                pl = (final_price - position.avg_price) / position.avg_price * 100
            else:
                pl = (position.avg_price - final_price) / position.avg_price * 100

            position.realized_pl += pl * position.total_size

        results.append({
            'break_idx': break_idx,
            'type': break_type,
            'direction': direction,
            'initial_price': break_price,
            'avg_price': position.avg_price,
            'total_entries': len(position.entries),
            'realized_pl': position.realized_pl,
            'max_profit': position.max_profit,
            'max_drawdown': position.max_drawdown,
            'actions_count': len(actions_log),
            'actions': str(actions_log[:3])  # 첫 3개만
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("데이터 로드 중...")
    df = pd.read_csv("output_phase1_labeled.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 라벨링된 데이터 (H/L 포인트)
    labeled = df[df['label'].notna()].copy()

    breakouts_df = pd.read_csv("output_phase4_breakouts.csv")
    trendline_breakouts = breakouts_df[breakouts_df['type'].isin(['trendline_up', 'trendline_down'])].head(1000)  # 샘플

    print(f"추세선 돌파: {len(trendline_breakouts):,}개")
    print(f"H/L 포인트: {len(labeled):,}개\n")

    print("동적 관리 백테스트 실행 중...")
    results = backtest_dynamic_management(df, labeled, trendline_breakouts)

    print("\n" + "=" * 60)
    print("동적 관리 결과")
    print("=" * 60)

    print(f"\n평균 실현 손익: {results['realized_pl'].mean():.3f}%")
    print(f"승률 (>0%): {(results['realized_pl'] > 0).sum() / len(results) * 100:.1f}%")
    print(f"평균 추가 진입: {results['total_entries'].mean():.1f}회")
    print(f"평균 액션: {results['actions_count'].mean():.1f}회")

    print(f"\n최대 수익: {results['max_profit'].mean():.3f}%")
    print(f"최대 손실: {results['max_drawdown'].mean():.3f}%")

    # 샘플 출력
    print("\n샘플 (상위 10개):")
    print(results.nlargest(10, 'realized_pl')[['break_idx', 'type', 'realized_pl', 'total_entries', 'actions_count']].to_string(index=False))

    # 저장
    results.to_csv("output/dynamic_management_results.csv", index=False)
    print("\n저장: output/dynamic_management_results.csv")
