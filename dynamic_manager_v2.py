"""
동적 매매관리 V2
- 이전 H/L은 참고만, TP/SL은 고정+동적 조합
- 되돌림 패턴 적극 활용
- 목표: 2% 수익 달성, 모든 케이스 수익화
"""

import pandas as pd
import numpy as np

def dynamic_trade_v2(df, labeled_df, breakouts_df, sample_size=None):
    """
    개선된 동적 매매 관리

    전략:
    1. 초기 진입: 30%
    2. 되돌림 패턴 인식:
       - 바로 가는 경우 (5봉 내 +0.3%): 추가 40% 진입
       - 되돌림 경우 (-0.4% ~ -0.8%): 지지 확인 후 추가 40%
    3. TP: +1.5% (또는 +2.0% trailing)
    4. SL: 동적
       - 바로 가는 경우: -0.6%
       - 되돌림 경우: -1.2% (되돌림 허용)
    5. 부분 청산: +1.0% 시 50%, +1.5% 시 나머지
    """

    if sample_size:
        breakouts_df = breakouts_df.head(sample_size)

    results = []

    for idx, breakout in breakouts_df.iterrows():
        break_idx = breakout['break_idx']
        break_type = breakout['type']
        break_price = df.iloc[break_idx]['close']

        direction = 'long' if break_type == 'trendline_up' else 'short'

        # 포지션 상태
        position = {
            'total_size': 0.3,  # 초기 30%
            'avg_price': break_price,
            'entries': [{'bar': 0, 'price': break_price, 'size': 0.3}],
            'exits': [],
            'pattern': None,  # 'direct' or 'pullback'
            'max_profit': 0,
            'max_dd': 0,
        }

        # 60봉 관리
        end_idx = min(break_idx + 60, len(df) - 1)

        for i in range(break_idx + 1, end_idx + 1):
            bar_num = i - break_idx
            current = df.iloc[i]['close']
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']

            # 현재 손익
            if direction == 'long':
                pl = (current - position['avg_price']) / position['avg_price'] * 100
                high_pl = (high - position['avg_price']) / position['avg_price'] * 100
                low_pl = (low - position['avg_price']) / position['avg_price'] * 100
            else:
                pl = (position['avg_price'] - current) / position['avg_price'] * 100
                high_pl = (position['avg_price'] - low) / position['avg_price'] * 100
                low_pl = (position['avg_price'] - high) / position['avg_price'] * 100

            position['max_profit'] = max(position['max_profit'], high_pl)
            position['max_dd'] = min(position['max_dd'], low_pl)

            # 패턴 인식 (첫 5봉)
            if bar_num <= 5 and position['pattern'] is None:
                if pl > 0.3:
                    position['pattern'] = 'direct'  # 바로 가는 패턴
                elif pl < -0.3:
                    position['pattern'] = 'pullback'  # 되돌림 패턴

            # === TP 로직 ===
            # 1차 TP: +1.0% → 50% 청산
            if high_pl >= 1.0 and position['total_size'] >= 0.5:
                if not any(e.get('reason') == 'tp1_1.0%' for e in position['exits']):
                    position['exits'].append({
                        'bar': bar_num,
                        'price': position['avg_price'] * (1.01 if direction == 'long' else 0.99),
                        'size': 0.5,
                        'reason': 'tp1_1.0%'
                    })
                    position['total_size'] -= 0.5

            # 2차 TP: +1.5% → 전체 청산
            if high_pl >= 1.5:
                exit_price = position['avg_price'] * (1.015 if direction == 'long' else 0.985)
                position['exits'].append({
                    'bar': bar_num,
                    'price': exit_price,
                    'size': position['total_size'],
                    'reason': 'tp2_1.5%'
                })
                position['total_size'] = 0
                break

            # === SL 로직 (패턴별) ===
            if position['pattern'] == 'direct':
                # 바로 가는 경우: -0.6% 손절
                if low_pl <= -0.6:
                    sl_price = position['avg_price'] * (0.994 if direction == 'long' else 1.006)
                    position['exits'].append({
                        'bar': bar_num,
                        'price': sl_price,
                        'size': position['total_size'],
                        'reason': 'sl_direct_-0.6%'
                    })
                    position['total_size'] = 0
                    break

            elif position['pattern'] == 'pullback':
                # 되돌림 경우: -1.2% 까지 허용
                if low_pl <= -1.2:
                    sl_price = position['avg_price'] * (0.988 if direction == 'long' else 1.012)
                    position['exits'].append({
                        'bar': bar_num,
                        'price': sl_price,
                        'size': position['total_size'],
                        'reason': 'sl_pullback_-1.2%'
                    })
                    position['total_size'] = 0
                    break

            else:
                # 패턴 미확정: -0.8% 손절
                if low_pl <= -0.8 and bar_num > 10:
                    sl_price = position['avg_price'] * (0.992 if direction == 'long' else 1.008)
                    position['exits'].append({
                        'bar': bar_num,
                        'price': sl_price,
                        'size': position['total_size'],
                        'reason': 'sl_unknown_-0.8%'
                    })
                    position['total_size'] = 0
                    break

            # === 추가 진입 로직 ===
            if position['total_size'] < 0.8:  # 아직 추가 가능

                # 바로 가는 패턴: 5봉 내 +0.3% 확인되면 추가 진입
                if position['pattern'] == 'direct' and bar_num <= 6:
                    if pl > 0.3 and not any(e.get('reason') == 'add_direct' for e in position['entries']):
                        add_size = 0.4
                        new_total = position['total_size'] + add_size
                        position['avg_price'] = (position['avg_price'] * position['total_size'] + current * add_size) / new_total
                        position['total_size'] = new_total
                        position['entries'].append({
                            'bar': bar_num,
                            'price': current,
                            'size': add_size,
                            'reason': 'add_direct'
                        })

                # 되돌림 패턴: -0.4% ~ -0.8% 에서 지지 확인 시 추가
                elif position['pattern'] == 'pullback' and -0.8 <= pl <= -0.4:
                    # 지지 확인: 3봉 연속 상승
                    if bar_num >= 3:
                        last_3_closes = [df.iloc[i-j]['close'] for j in range(3)]
                        if direction == 'long':
                            support_confirmed = all(last_3_closes[j] > last_3_closes[j+1] for j in range(2))
                        else:
                            support_confirmed = all(last_3_closes[j] < last_3_closes[j+1] for j in range(2))

                        if support_confirmed and not any(e.get('reason') == 'add_pullback' for e in position['entries']):
                            add_size = 0.4
                            new_total = position['total_size'] + add_size
                            position['avg_price'] = (position['avg_price'] * position['total_size'] + current * add_size) / new_total
                            position['total_size'] = new_total
                            position['entries'].append({
                                'bar': bar_num,
                                'price': current,
                                'size': add_size,
                                'reason': 'add_pullback'
                            })

        # 60봉 종료 시 남은 포지션 강제 청산
        if position['total_size'] > 0:
            final_price = df.iloc[end_idx]['close']
            position['exits'].append({
                'bar': 60,
                'price': final_price,
                'size': position['total_size'],
                'reason': 'timeout_60bar'
            })

        # 실현 손익 계산
        realized_pl = 0
        for entry in position['entries']:
            entry_pl = 0
            remaining_size = entry['size']

            for exit in position['exits']:
                if remaining_size <= 0:
                    break

                exit_size = min(remaining_size, exit['size'])

                if direction == 'long':
                    pl_pct = (exit['price'] - entry['price']) / entry['price'] * 100
                else:
                    pl_pct = (entry['price'] - exit['price']) / entry['price'] * 100

                entry_pl += pl_pct * exit_size
                remaining_size -= exit_size

            realized_pl += entry_pl

        results.append({
            'break_idx': break_idx,
            'type': break_type,
            'pattern': position['pattern'] or 'unknown',
            'entries': len(position['entries']),
            'exits': len(position['exits']),
            'realized_pl': realized_pl,
            'max_profit': position['max_profit'],
            'max_dd': position['max_dd'],
            'exit_reason': position['exits'][-1]['reason'] if position['exits'] else 'none'
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("데이터 로드 중...")
    df = pd.read_csv("output_phase1_labeled.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])

    labeled = df[df['label'].notna()].copy()

    breakouts_df = pd.read_csv("output_phase4_breakouts.csv")
    trendline_breakouts = breakouts_df[breakouts_df['type'].isin(['trendline_up', 'trendline_down'])]

    print(f"추세선 돌파: {len(trendline_breakouts):,}개\n")

    print("동적 관리 V2 백테스트 실행 중...")
    results = dynamic_trade_v2(df, labeled, trendline_breakouts, sample_size=None)

    print("\n" + "=" * 60)
    print("동적 관리 V2 결과")
    print("=" * 60)

    print(f"\n전체: {len(results):,}개")
    print(f"평균 실현 손익: {results['realized_pl'].mean():.3f}%")
    print(f"승률 (>0%): {(results['realized_pl'] > 0).sum() / len(results) * 100:.1f}%")
    print(f"승률 (>0.5%): {(results['realized_pl'] > 0.5).sum() / len(results) * 100:.1f}%")
    print(f"승률 (>1.0%): {(results['realized_pl'] > 1.0).sum() / len(results) * 100:.1f}%")

    print(f"\n평균 진입: {results['entries'].mean():.1f}회")
    print(f"평균 청산: {results['exits'].mean():.1f}회")

    print(f"\n평균 최대 수익: {results['max_profit'].mean():.3f}%")
    print(f"평균 최대 손실: {results['max_dd'].mean():.3f}%")

    # 패턴별
    print("\n" + "=" * 60)
    print("패턴별 분석")
    print("=" * 60)

    for pattern in ['direct', 'pullback', 'unknown']:
        pattern_data = results[results['pattern'] == pattern]
        if len(pattern_data) > 0:
            print(f"\n[{pattern}] {len(pattern_data):,}개 ({len(pattern_data)/len(results)*100:.1f}%)")
            print(f"  평균 손익: {pattern_data['realized_pl'].mean():.3f}%")
            print(f"  승률 >0%: {(pattern_data['realized_pl'] > 0).sum() / len(pattern_data) * 100:.1f}%")
            print(f"  승률 >1%: {(pattern_data['realized_pl'] > 1.0).sum() / len(pattern_data) * 100:.1f}%")

    # 청산 사유별
    print("\n" + "=" * 60)
    print("청산 사유별")
    print("=" * 60)

    exit_reasons = results['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        reason_data = results[results['exit_reason'] == reason]
        avg_pl = reason_data['realized_pl'].mean()
        print(f"{reason:>20}: {count:>5}개 ({count/len(results)*100:>5.1f}%) | 평균 {avg_pl:>6.3f}%")

    # 상위/하위
    print("\n상위 10개:")
    print(results.nlargest(10, 'realized_pl')[['break_idx', 'pattern', 'realized_pl', 'max_profit', 'exit_reason']].to_string(index=False))

    print("\n하위 10개:")
    print(results.nsmallest(10, 'realized_pl')[['break_idx', 'pattern', 'realized_pl', 'max_dd', 'exit_reason']].to_string(index=False))

    # 저장
    results.to_csv("output/dynamic_v2_results.csv", index=False)
    print("\n저장: output/dynamic_v2_results.csv")
