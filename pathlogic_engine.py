"""
PathLogic 포지션 관리 엔진
규칙 기반 의사결정 엔진 (LLM 시뮬레이션)
"""

import json
import numpy as np


class PathLogicEngine:
    """PathLogic 의사결정 엔진 (규칙 기반)"""

    def __init__(self, system_prompt_path="pathlogic_system_prompt.md"):
        self.system_prompt_path = system_prompt_path

    def calculate_ev(self, context, state):
        """
        기댓값(EV) 계산

        간단한 휴리스틱:
        - HL intact + 상승추세 = 높은 EV
        - HL broken_deep + 하락추세 = 낮은 EV
        """
        hl_status = context['hl_status']
        trend = context['heikin_ashi_trend']
        path_hint = context['path_hint']
        current_pnl_R = state['unrealized_pnl_R']

        # 기본 EV
        ev_hold = 0.0
        ev_exit = current_pnl_R  # 현재 수익으로 청산

        # HL 상태에 따른 가중치
        if hl_status == "intact":
            hl_weight = 1.5
        elif hl_status == "broken_shallow":
            hl_weight = 0.8
        else:  # broken_deep
            hl_weight = -0.5

        # 추세에 따른 가중치
        if trend == "strong_up":
            trend_weight = 1.2
        elif trend == "weak_up":
            trend_weight = 0.5
        elif trend == "side":
            trend_weight = 0.0
        elif trend == "weak_down":
            trend_weight = -0.5
        else:  # strong_down
            trend_weight = -1.0

        # Path hint에 따른 가중치
        if path_hint == "slow_grind":
            path_weight = 0.8  # 지속 가능
        elif path_hint == "fast_spike":
            path_weight = -0.3  # 급등 후 조정 위험
        elif path_hint == "waterfall":
            path_weight = -1.0  # 급락
        else:  # no_follow
            path_weight = 0.0

        # EV 계산
        ev_hold = current_pnl_R + (hl_weight + trend_weight + path_weight) * 0.5

        return ev_hold, ev_exit

    def decide_action(self, json_input):
        """
        PathLogic 의사결정

        Args:
            json_input: PathLogic 입력 JSON (dict)

        Returns:
            dict: {action, new_stop_R, partial_size, comment}
        """
        context = json_input['context']
        entry = json_input['entry']
        state = json_input['state']
        history = json_input['history_bucket']

        hl_status = context['hl_status']
        trend = context['heikin_ashi_trend']
        path_hint = context['path_hint']
        macd_hist = context['current_macd_hist']

        current_pnl_R = state['unrealized_pnl_R']
        elapsed_hours = state['elapsed_hours']
        target_R = entry['target_R']

        # EV 계산
        ev_hold, ev_exit = self.calculate_ev(context, state)

        # 결정 로직
        action = "hold"
        new_stop_R = None
        partial_size = None
        comment = ""

        # 1. 즉시 청산 조건
        if hl_status == "broken_deep" and trend in ["strong_down", "weak_down"]:
            action = "full_exit"
            comment = f"HL broken_deep + {trend}. 추세 전환 확정. EV(exit)={ev_exit:.2f}R 우위."

        # 2. 기회비용: 72시간 초과 + 수익 미미
        elif elapsed_hours > 72 and current_pnl_R < 0.5:
            action = "full_exit"
            comment = f"경과시간 {elapsed_hours:.1f}h 초과 + 수익 {current_pnl_R:.2f}R 미미. 기회비용 고려 청산."

        # 3. 목표 달성 시 trailing stop
        elif current_pnl_R >= target_R:
            new_stop_R = max(current_pnl_R - 1.0, target_R * 0.7)  # 현재 수익 - 1R, 최소 목표의 70%
            action = "tighten_sl"
            comment = f"목표 {target_R}R 달성 ({current_pnl_R:.2f}R). Trailing stop {new_stop_R:.2f}R로 상향."

        # 4. 목표 근접 + fast_spike: 부분익절
        elif current_pnl_R >= target_R * 0.7 and path_hint == "fast_spike":
            action = "partial_tp"
            partial_size = 0.5  # 50% 청산
            comment = f"목표 70% 달성 + fast_spike. 급등 후 조정 대비 50% 익절. 잔여로 추가 상승 추구."

        # 5. 쉐이크아웃 의심 (HL broken_shallow + weak_down)
        elif hl_status == "broken_shallow" and trend == "weak_down":
            # History bucket에서 유사 패턴 확인
            similar = history.get('similar_patterns', [])
            if similar:
                success_count = sum(1 for p in similar if p['outcome'] == 'target_hit')
                success_rate = success_count / len(similar)

                if success_rate >= 0.5:
                    action = "hold"
                    comment = f"쉐이크아웃 의심 (HL broken_shallow + weak_down). 유사 패턴 성공률 {success_rate*100:.0f}%. Hold 우선."
                else:
                    action = "tighten_sl"
                    new_stop_R = max(0, current_pnl_R * 0.5)
                    comment = f"HL broken_shallow + 유사 패턴 성공률 낮음 ({success_rate*100:.0f}%). 손절 강화."
            else:
                action = "hold"
                comment = f"HL broken_shallow. 일시적 흔들림 가능. MACD {macd_hist:.1f} 관찰."

        # 6. 안정적 상승 (HL intact + slow_grind)
        elif hl_status == "intact" and path_hint == "slow_grind":
            action = "hold"
            comment = f"HL intact + slow_grind. 지속 가능한 상승. EV(hold)={ev_hold:.2f}R > EV(exit)={ev_exit:.2f}R. 현재 stop 유지."

        # 7. HL intact이지만 약한 추세
        elif hl_status == "intact" and trend in ["side", "weak_down"]:
            # 목표 50% 이상 달성 시 부분익절 고려
            if current_pnl_R >= target_R * 0.5:
                action = "partial_tp"
                partial_size = 0.3  # 30% 청산
                comment = f"HL intact이나 추세 약화 ({trend}). 목표 50% 달성. 30% 익절로 위험 감소."
            else:
                action = "hold"
                comment = f"HL intact이나 {trend}. 추세 회복 대기. EV(hold)={ev_hold:.2f}R."

        # 8. 기본: Hold
        else:
            action = "hold"
            comment = f"현재 상태: HL {hl_status}, {trend}, {path_hint}. EV(hold)={ev_hold:.2f}R vs EV(exit)={ev_exit:.2f}R. Hold."

        return {
            "action": action,
            "new_stop_R": new_stop_R,
            "partial_size": partial_size,
            "comment": comment
        }

    def process_jsonl_file(self, input_path, output_path):
        """
        JSONL 파일을 읽어 각 입력에 대해 결정 생성

        Args:
            input_path: 입력 JSONL 파일 경로
            output_path: 출력 JSONL 파일 경로
        """
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            for line in infile:
                json_input = json.loads(line)
                decision = self.decide_action(json_input)

                # 입력 + 결정을 함께 저장
                output = {
                    'input': json_input,
                    'decision': decision
                }

                outfile.write(json.dumps(output, ensure_ascii=False) + '\n')

        print(f"처리 완료: {input_path} -> {output_path}")


if __name__ == "__main__":
    engine = PathLogicEngine()

    # 샘플 입력 처리
    input_file = "sample_pathlogic_inputs.jsonl"
    output_file = "pathlogic_decisions.jsonl"

    print("PathLogic 엔진 실행 중...\n")
    engine.process_jsonl_file(input_file, output_file)

    # 결과 샘플 출력
    print("\n=== 결정 샘플 ===")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:  # 처음 5개만
                break

            result = json.loads(line)
            decision = result['decision']
            input_data = result['input']

            print(f"\n[{i+1}] 시간: {input_data['context']['current_time']}")
            print(f"    수익: {input_data['state']['unrealized_pnl_R']:.2f}R")
            print(f"    HL: {input_data['context']['hl_status']}, 추세: {input_data['context']['heikin_ashi_trend']}")
            print(f"    결정: {decision['action']}")
            print(f"    사유: {decision['comment']}")
