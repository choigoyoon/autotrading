"""
나우캐스트 시뮬레이션 시스템
과거 데이터를 순차적으로 처리하여 실시간 예측을 시뮬레이션하고 정확도를 측정합니다.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from macd_hl_labeling import MACDHighLowLabeler, resample_to_timeframe


class NowcastSimulator:
    """나우캐스트 시뮬레이터"""

    def __init__(self):
        self.timeframe_hierarchy = ['15T', '1H', '4H', '1D', '3D', '1W']
        self.labeler = MACDHighLowLabeler()
        self.predictions = []
        self.actual_promotions = {}

    def simulate_realtime_detection(self, df_15m, target_upper_tf, lookback_window=None):
        """
        실시간 H/L 감지 시뮬레이션

        Args:
            df_15m: 15분봉 DataFrame
            target_upper_tf: 목표 상위 타임프레임
            lookback_window: 재계산 윈도우 (None이면 전체)

        Returns:
            {timestamp: {hl_15m, hl_upper}, ...}
        """
        if 'datetime' not in df_15m.columns:
            df_15m['datetime'] = pd.to_datetime(df_15m['timestamp'], unit='ms')

        # 타임프레임별 데이터 준비
        if target_upper_tf != '15T':
            df_upper = resample_to_timeframe(df_15m, target_upper_tf)
        else:
            df_upper = df_15m.copy()

        detection_history = {}
        hl_15m_history = []
        hl_upper_history = []

        # 최소 필요 데이터 포인트
        min_points_15m = 100
        min_points_upper = 50

        print(f"\n실시간 감지 시뮬레이션: 15T → {target_upper_tf}")
        print(f"총 15분봉: {len(df_15m)}")

        # 순차 처리 (최적화: 100개 간격으로 샘플링)
        for i in range(min_points_15m, len(df_15m), 100):  # 25시간 단위로 진행 (충분한 샘플)
            # 현재까지 데이터만 사용 (미래를 모르는 상태)
            current_df_15m = df_15m.iloc[:i].copy()
            current_time = current_df_15m['datetime'].iloc[-1]

            # 15분봉 H/L 감지
            hl_15m = self.labeler.label_highs_lows(current_df_15m)

            # 상위 TF H/L 감지
            if target_upper_tf != '15T':
                current_df_upper = resample_to_timeframe(current_df_15m, target_upper_tf)

                if len(current_df_upper) >= min_points_upper:
                    hl_upper = self.labeler.label_highs_lows(current_df_upper)
                else:
                    hl_upper = pd.DataFrame()
            else:
                hl_upper = hl_15m.copy()

            # 새로운 H/L 발견 체크
            if len(hl_15m) > len(hl_15m_history):
                # 새로운 15분 H/L 발견
                new_hl_15m = hl_15m.iloc[len(hl_15m_history):]

                for _, hl in new_hl_15m.iterrows():
                    # 이 H/L이 상위 TF H/L로 승격될지 예측
                    # 간단한 휴리스틱: 최근 가격 극값과 비교
                    prediction = self.predict_promotion(
                        hl, current_df_15m, current_df_upper
                    )

                    self.predictions.append({
                        'timestamp_15m': hl['timestamp'],
                        'type': hl['type'],
                        'price': hl['price'],
                        'predicted_promotion': prediction,
                        'target_tf': target_upper_tf,
                        'detection_time': current_time
                    })

                hl_15m_history = hl_15m.copy()

            if len(hl_upper) > len(hl_upper_history):
                hl_upper_history = hl_upper.copy()

            # 주기적으로 진행 상황 출력
            if i % 1000 == 0:
                progress = i / len(df_15m) * 100
                print(f"  진행: {progress:.1f}% - 15m H/L: {len(hl_15m)}, Upper H/L: {len(hl_upper)}")

        return {
            'hl_15m_final': hl_15m_history,
            'hl_upper_final': hl_upper_history,
            'num_predictions': len(self.predictions)
        }

    def predict_promotion(self, hl_15m, df_15m, df_upper, threshold_percentile=90):
        """
        H/L이 상위 TF로 승격될지 예측

        간단한 휴리스틱:
        - 최근 N 기간 내 극값인지
        - MACD 히스토그램 강도
        - 거래량

        Args:
            hl_15m: 15분 H/L 정보
            df_15m: 15분봉 DataFrame
            df_upper: 상위 TF DataFrame
            threshold_percentile: 극값 판단 백분위수

        Returns:
            예측 (True/False)
        """
        hl_price = hl_15m['price']
        hl_type = hl_15m['type']

        # 최근 데이터에서 극값 여부 확인
        recent_window = 100  # 최근 100개 봉
        recent_df = df_15m.tail(recent_window)

        if hl_type == 'H':
            percentile = np.percentile(recent_df['high'], threshold_percentile)
            is_extreme = hl_price >= percentile
        else:  # 'L'
            percentile = np.percentile(recent_df['low'], 100 - threshold_percentile)
            is_extreme = hl_price <= percentile

        # MACD 히스토그램 강도 (절댓값이 크면 강한 신호)
        hist_strength = abs(hl_15m.get('histogram', 0)) if 'histogram' in hl_15m else 0
        hist_threshold = 50  # 임계값 (데이터에 따라 조정 필요)

        # 승격 예측
        prediction = is_extreme and (hist_strength > hist_threshold or is_extreme)

        return prediction

    def match_predictions_with_actual(self, predictions_df, actual_upper_hl_df,
                                      time_window_hours=24, price_tolerance_pct=2.0):
        """
        예측과 실제 결과 매칭

        Args:
            predictions_df: 예측 DataFrame
            actual_upper_hl_df: 실제 상위 TF H/L DataFrame
            time_window_hours: 시간 윈도우
            price_tolerance_pct: 가격 허용 오차

        Returns:
            평가 결과 DataFrame
        """
        results = []

        for _, pred in predictions_df.iterrows():
            pred_time = pred['timestamp_15m']
            pred_type = pred['type']
            pred_price = pred['price']
            predicted_promotion = pred['predicted_promotion']

            # 실제로 승격되었는지 확인
            # 예측 후 time_window_hours 내에 상위 TF H/L이 발생했는지
            search_start = pred_time
            search_end = pred_time + timedelta(hours=time_window_hours)

            actual_matches = actual_upper_hl_df[
                (actual_upper_hl_df['type'] == pred_type) &
                (actual_upper_hl_df['timestamp'] >= search_start) &
                (actual_upper_hl_df['timestamp'] <= search_end)
            ]

            # 가격 허용 오차 내 매치
            if len(actual_matches) > 0:
                price_diffs = abs(actual_matches['price'] - pred_price) / pred_price * 100
                valid_matches = actual_matches[price_diffs <= price_tolerance_pct]

                if len(valid_matches) > 0:
                    actual_promotion = True
                    best_match = valid_matches.iloc[0]
                    actual_time = best_match['timestamp']
                    actual_price = best_match['price']
                else:
                    actual_promotion = False
                    actual_time = None
                    actual_price = None
            else:
                actual_promotion = False
                actual_time = None
                actual_price = None

            results.append({
                'pred_time': pred_time,
                'type': pred_type,
                'pred_price': pred_price,
                'predicted_promotion': predicted_promotion,
                'actual_promotion': actual_promotion,
                'actual_time': actual_time,
                'actual_price': actual_price,
                'correct': predicted_promotion == actual_promotion
            })

        return pd.DataFrame(results)

    def calculate_performance_metrics(self, results_df):
        """
        성능 메트릭 계산

        Args:
            results_df: match_predictions_with_actual() 결과

        Returns:
            메트릭 딕셔너리
        """
        y_true = results_df['actual_promotion'].astype(int)
        y_pred = results_df['predicted_promotion'].astype(int)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # 메트릭
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            'total_predictions': len(results_df),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }

        return metrics

    def plot_confusion_matrix(self, metrics, save_path='confusion_matrix.png'):
        """Confusion Matrix 시각화"""
        cm = metrics['confusion_matrix']

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Nowcast Prediction Confusion Matrix')
        plt.colorbar()

        classes = ['No Promotion', 'Promotion']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        # 셀에 숫자 표시
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion Matrix 저장: {save_path}")
        plt.close()


if __name__ == "__main__":
    import os
    import glob

    print("=" * 60)
    print("나우캐스트 시뮬레이션")
    print("=" * 60)

    # 데이터 로드
    data_dir = "data"
    pattern = os.path.join(data_dir, "BTC_USDT_USDT_15m_*.csv")
    files = glob.glob(pattern)

    if not files:
        print("\nBTC 데이터 파일이 없습니다.")
    else:
        latest_file = max(files, key=os.path.getmtime)
        print(f"\n데이터 로드: {latest_file}")

        df = pd.read_csv(latest_file)
        df['datetime'] = pd.to_datetime(df['datetime'])

        # 시뮬레이션 실행
        simulator = NowcastSimulator()

        # 15T → 1H 예측
        target_tf = '1H'
        results = simulator.simulate_realtime_detection(df, target_tf)

        print(f"\n시뮬레이션 완료")
        print(f"총 예측: {results['num_predictions']}")
        print(f"최종 15m H/L: {len(results['hl_15m_final'])}")
        print(f"최종 {target_tf} H/L: {len(results['hl_upper_final'])}")

        # 예측 vs 실제 매칭
        if len(simulator.predictions) > 0:
            predictions_df = pd.DataFrame(simulator.predictions)
            matched_results = simulator.match_predictions_with_actual(
                predictions_df,
                results['hl_upper_final']
            )

            # 성능 메트릭
            metrics = simulator.calculate_performance_metrics(matched_results)

            print("\n" + "=" * 60)
            print("나우캐스트 성능")
            print("=" * 60)
            print(f"정확도: {metrics['accuracy']*100:.2f}%")
            print(f"정밀도: {metrics['precision']*100:.2f}%")
            print(f"재현율: {metrics['recall']*100:.2f}%")
            print(f"F1 Score: {metrics['f1_score']:.3f}")
            print(f"\nTrue Positives: {metrics['true_positives']}")
            print(f"True Negatives: {metrics['true_negatives']}")
            print(f"False Positives: {metrics['false_positives']}")
            print(f"False Negatives: {metrics['false_negatives']}")

            # 시각화
            os.makedirs('models', exist_ok=True)
            simulator.plot_confusion_matrix(metrics, 'models/nowcast_confusion_matrix.png')
