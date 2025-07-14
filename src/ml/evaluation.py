import json
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    학습된 모델의 성능을 평가하고, 결과를 저장합니다.
    """
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 예측
    y_pred_prob: np.ndarray = model.predict(X_test)
    y_pred: np.ndarray = (y_pred_prob > 0.5).astype(int)
    
    # 기본 평가 지표
    results: Dict[str, Any] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_prob)
    }
    
    # Classification Report
    report: Dict[str, Any] = classification_report(y_test, y_pred, output_dict=True)
    results['classification_report'] = report
    
    logger.info("--- 분류 리포트 ---")
    logger.info(classification_report(y_test, y_pred))
    
    # Confusion Matrix 시각화
    cm: np.ndarray = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    _ = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=ax)
    _ = ax.set_title('Confusion Matrix')
    _ = ax.set_ylabel('Actual')
    _ = ax.set_xlabel('Predicted')
    
    cm_path = output_dir / 'confusion_matrix.png'
    fig.savefig(cm_path)
    plt.close(fig)
    logger.info(f"Confusion Matrix 저장 완료: {cm_path}")
    
    # 결과 JSON 파일로 저장
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"평가 결과 저장 완료: {results_path}")
    
    return results 