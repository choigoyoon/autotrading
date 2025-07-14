import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Keras/TensorFlow import 오류를 피하기 위해 모든 관련 타입을 Any로 처리합니다.
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.metrics import Precision, Recall
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

logger = logging.getLogger(__name__)


def build_and_compile_model(input_shape: Tuple[int, int]) -> Any:
    """
    LSTM 기반의 딥러닝 모델을 구축하고 컴파일합니다.
    (타입 힌트는 Any로 대체됨)
    """
    # Dynamic import
    Sequential = __import__('tensorflow.keras.models', fromlist=['Sequential']).Sequential
    LSTM = __import__('tensorflow.keras.layers', fromlist=['LSTM']).LSTM
    Dense = __import__('tensorflow.keras.layers', fromlist=['Dense']).Dense
    Dropout = __import__('tensorflow.keras.layers', fromlist=['Dropout']).Dropout
    Precision = __import__('tensorflow.keras.metrics', fromlist=['Precision']).Precision
    Recall = __import__('tensorflow.keras.metrics', fromlist=['Recall']).Recall

    model: Any = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(100, return_sequences=False),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', Precision(), Recall()])
    
    logger.info("--- 모델 요약 ---")
    model.summary(print_fn=logger.info)
    return model


def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any]
) -> Any:
    """
    모델을 학습시킵니다.
    (타입 힌트는 Any로 대체됨)
    """
    # Dynamic import
    EarlyStopping = __import__('tensorflow.keras.callbacks', fromlist=['EarlyStopping']).EarlyStopping
    ModelCheckpoint = __import__('tensorflow.keras.callbacks', fromlist=['ModelCheckpoint']).ModelCheckpoint
    Callback = __import__('tensorflow.keras.callbacks', fromlist=['Callback']).Callback

    model_dir = Path(config['training']['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path: Path = model_dir / 'best_model.keras'
    
    callbacks: List[Callback] = [
        EarlyStopping(monitor='val_loss', patience=config['training']['patience'], restore_best_weights=True),
        ModelCheckpoint(filepath=str(model_path), save_best_only=True, monitor='val_loss')
    ]
    
    history: Any = model.fit(
        X_train, y_train,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history 