from typing import Any

def create_model(input_dim: int) -> Any:
    """Keras 모델 생성 함수 (scikeras 래퍼용)"""
    Sequential = __import__('tensorflow.keras.models', fromlist=['Sequential']).Sequential
    Dense = __import__('tensorflow.keras.layers', fromlist=['Dense']).Dense
    Dropout = __import__('tensorflow.keras.layers', fromlist=['Dropout']).Dropout
    BatchNormalization = __import__('tensorflow.keras.layers', fromlist=['BatchNormalization']).BatchNormalization
    Precision = __import__('tensorflow.keras.metrics', fromlist=['Precision']).Precision
    Recall = __import__('tensorflow.keras.metrics', fromlist=['Recall']).Recall

    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
    )
    return model 