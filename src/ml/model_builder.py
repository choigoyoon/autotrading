from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.metrics import Precision, Recall

def create_model(input_dim):
    """Keras 모델 생성 함수 (scikeras 래퍼용)"""
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