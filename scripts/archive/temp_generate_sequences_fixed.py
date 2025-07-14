import pandas as pd
import numpy as np
from pathlib import Path
import joblib  # type: ignore[reportMissingTypeStubs]
from sklearn.preprocessing import MinMaxScaler  # type: ignore[reportMissingTypeStubs]
from tqdm import tqdm
from typing import cast
from numpy.typing import NDArray

def handle_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame의 인덱스를 타임존이 인식되는 UTC로 통일합니다."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    return df

def create_sequences(
    data: NDArray[np.float64], 
    feature_cols_indices: list[int], 
    label_cols_indices: list[int], 
    sequence_length: int = 60
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    데이터프레임에서 시퀀스 데이터를 생성합니다.
    """
    sequences: list[NDArray[np.float64]] = []
    labels: list[NDArray[np.float64]] = []
    
    for i in tqdm(range(len(data) - sequence_length), desc="시퀀스 생성 중", unit="seq"):
        seq: NDArray[np.float64] = data[i:i+sequence_length, feature_cols_indices]
        
        # fancy indexing with a list of indices always returns an array
        label_raw: NDArray[np.float64] = data[i + sequence_length - 1, label_cols_indices]
        label: NDArray[np.float64] = label_raw.astype(np.float64)
        
        sequences.append(seq)
        labels.append(label)
    
    sequences_array: NDArray[np.float64] = np.array(sequences, dtype=np.float64)
    labels_array: NDArray[np.float64] = np.array(labels, dtype=np.float64)
    
    return sequences_array, labels_array

def validate_data_shapes(X: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """LSTM 호환성을 위한 데이터 shape 검증 및 수정"""
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        print(f"⚠️  y shape을 LSTM 호환 형태로 변환: {y.shape}")
    
    if not (X.shape[0] == y.shape[0]):
        raise ValueError(f"샘플 수 불일치: X={X.shape[0]}, y={y.shape[0]}")
    if not X.ndim == 3:
        raise ValueError(f"X는 3차원 배열이어야 함: 현재 {X.ndim}차원")
    if not y.ndim == 2:
        raise ValueError(f"y는 2차원 배열이어야 함: 현재 {y.ndim}차원")
    
    print(f"✅ 데이터 shape 검증 완료: X={X.shape}, y={y.shape}")
    return X, y

def main() -> None:
    """메인 실행 함수"""
    print("🚀 시퀀스 데이터 생성 시작 🚀")
    
    features_path = Path('results/ml_analysis_v2/enhanced_features_dataset.parquet')
    labels_path = Path('data/processed/btc_usdt_kst/advanced_labels.parquet')
    output_dir = Path('results/sequences')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📂 데이터 로딩...")
    try:
        features_df_raw: pd.DataFrame = pd.read_parquet(features_path)
        labels_df_raw: pd.DataFrame = pd.read_parquet(labels_path)
    except FileNotFoundError as e:
        print(f"❌ 오류: 파일을 찾을 수 없습니다 - {e}")
        return
    
    print("🕰️  타임존 통일 (UTC)...")
    features_df = handle_timezone(features_df_raw)
    labels_df = handle_timezone(labels_df_raw)

    print("🔄 데이터 병합 중...")
    merged_df: pd.DataFrame = pd.merge(
        features_df,
        labels_df[["trade_type"]],
        left_index=True,
        right_index=True,
        how="inner",
    )

    data_with_dummies = pd.get_dummies(
        merged_df, columns=["type", "timeframe", "trade_type"], drop_first=True, dtype=float
    ) # type: ignore[reportUnknownMemberType]
    data: pd.DataFrame = data_with_dummies

    feature_cols: list[str] = [col for col in data.columns if "trade_type" not in col]
    label_cols: list[str] = [col for col in data.columns if "trade_type" in col]

    print(f"📊 피처 컬럼 수: {len(feature_cols)}")
    print(f"📊 라벨 컬럼 수: {len(label_cols)}")

    print("📏 데이터 정규화 중...")
    scaler = MinMaxScaler()

    scaler_input = data[feature_cols].to_numpy(dtype=np.float64)
    scaled_features = scaler.fit_transform(scaler_input)  # type: ignore[reportUnknownMemberType]

    _ = joblib.dump(scaler, output_dir / "sequence_scaler.joblib")  # type: ignore[reportUnknownMemberType]

    data_scaled = data.copy()
    data_scaled[feature_cols] = scaled_features

    feature_cols_indices: list[int] = [
        cast(int, data.columns.get_loc(c)) for c in feature_cols
    ]
    label_cols_indices: list[int] = [
        cast(int, data.columns.get_loc(c)) for c in label_cols
    ]

    sequence_length: int = 60

    data_values = data_scaled.to_numpy(dtype=np.float64)

    X, y = create_sequences(
        data_values, feature_cols_indices, label_cols_indices, sequence_length
    )
    
    x_validated, y_validated = validate_data_shapes(X, y)
    
    print(f"📊 최종 시퀀스 Shape:")
    print(f"   X (피처): {x_validated.shape} - (샘플수, 시퀀스길이, 피처수)")
    print(f"   y (라벨): {y_validated.shape} - (샘플수, 라벨수)")
    print(f"   X dtype: {x_validated.dtype}")
    print(f"   y dtype: {y_validated.dtype}")
    
    metadata: dict[str, object] = {
        'feature_cols': feature_cols,
        'label_cols': label_cols,
        'sequence_length': sequence_length,
        'scaler_features': getattr(scaler, 'n_features_in_', 0),
        'data_shape': x_validated.shape,
        'label_shape': y_validated.shape
    }
    
    print("💾 시퀀스 데이터 저장 중...")
    np.save(output_dir / "sequences_X.npy", x_validated)
    np.save(output_dir / "sequences_y.npy", y_validated)
    _ = joblib.dump(metadata, output_dir / "metadata.joblib")  # type: ignore[reportUnknownMemberType]
    
    print(f"✅ 시퀀스 데이터 생성 완료! 저장 위치: {output_dir}")
    print(f"   - sequences_X.npy: 피처 시퀀스 데이터")
    print(f"   - sequences_y.npy: 라벨 데이터")
    print(f"   - metadata.joblib: 메타데이터 (피처 컬럼 포함)")
    print(f"   - sequence_scaler.joblib: 정규화 스케일러")

if __name__ == "__main__":
    main() 