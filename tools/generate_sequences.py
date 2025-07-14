import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
import joblib
import warnings
import shutil

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

def generate_and_save_sequences(sequence_length: int = 240, test_size: float = 0.1, val_size: float = 0.2):
    """
    MACD 라벨을 사용하여 시퀀스를 생성하고, 시계열 순서를 유지하여 분할 후 저장합니다.
    """
    print("🚀 MACD 라벨 기반 시퀀스 생성을 시작합니다...")

    # --- 1. 프로젝트 경로 설정 ---
    project_root = Path(__file__).parent.parent
    print(f"📁 프로젝트 루트: {project_root}")
    
    # --- 2. 데이터 경로 설정 및 확인 ---
    labels_dir = project_root / 'data' / 'processed' / 'btc_usdt_kst' / 'labeled'
    ohlcv_path = project_root / 'data' / 'processed' / 'btc_usdt_kst' / 'resampled_ohlcv' / '1min.parquet'
    
    print(f"📊 라벨 디렉토리: {labels_dir}")
    print(f"📈 OHLCV 파일: {ohlcv_path}")
    
    # --- 3. 경로 존재 여부 확인 ---
    if not labels_dir.exists():
        print(f"❌ 라벨 디렉토리가 존재하지 않습니다: {labels_dir}")
        return False
    
    if not ohlcv_path.exists():
        print(f"❌ OHLCV 파일이 존재하지 않습니다: {ohlcv_path}")
        return False
    
    # --- 4. 라벨 파일 찾기 및 로드 ---
    label_files = list(labels_dir.glob('*.parquet'))
    if not label_files:
        print(f"❌ 라벨 파일이 없습니다: {labels_dir}")
        return False
    
    print(f"📋 발견된 라벨 파일: {len(label_files)}개")
    
    # 1분봉 라벨 파일을 우선 찾기
    one_min_label_file = None
    for file in label_files:
        if '1min' in file.name.lower() or 'merged' in file.name.lower():
            one_min_label_file = file
            break
    
    if one_min_label_file is None:
        # 가장 큰 파일 선택
        one_min_label_file = max(label_files, key=lambda f: f.stat().st_size)
    
    print(f"🎯 사용할 라벨 파일: {one_min_label_file.name}")
    
    # --- 5. 데이터 로드 ---
    print("📊 라벨 데이터 로드 중...")
    try:
        df_labels = pd.read_parquet(one_min_label_file)
        print(f"✅ 라벨 데이터 로드 완료: {len(df_labels):,}행")
        print(f"   라벨 컬럼: {df_labels.columns.tolist()}")
        print(f"   라벨 인덱스 타입: {type(df_labels.index)}")
        print(f"   라벨 인덱스 샘플: {df_labels.index[:3]}")
    except Exception as e:
        print(f"❌ 라벨 데이터 로드 실패: {e}")
        return False
    
    print("📈 OHLCV 데이터 로드 중...")
    try:
        df_ohlcv = pd.read_parquet(ohlcv_path)
        print(f"✅ OHLCV 데이터 로드 완료: {len(df_ohlcv):,}행")
        print(f"   OHLCV 컬럼: {df_ohlcv.columns.tolist()}")
        print(f"   OHLCV 인덱스 타입: {type(df_ohlcv.index)}")
        print(f"   OHLCV 인덱스 샘플: {df_ohlcv.index[:3]}")
    except Exception as e:
        print(f"❌ OHLCV 데이터 로드 실패: {e}")
        return False
    
    # --- 6. 인덱스 구조 분석 및 수정 ---
    print("\n🔍 인덱스 구조 분석...")
    
    # 라벨 데이터에 timestamp 컬럼이 있는 경우 인덱스로 설정
    if 'timestamp' in df_labels.columns:
        print("🔄 라벨 데이터의 timestamp 컬럼을 인덱스로 설정...")
        df_labels = df_labels.set_index('timestamp')
        print(f"   새 라벨 인덱스: {df_labels.index[:3]}")
    
    # OHLCV 데이터 인덱스가 문자열인 경우 datetime으로 변환
    if not isinstance(df_ohlcv.index, pd.DatetimeIndex):
        print("🔄 OHLCV 인덱스를 datetime으로 변환...")
        try:
            df_ohlcv.index = pd.to_datetime(df_ohlcv.index)
        except:
            # 인덱스 변환 실패시 reset_index 후 다시 시도
            df_ohlcv = df_ohlcv.reset_index()
            if 'timestamp' in df_ohlcv.columns:
                df_ohlcv = df_ohlcv.set_index('timestamp')
            elif 'index' in df_ohlcv.columns:
                df_ohlcv.index = pd.to_datetime(df_ohlcv['index'])
                df_ohlcv = df_ohlcv.drop('index', axis=1)
        print(f"   새 OHLCV 인덱스: {df_ohlcv.index[:3]}")
    
    # 라벨 데이터 인덱스도 datetime으로 변환
    if not isinstance(df_labels.index, pd.DatetimeIndex):
        print("🔄 라벨 인덱스를 datetime으로 변환...")
        try:
            df_labels.index = pd.to_datetime(df_labels.index)
        except Exception as e:
            print(f"❌ 라벨 인덱스 변환 실패: {e}")
            return False
    
    # --- 7. 1분봉 데이터만 필터링 (라벨 데이터가 다중 타임프레임인 경우) ---
    if 'timeframe' in df_labels.columns:
        print("🔄 1분봉 라벨만 필터링...")
        one_min_labels = df_labels[df_labels['timeframe'] == '1min'].copy()
        if len(one_min_labels) == 0:
            # 1min이 없으면 다른 형식 시도
            for tf in ['1m', '1Min', '1T']:
                one_min_labels = df_labels[df_labels['timeframe'] == tf].copy()
                if len(one_min_labels) > 0:
                    break
        
        if len(one_min_labels) > 0:
            df_labels = one_min_labels
            print(f"   1분봉 라벨: {len(df_labels):,}행")
        else:
            print("⚠️ 1분봉 라벨을 찾을 수 없어 전체 데이터 사용")
    
    # --- 8. 라벨 컬럼 확인 ---
    label_column = None
    possible_label_cols = ['label', 'labels', 'target', 'signal', 'position']
    
    for col in possible_label_cols:
        if col in df_labels.columns:
            label_column = col
            break
    
    if label_column is None:
        print("❌ 라벨 컬럼을 찾을 수 없습니다.")
        return False
    
    print(f"✅ 라벨 컬럼: {label_column}")
    print(f"   라벨 분포: {df_labels[label_column].value_counts().head()}")
    
    # --- 9. 피처 엔지니어링 ---
    print("🔧 피처 생성 중...")
    df_ohlcv['return'] = df_ohlcv['close'].pct_change().fillna(0)
    df_ohlcv['volatility'] = df_ohlcv['return'].rolling(20).std().fillna(0)
    df_ohlcv['volume_ma'] = df_ohlcv['volume'].rolling(20).mean().fillna(df_ohlcv['volume'])
    
    features_to_use = ['open', 'high', 'low', 'close', 'volume', 'return', 'volatility', 'volume_ma']
    available_features = [col for col in features_to_use if col in df_ohlcv.columns]
    print(f"📊 사용할 피처: {available_features}")
    
    df_features = df_ohlcv[available_features]

    # --- 10. 인덱스 매칭 ---
    print("🔄 데이터 인덱스 매칭 중...")
    print(f"   OHLCV 데이터 범위: {df_features.index.min()} ~ {df_features.index.max()}")
    print(f"   라벨 데이터 범위: {df_labels.index.min()} ~ {df_labels.index.max()}")
    
    # 공통 인덱스 찾기
    common_index = df_features.index.intersection(df_labels.index)
    print(f"📊 공통 데이터 포인트: {len(common_index):,}개")
    
    if len(common_index) == 0:
        print("❌ 공통 인덱스가 없습니다!")
        print("\n🔍 디버깅 정보:")
        print(f"   OHLCV 첫 5개 인덱스: {df_features.index[:5].tolist()}")
        print(f"   라벨 첫 5개 인덱스: {df_labels.index[:5].tolist()}")
        print(f"   OHLCV 타임존: {getattr(df_features.index, 'tz', 'None')}")
        print(f"   라벨 타임존: {getattr(df_labels.index, 'tz', 'None')}")
        
        # 타임존 제거 후 재시도
        print("\n🔄 타임존 제거 후 재시도...")
        if hasattr(df_features.index, 'tz') and df_features.index.tz is not None:
            df_features.index = df_features.index.tz_localize(None)
        if hasattr(df_labels.index, 'tz') and df_labels.index.tz is not None:
            df_labels.index = df_labels.index.tz_localize(None)
        
        common_index = df_features.index.intersection(df_labels.index)
        print(f"📊 타임존 제거 후 공통 데이터: {len(common_index):,}개")
    
    if len(common_index) < sequence_length * 2:
        print(f"❌ 데이터가 부족합니다. 최소 {sequence_length * 2}개 필요, 현재 {len(common_index)}개")
        return False
    
    # 공통 인덱스로 정렬
    df_features = df_features.loc[common_index].sort_index()
    df_labels = df_labels.loc[common_index].sort_index()
    
    print(f"✅ 최종 데이터: {len(common_index):,}개")
    print(f"   시간 범위: {common_index.min()} ~ {common_index.max()}")

    # --- 11. 데이터 정규화 ---
    print("📏 피처 데이터 정규화 중...")
    scaler = MinMaxScaler()
    df_features_scaled_values = scaler.fit_transform(df_features)
    
    # --- 12. 시퀀스 생성 ---
    print(f"🔄 과거 {sequence_length}분 길이의 시퀀스 생성 중...")
    
    feature_array = df_features_scaled_values
    label_array = df_labels[label_column].values
    
    num_sequences = len(feature_array) - sequence_length + 1
    print(f"📊 생성 가능한 시퀀스 수: {num_sequences:,}개")
    
    if num_sequences <= 0:
        print("❌ 시퀀스 생성 불가능")
        return False
    
    # --- 13. 데이터 분할 ---
    print("✂️ train/validation/test 분할 중...")
    train_end_idx = int(num_sequences * (1 - test_size - val_size))
    val_end_idx = int(num_sequences * (1 - test_size))

    datasets_info = {
        "train": (0, train_end_idx),
        "val": (train_end_idx, val_end_idx),
        "test": (val_end_idx, num_sequences)
    }
    
    print(f"📊 데이터 분할:")
    for name, (start, end) in datasets_info.items():
        print(f"   {name}: {end-start:,}개 시퀀스")

    # --- 14. 시퀀스 저장 ---
    output_base_dir = project_root / 'data' / 'sequences_macd'
    if output_base_dir.exists():
        print(f"🗑️ 기존 시퀀스 디렉토리 삭제: {output_base_dir}")
        shutil.rmtree(output_base_dir)

    for name, (start, end) in datasets_info.items():
        output_dir = output_base_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 '{name}' 데이터셋 저장 중...")
        for i in tqdm(range(start, end), desc=f"'{name}' 저장"):
            sequence = feature_array[i : i + sequence_length]
            label = label_array[i + sequence_length - 1]
            
            feature_tensor = torch.tensor(sequence, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            save_idx = i - start
            torch.save({
                'features': feature_tensor,
                'label': label_tensor,
                'timestamp': common_index[i + sequence_length - 1]
            }, output_dir / f"{save_idx}.pt")

    # --- 15. 메타데이터 저장 ---
    metadata = {
        'sequence_length': sequence_length,
        'num_features': len(available_features),
        'feature_names': available_features,
        'label_column': label_column,
        'num_sequences': {name: end-start for name, (start, end) in datasets_info.items()},
        'data_range': {
            'start': str(common_index[0]),
            'end': str(common_index[-1])
        },
        'label_distribution': df_labels[label_column].value_counts().to_dict()
    }
    
    metadata_path = output_base_dir / "metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # --- 16. Scaler 저장 ---
    scaler_path = output_base_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    
    print(f"\n✅ 시퀀스 생성 완료!")
    print(f"📁 저장 위치: {output_base_dir}")
    print(f"📊 메타데이터: {metadata_path}")
    print(f"⚖️ 정규화 Scaler: {scaler_path}")
    
    return True

if __name__ == '__main__':
    success = generate_and_save_sequences()
    if success:
        print("\n🎉 시퀀스 생성 성공!")
    else:
        print("\n❌ 시퀀스 생성 실패!")