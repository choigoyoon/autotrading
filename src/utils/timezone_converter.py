import pandas as pd
import glob
import os
from pathlib import Path

def convert_timezone(df: pd.DataFrame, timestamp_col: str = 'timestamp', source_tz: str = 'UTC', target_tz: str = 'Asia/Seoul') -> pd.DataFrame:
    """
    DataFrame의 타임스탬프 열 시간대를 변환합니다.

    Args:
        df (pd.DataFrame): 변환할 데이터프레임.
        timestamp_col (str): 타임스탬프 열 이름.
        source_tz (str): 원본 시간대 (예: 'UTC', 'Asia/Seoul').
        target_tz (str): 대상 시간대 (예: 'Asia/Seoul', 'UTC').

    Returns:
        pd.DataFrame: 시간대가 변환된 데이터프레임.
    """
    df = df.copy()
    
    # 타임스탬프 열이 datetime 타입이 아니면 변환
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # 원본 시간대 설정 (타임존 정보가 없는 경우)
    if df[timestamp_col].dt.tz is None:
        df[timestamp_col] = df[timestamp_col].dt.tz_localize(source_tz)
    
    # 대상 시간대로 변환
    df[timestamp_col] = df[timestamp_col].dt.tz_convert(target_tz)
    
    # 시간대 정보를 제거하고 naive datetime으로 만듦
    df[timestamp_col] = df[timestamp_col].dt.tz_localize(None)
    
    return df

def utc_to_kst(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """UTC 시간을 KST로 변환하고 시간대 컬럼을 'kst'로 업데이트/생성합니다."""
    df_converted = convert_timezone(df, timestamp_col, 'UTC', 'Asia/Seoul')
    
    # 기존 시간대 컬럼(utc, timezone)을 찾아 'kst'로 변경
    for col in ['utc', 'timezone']:
        if col in df_converted.columns:
            df_converted = df_converted.rename(columns={col: 'kst'})
            break
            
    df_converted['kst'] = 'KST'
    return df_converted

def kst_to_utc(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """KST 시간을 UTC로 변환하고 시간대 컬럼을 'utc'로 업데이트/생성합니다."""
    df_converted = convert_timezone(df, timestamp_col, 'Asia/Seoul', 'UTC')
    
    # 기존 시간대 컬럼(kst, timezone)을 찾아 'utc'로 변경
    for col in ['kst', 'timezone']:
        if col in df_converted.columns:
            df_converted = df_converted.rename(columns={col: 'utc'})
            break

    df_converted['utc'] = 'UTC'
    return df_converted

def auto_convert_timezone(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    데이터프레임의 시간대를 자동으로 감지하여 반대 시간대로 변환합니다.
    'utc', 'kst', 'timezone' 컬럼을 기준으로 변환 방향을 결정하고 컬럼명을 교체합니다.

    Args:
        df (pd.DataFrame): 변환할 데이터프레임.
        timestamp_col (str): 타임스탬프 열 이름.

    Returns:
        pd.DataFrame: 시간대가 변환된 데이터프레임.
    """
    df = df.copy()
    
    # 컬럼 이름 또는 값으로 UTC/KST 감지
    is_utc = 'utc' in df.columns or ('timezone' in df.columns and not df['timezone'].empty and str(df['timezone'].iloc[0]).upper() == 'UTC')
    is_kst = 'kst' in df.columns or ('timezone' in df.columns and not df['timezone'].empty and str(df['timezone'].iloc[0]).upper() == 'KST')

    if is_utc:
        print("🕐 감지된 시간대: UTC. KST로 변환합니다.")
        df_converted = utc_to_kst(df, timestamp_col)
    elif is_kst:
        print("🕐 감지된 시간대: KST. UTC로 변환합니다.")
        df_converted = kst_to_utc(df, timestamp_col)
    else:
        # 기본 동작: 시간대 정보가 불명확하면 KST로 간주하고 UTC로 변환
        print("🕐 시간대를 명확히 감지할 수 없습니다. KST로 가정하고 UTC로 변환합니다.")
        df_converted = kst_to_utc(df, timestamp_col)
            
    return df_converted

def batch_convert_timezone(
    input_dir: str,
    output_dir: str,
    timestamp_col: str = 'timestamp'
):
    """
    특정 디렉토리의 모든 CSV/Parquet 파일의 시간대를 일괄 변환합니다.
    파일 내용을 기반으로 변환 방향을 자동 감지하고,
    파일 이름에 '_utc_' 또는 '_kst_'를 포함하여 저장합니다.

    Args:
        input_dir (str): 입력 파일이 있는 디렉토리 경로.
        output_dir (str): 변환된 파일을 저장할 디렉토리 경로.
        timestamp_col (str): 타임스탬프 열 이름.
    """
    # 출력 디렉토리가 없으면 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # CSV와 Parquet 파일 모두 처리
    files = glob.glob(os.path.join(input_dir, '*.csv')) + glob.glob(os.path.join(input_dir, '*.parquet'))

    if not files:
        print(f"⚠️ {input_dir} 에서 처리할 파일을 찾지 못했습니다.")
        return

    for file_path in files:
        try:
            print(f"🔄 처리 중: {file_path}")
            
            # 파일 확장자에 따라 읽기
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_parquet(file_path)
            
            # --- 목표 파일명 생성 및 존재 여부 확인 ---
            is_utc = 'utc' in df.columns or ('timezone' in df.columns and not df['timezone'].empty and str(df['timezone'].iloc[0]).upper() == 'UTC')
            tz_tag = 'kst' if is_utc else 'utc'

            base_name, ext = os.path.splitext(os.path.basename(file_path))
            parts = base_name.split('_')
            clean_parts = [p for p in parts if p.lower() not in ['utc', 'kst']]
            
            if len(clean_parts) > 0:
                clean_parts.insert(1, tz_tag)
            else:
                clean_parts.append(tz_tag)

            new_base_name = '_'.join(clean_parts)
            new_file_name = f"{new_base_name}{ext}"
            new_path = os.path.join(output_dir, new_file_name)
            
            # 파일이 이미 존재하면 건너뛰기
            if os.path.exists(new_path):
                print(f"⏩ 건너뛰기: {new_path} 파일이 이미 존재합니다.")
                continue
            
            # --- 시간대 변환 및 저장 ---
            if is_utc:
                print(f"  -> UTC 감지. KST로 변환합니다.")
                df_converted = utc_to_kst(df, timestamp_col)
            else: # KST 또는 미지정이면 UTC로 변환
                print(f"  -> KST (또는 미지정) 감지. UTC로 변환합니다.")
                df_converted = kst_to_utc(df, timestamp_col)

            # 새 파일명으로 저장
            if file_path.endswith('.csv'):
                df_converted.to_csv(new_path, index=False)
            elif file_path.endswith('.parquet'):
                df_converted.to_parquet(new_path, index=False)
            
            print(f"✅ 완료: {new_path}")
        except Exception as e:
            print(f"🚨 오류 발생 ({file_path}): {e}")


if __name__ == '__main__':
    # --- 사용 예시 ---
    # 이 스크립트를 직접 실행하면 아래 로직이 동작합니다.
    # 파일 내용을 자동 감지하여 시간대를 변환하고, 파일명에 '_kst_' 또는 '_utc_'를 추가하여 저장합니다.
    
    # 1. CSV 파일 변환 (data/rwa/csv -> data/rwa/csv_converted)
    print("\n--- CSV 파일 시간대 자동 변환 시작 ---")
    batch_convert_timezone(
        input_dir='data/rwa/csv',
        output_dir='data/rwa/csv_converted'
    )
    
    # 2. Parquet 파일 변환 (data/rwa/parquet -> data/rwa/parquet_converted)
    print("\n--- Parquet 파일 시간대 자동 변환 시작 ---")
    batch_convert_timezone(
        input_dir='data/rwa/parquet',
        output_dir='data/rwa/parquet_converted'
    )

    print("\n🎉 모든 작업이 완료되었습니다.") 