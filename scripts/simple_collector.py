# simple_collector.py
#
# Description: Script for collecting and managing BTC/USDT 1-minute OHLCV data.
# Note: All timestamps in CSV files are stored as timezone-naive datetime strings,
#       but they represent UTC time. Parquet files store timezone-aware UTC timestamps.

import ccxt  # type: ignore[reportMissingTypeStubs]
import pandas as pd
import argparse
import os
from pathlib import Path
import time
import sys
import shutil
from datetime import datetime
from typing import List, Any, Union, cast

def collect_all_ohlcv(symbol: str, timeframe: str, since_timestamp: int) -> pd.DataFrame:
    """
    지정된 시점부터 현재까지의 모든 OHLCV 데이터를 반복적으로 수집합니다.
    
    Args:
        symbol (str): 수집할 심볼.
        timeframe (str): 데이터 타임프레임.
        since_timestamp (int): 수집 시작 시점 (millisecond timestamp).

    Returns:
        pd.DataFrame: 수집된 전체 OHLCV 데이터프레임.
    """
    binance = ccxt.binance()
    all_data = []
    limit = 1000
    
    start_date = pd.to_datetime(since_timestamp, unit='ms')
    print(f"'{symbol}' '{timeframe}' 데이터 수집을 시작합니다. 시작 시점: {start_date}")
    
    while True:
        try:
            ohlcv_data = binance.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=limit)
            ohlcv = cast(List[List[Any]], ohlcv_data)
            
            if not ohlcv:
                print("더 이상 가져올 데이터가 없어 수집을 종료합니다.")
                break

            first_ts = pd.to_datetime(ohlcv[0][0], unit='ms')
            last_ts = pd.to_datetime(ohlcv[-1][0], unit='ms')
            print(f"  - 수집된 캔들: {len(ohlcv)}개 (from: {first_ts}, to: {last_ts})")

            all_data.extend(ohlcv)
            since_timestamp = int(ohlcv[-1][0]) + 1  # 다음 요청 시점 설정

            if len(ohlcv) < limit:
                print("가장 최신 데이터까지 수집 완료했습니다.")
                break
            
            time.sleep(binance.rateLimit / 1000)  # API 요청 제한 존중

        except Exception as e:
            print(f"API 요청 중 오류 발생: {e}. 5초 후 재시도합니다...", file=sys.stderr)
            time.sleep(5)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])  # type: ignore
    df = df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'})
    
    # timestamp(ms)를 시간대 정보가 없는(naive) datetime 객체로 변환 (UTC 기준)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def update_data(existing_df: pd.DataFrame, new_data_df: pd.DataFrame) -> pd.DataFrame:
    """기존 데이터에 새로운 데이터를 추가하고 중복을 제거합니다."""
    if existing_df.empty:
        return new_data_df
    
    # 두 데이터프레임의 'timestamp'를 기준으로 병합
    combined_df = pd.concat([existing_df, new_data_df])
    
    # 'timestamp' 기준으로 중복 제거 (마지막 값 유지)
    combined_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
    
    # 'timestamp' 기준으로 정렬
    combined_df.sort_values(by='timestamp', inplace=True)
    
    return combined_df

def recover_from_parquet(parquet_path: Path, csv_path: Path):
    """Parquet 파일에서 데이터를 읽어 CSV 파일로 복구합니다."""
    if not parquet_path.exists():
        print(f"오류: 복구할 Parquet 파일이 없습니다: '{parquet_path}'", file=sys.stderr)
        sys.exit(1)
        
    try:
        print(f"Parquet 파일에서 데이터 로딩 중: '{parquet_path}'")
        df = pd.read_parquet(parquet_path)
        
        # Parquet는 UTC-aware 타임스탬프를 저장하므로, naive UTC로 변환하여 CSV에 저장
        if 'timestamp' not in df.columns:
            df.reset_index(inplace=True)
        
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_convert(None)
        
        print(f"'{csv_path}' 파일에 데이터 저장 중...")
        # df의 'timestamp'는 naive datetime 객체이며, to_csv는 'YYYY-MM-DD HH:MM:SS'로 변환
        df.to_csv(csv_path, index=False)
        print(f"성공: '{parquet_path}'의 데이터를 '{csv_path}'로 복구했습니다.")
        
    except Exception as e:
        print(f"Parquet 파일 복구 중 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)

def safe_update_csv(csv_path: Path, symbol: str, timeframe: str):
    """기존 CSV 파일을 백업하고 마지막 시점부터 데이터를 이어붙입니다."""
    if not csv_path.exists():
        print(f"오류: 업데이트할 원본 CSV 파일이 없습니다: '{csv_path}'", file=sys.stderr)
        print("팁: 먼저 '--recover' 옵션을 사용해 Parquet에서 복구하거나, 최초 데이터 수집을 실행하세요.")
        sys.exit(1)

    # 1. 백업 생성
    backup_filename = f"{csv_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{csv_path.suffix}"
    backup_path = csv_path.with_name(backup_filename)
    try:
        shutil.copy(csv_path, backup_path)
        print(f"안전 백업 생성 완료: '{backup_path}'")
    except Exception as e:
        print(f"백업 파일 생성 중 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. 마지막 시점 확인
    print(f"기존 데이터 로딩 중: '{csv_path}'")
    # CSV를 읽을 때 timestamp 컬럼을 datetime 객체로 바로 파싱 (naive datetime이 됨)
    existing_data = pd.read_csv(csv_path, parse_dates=['timestamp'])
    
    last_timestamp: int
    if existing_data.empty:
        # 바이낸스 BTC/USDT 현물 거래 시작 시점 (UTC)
        since_timestamp = int(pd.to_datetime('2017-08-17 04:00:00').timestamp() * 1000)
        print("기존 데이터가 비어있습니다. 2017-08-17 04:00:00 (UTC) 부터 데이터 수집을 시작합니다.")
        last_timestamp = since_timestamp
    else:
        # 마지막 naive datetime 객체를 가져옴
        last_dt_naive = existing_data['timestamp'].iloc[-1]
        
        # ccxt와 통신하기 위해 naive datetime에 UTC 시간대 정보를 잠시 부여
        last_dt_aware = last_dt_naive.tz_localize('UTC')
        last_timestamp = int(last_dt_aware.timestamp() * 1000)
        print(f"마지막 데이터 시점 (UTC): {last_dt_naive} (timestamp: {last_timestamp})")

    # 3. 마지막 시점 이후 데이터만 추가
    print("최신 데이터 수집을 시작합니다...")
    # `collect_all_ohlcv`는 이제 datetime이 포함된 DataFrame을 반환
    new_data = collect_all_ohlcv(symbol=symbol, timeframe=timeframe, since_timestamp=last_timestamp)

    if new_data.empty:
        print("새롭게 추가할 데이터가 없습니다. 업데이트를 종료합니다.")
        return

    # 4. 데이터 합치기 및 저장
    print(f"총 {len(new_data)}개의 신규 캔들을 추가합니다.")
    updated_data = update_data(existing_data, new_data)

    print(f"업데이트 완료. 총 데이터 개수: {len(updated_data)}. CSV 파일에 저장합니다...")
    # 'timestamp' 컬럼이 naive datetime 객체이므로 to_csv에서 깔끔한 포맷으로 저장됨
    updated_data.to_csv(csv_path, index=False)
    
    # Parquet 파일도 업데이트
    parquet_dir = csv_path.parent.parent / 'parquet'
    parquet_file = parquet_dir / f"{csv_path.stem}.parquet"
    print(f"\nParquet 파일도 업데이트합니다: '{parquet_file}'")

    # Parquet 저장을 위해 naive UTC datetime을 aware UTC datetime으로 변환 후 인덱스로 설정
    updated_data_for_parquet = updated_data.copy()
    updated_data_for_parquet['timestamp'] = updated_data_for_parquet['timestamp'].dt.tz_localize('UTC')
    updated_data_for_parquet.set_index('timestamp', inplace=True)
    updated_data_for_parquet.to_parquet(parquet_file)
    
    print("CSV 및 Parquet 파일 업데이트가 성공적으로 완료되었습니다.")

def convert_csv_to_parquet(csv_path: Path, parquet_path: Path):
    """CSV 파일을 Parquet 파일로 변환합니다."""
    if not csv_path.exists():
        print(f"'{csv_path}' 파일이 존재하지 않아 변환할 수 없습니다.")
        return
    try:
        # CSV를 읽으면 naive datetime이 됨 (UTC 기준이라고 가정)
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        
        # Parquet 저장을 위해 naive UTC를 aware UTC로 변환 후 인덱스로 설정
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        df.set_index('timestamp', inplace=True)
        
        df.to_parquet(parquet_path)
        print(f"'{csv_path}'를 '{parquet_path}'로 성공적으로 변환했습니다.")
    except Exception as e:
        print(f"Parquet 변환 중 오류 발생: {e}")

def main():
    parser = argparse.ArgumentParser(description="BTC 1분봉 데이터 수집 및 관리 스크립트")
    parser.add_argument('--csv-dir', type=str, default='data/rwa/csv', help="CSV 파일을 저장할 디렉터리. 기본값: 'data/rwa/csv'")
    parser.add_argument('--parquet-dir', type=str, default='data/rwa/parquet', help="Parquet 파일을 저장할 디렉터리. 기본값: 'data/rwa/parquet'")
    
    # 기능 선택을 위한 상호 배타적 그룹
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--initial-collect', action='store_true', help='최근 1000개의 데이터로 초기 CSV 파일을 생성합니다.')
    group.add_argument('--recover', action='store_true', help='Parquet 파일에서 데이터를 복구하여 CSV 파일을 생성/덮어씁니다.')
    group.add_argument('--safe-update', action='store_true', help='기존 CSV를 백업하고 최신 데이터로 안전하게 이어붙입니다.')
    group.add_argument('--convert-only', action='store_true', help='기존 CSV 파일을 Parquet 형식으로 변환만 수행합니다.')
    group.add_argument('--convert-to-utc', action='store_true', help='KST로 잘못 저장된 기존 CSV를 UTC로 변환합니다 (9시간을 뺌).')

    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    parquet_dir = Path(args.parquet_dir)

    # 지정된 경로에 폴더가 없으면 생성
    csv_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    csv_file = csv_dir / 'btc_1min.csv'
    parquet_file = parquet_dir / 'btc_1min.parquet'
    
    symbol = 'BTC/USDT'
    timeframe = '1m'

    if args.recover:
        print("데이터 복구 작업을 시작합니다...")
        recover_from_parquet(parquet_file, csv_file)

    elif args.safe_update:
        print("안전한 데이터 업데이트를 시작합니다...")
        safe_update_csv(csv_file, symbol, timeframe)
    
    elif args.convert_only:
        print("CSV to Parquet 변환을 시작합니다...")
        convert_csv_to_parquet(csv_file, parquet_file)

    elif args.convert_to_utc:
        print("기존 CSV의 KST 시간을 UTC로 변환하는 작업을 시작합니다...")
        if not csv_file.exists():
            print(f"오류: 변환할 CSV 파일이 없습니다: '{csv_file}'", file=sys.stderr)
            sys.exit(1)
        
        # 1. 백업 생성
        backup_filename = f"{csv_file.stem}_backup_kst_{datetime.now().strftime('%Y%m%d_%H%M%S')}{csv_file.suffix}"
        backup_path = csv_file.with_name(backup_filename)
        try:
            shutil.copy(csv_file, backup_path)
            print(f"원본 KST 데이터 백업 완료: '{backup_path}'")
        except Exception as e:
            print(f"백업 파일 생성 중 오류 발생: {e}", file=sys.stderr)
            sys.exit(1)
            
        # 2. KST(naive) -> UTC(naive) 변환
        print("타임스탬프 변환 중 (KST -> UTC)...")
        df = pd.read_csv(csv_file, parse_dates=['timestamp'])
        df['timestamp'] = df['timestamp'] - pd.Timedelta(hours=9)
        
        # 3. 변환된 파일 저장
        df.to_csv(csv_file, index=False)
        print(f"성공: '{csv_file}'의 모든 타임스탬프를 UTC 기준으로 변환했습니다.")
        
        # 4. Parquet 파일도 업데이트
        print("\nParquet 파일도 함께 업데이트합니다...")
        convert_csv_to_parquet(csv_file, parquet_file)

    elif args.initial_collect:
        print("최초 데이터 수집을 시작합니다... (최근 1000개)")
        # 최초 실행 시에는 간단하게 최근 1000개만 가져옴
        binance = ccxt.binance()
        ohlcv = binance.fetch_ohlcv(symbol, timeframe, limit=1000)
        
        # fetch_ohlcv는 이미 timestamp, open, high, low, close, volume 순서
        initial_data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])  # type: ignore
        
        if not initial_data.empty:
            # CSV 저장을 위해 timestamp(ms)를 naive UTC datetime 객체로 변환
            initial_data['timestamp'] = pd.to_datetime(initial_data['timestamp'], unit='ms')
            
            initial_data.to_csv(csv_file, index=False)
            print(f"'{csv_file}'에 초기 데이터를 저장했습니다.")
            
            # Parquet 변환도 함께 수행
            print("\nParquet 파일로 변환합니다...")
            # Parquet 저장을 위해 naive UTC를 aware UTC로 변환 후 인덱스로 설정
            initial_data_for_parquet = initial_data.copy()
            initial_data_for_parquet['timestamp'] = initial_data_for_parquet['timestamp'].dt.tz_localize('UTC')
            initial_data_for_parquet.set_index('timestamp', inplace=True)
            initial_data_for_parquet.to_parquet(parquet_file)
            print(f"'{parquet_file}'에 Parquet 데이터를 저장했습니다.")
        else:
            print("데이터를 가져오지 못했습니다.")

if __name__ == '__main__':
    main() 