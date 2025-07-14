#!/usr/bin/env python3
"""
데이터 압축 및 최적화 스크립트
350만개 데이터 성능 최적화용
"""

import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
import gzip
import pickle
from datetime import datetime

def compress_csv_files():
    """CSV 파일들을 압축하여 저장"""
    
    print("🗜️ CSV 파일 압축 중...")
    
    # 압축할 CSV 파일들
    csv_files = [
        "results/profit_distribution_summary.csv",
        "results/profit_distribution/profit_distribution_1min.csv",
        "results/profit_distribution/profit_distribution_5min.csv",
        "results/profit_distribution/profit_distribution_10min.csv",
        "results/profit_distribution/profit_distribution_15min.csv",
        "results/profit_distribution/profit_distribution_30min.csv",
        "results/profit_distribution/profit_distribution_1h.csv",
        "results/profit_distribution/profit_distribution_2h.csv",
        "results/profit_distribution/profit_distribution_4h.csv",
        "results/profit_distribution/profit_distribution_6h.csv",
        "results/profit_distribution/profit_distribution_8h.csv",
        "results/profit_distribution/profit_distribution_12h.csv",
        "results/profit_distribution/profit_distribution_1day.csv",
        "results/profit_distribution/profit_distribution_3day.csv",
        "results/profit_distribution/profit_distribution_1week.csv",
        "results/profit_distribution/profit_distribution_1month.csv"
    ]
    
    compressed_count = 0
    total_saved = 0
    
    for csv_file in csv_files:
        file_path = Path(csv_file)
        if not file_path.exists():
            continue
            
        # 원본 파일 크기
        original_size = file_path.stat().st_size
        
        # 압축 파일 경로
        compressed_path = file_path.with_suffix('.csv.gz')
        
        try:
            # CSV 읽기
            df = pd.read_csv(file_path)
            
            # 압축 저장
            with gzip.open(compressed_path, 'wt', encoding='utf-8') as f:
                df.to_csv(f, index=False)
            
            # 압축 후 크기
            compressed_size = compressed_path.stat().st_size
            saved_space = original_size - compressed_size
            compression_ratio = (1 - compressed_size / original_size) * 100
            
            print(f"📁 {csv_file}")
            print(f"   원본: {original_size:,} bytes")
            print(f"   압축: {compressed_size:,} bytes")
            print(f"   절약: {saved_space:,} bytes ({compression_ratio:.1f}%)")
            print()
            
            compressed_count += 1
            total_saved += saved_space
            
        except Exception as e:
            print(f"❌ 압축 실패 ({csv_file}): {e}")
    
    print(f"✅ 압축 완료: {compressed_count}개 파일")
    print(f"💾 총 절약 공간: {total_saved:,} bytes ({total_saved/1024/1024:.1f} MB)")

def create_sample_data():
    """개발용 샘플 데이터 생성"""
    
    print("📊 개발용 샘플 데이터 생성 중...")
    
    # 샘플 데이터 디렉토리
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    # 작은 크기의 샘플 데이터 생성
    sample_sizes = {
        "profit_distribution_sample.csv": 1000,
        "label_analysis_sample.csv": 500,
        "trading_data_sample.csv": 2000
    }
    
    for filename, size in sample_sizes.items():
        file_path = sample_dir / filename
        
        if "profit" in filename:
            # 수익 분포 샘플 데이터
            data = {
                'timestamp': pd.date_range('2024-01-01', periods=size, freq='1min'),
                'profit': np.random.normal(0, 100, size),
                'holding_time': np.random.randint(1, 1440, size),
                'timeframe': np.random.choice(['1min', '5min', '15min', '1h'], size)
            }
        elif "label" in filename:
            # 라벨 분석 샘플 데이터
            data = {
                'timestamp': pd.date_range('2024-01-01', periods=size, freq='5min'),
                'label': np.random.choice([0, 1, 2], size),
                'confidence': np.random.uniform(0.5, 1.0, size),
                'timeframe': np.random.choice(['1min', '5min', '15min'], size)
            }
        else:
            # 거래 데이터 샘플
            data = {
                'timestamp': pd.date_range('2024-01-01', periods=size, freq='1min'),
                'open': np.random.uniform(100, 200, size),
                'high': np.random.uniform(100, 200, size),
                'low': np.random.uniform(100, 200, size),
                'close': np.random.uniform(100, 200, size),
                'volume': np.random.randint(1000, 10000, size)
            }
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        
        print(f"📁 생성: {filename} ({size:,} rows)")
    
    print("✅ 샘플 데이터 생성 완료")

def optimize_memory_usage():
    """메모리 사용량 최적화"""
    
    print("🧠 메모리 사용량 최적화 중...")
    
    # 데이터 타입 최적화 함수
    def optimize_dtypes(df):
        """DataFrame의 데이터 타입을 최적화"""
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # 문자열 컬럼 최적화
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
            elif df[col].dtype == 'int64':
                # 정수 컬럼 최적화
                c_min = df[col].min()
                c_max = df[col].max()
                
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            elif df[col].dtype == 'float64':
                # 실수 컬럼 최적화
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df
    
    # 최적화할 파일들
    files_to_optimize = [
        "results/profit_distribution_summary.csv"
    ]
    
    for file_path in files_to_optimize:
        path = Path(file_path)
        if not path.exists():
            continue
            
        try:
            print(f"🔧 최적화 중: {file_path}")
            
            # 원본 크기
            original_size = path.stat().st_size
            
            # 데이터 읽기 및 최적화
            df = pd.read_csv(path)
            df_optimized = optimize_dtypes(df)
            
            # 최적화된 파일 저장
            optimized_path = path.with_suffix('.optimized.csv')
            df_optimized.to_csv(optimized_path, index=False)
            
            # 최적화 후 크기
            optimized_size = optimized_path.stat().st_size
            saved_space = original_size - optimized_size
            
            print(f"   원본: {original_size:,} bytes")
            print(f"   최적화: {optimized_size:,} bytes")
            print(f"   절약: {saved_space:,} bytes")
            print()
            
        except Exception as e:
            print(f"❌ 최적화 실패 ({file_path}): {e}")

def create_data_index():
    """데이터 인덱스 생성"""
    
    print("📋 데이터 인덱스 생성 중...")
    
    # 인덱스 파일 생성
    index_data = {
        'file_path': [],
        'file_size': [],
        'row_count': [],
        'last_modified': [],
        'file_type': []
    }
    
    # 스캔할 디렉토리들
    scan_dirs = ['results', 'models', 'logs']
    
    for scan_dir in scan_dirs:
        dir_path = Path(scan_dir)
        if not dir_path.exists():
            continue
            
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                try:
                    # 파일 정보 수집
                    stat = file_path.stat()
                    
                    index_data['file_path'].append(str(file_path))
                    index_data['file_size'].append(stat.st_size)
                    index_data['last_modified'].append(datetime.fromtimestamp(stat.st_mtime))
                    index_data['file_type'].append(file_path.suffix)
                    
                    # 행 수 계산 (CSV 파일만)
                    if file_path.suffix == '.csv':
                        try:
                            df = pd.read_csv(file_path, nrows=1)
                            # 전체 행 수 추정 (첫 1000행으로)
                            sample_df = pd.read_csv(file_path, nrows=1000)
                            if len(sample_df) == 1000:
                                # 파일이 1000행보다 크면 추정
                                estimated_rows = int(stat.st_size / (sample_df.memory_usage(deep=True).sum() / len(sample_df)))
                            else:
                                estimated_rows = len(sample_df)
                            index_data['row_count'].append(estimated_rows)
                        except:
                            index_data['row_count'].append(0)
                    else:
                        index_data['row_count'].append(0)
                        
                except Exception as e:
                    print(f"⚠️ 파일 스캔 실패 ({file_path}): {e}")
    
    # 인덱스 DataFrame 생성
    index_df = pd.DataFrame(index_data)
    
    # 인덱스 저장
    index_path = Path("data_index.csv")
    index_df.to_csv(index_path, index=False)
    
    # 통계 출력
    total_files = len(index_df)
    total_size = index_df['file_size'].sum()
    total_rows = index_df['row_count'].sum()
    
    print(f"📊 인덱스 생성 완료:")
    print(f"   총 파일 수: {total_files:,}")
    print(f"   총 크기: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    print(f"   총 행 수: {total_rows:,}")
    print(f"   인덱스 파일: {index_path}")

def main():
    """메인 함수"""
    
    print("🚀 데이터 최적화 시작...")
    print("="*50)
    
    # 1. CSV 파일 압축
    compress_csv_files()
    print()
    
    # 2. 샘플 데이터 생성
    create_sample_data()
    print()
    
    # 3. 메모리 최적화
    optimize_memory_usage()
    print()
    
    # 4. 데이터 인덱스 생성
    create_data_index()
    print()
    
    print("="*50)
    print("✅ 데이터 최적화 완료!")
    print("💡 이제 Cursor 성능이 크게 개선될 것입니다.")

if __name__ == "__main__":
    main() 