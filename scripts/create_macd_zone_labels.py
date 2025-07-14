import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

class MACDZoneLabeler:
    """MACD 히스토그램 구역 기반 라벨링"""
    
    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = fast
        self.slow = slow  
        self.signal = signal
    
    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD 지표 계산"""
        df = df.copy()
        
        # MACD 계산
        exp1 = df['close'].ewm(span=self.fast).mean()
        exp2 = df['close'].ewm(span=self.slow).mean()
        
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=self.signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def detect_zones_and_extremes(self, df: pd.DataFrame) -> pd.DataFrame:
        """구역 탐지 및 극값 라벨링"""
        df = df.copy()
        df['label'] = 0  # 기본값: 관망
        
        histogram = df['macd_histogram'].to_numpy()
        labels = df['label'].to_numpy()
        
        # 구역 변화점 찾기
        sign_changes = np.where(np.diff(np.sign(histogram)))[0]
        zones = []
        
        start_idx = 0
        for change_idx in sign_changes:
            end_idx = change_idx
            zone_values = histogram[start_idx:end_idx+1]
            
            if len(zone_values) > 0:
                zones.append({
                    'start': start_idx,
                    'end': end_idx,
                    'type': 'negative' if zone_values[0] < 0 else 'positive',
                    'values': zone_values
                })
            
            start_idx = end_idx + 1
        
        # 마지막 구역 처리
        if start_idx < len(histogram):
            zone_values = histogram[start_idx:]
            if len(zone_values) > 0:
                zones.append({
                    'start': start_idx,
                    'end': len(histogram) - 1,
                    'type': 'negative' if zone_values[0] < 0 else 'positive',
                    'values': zone_values
                })
        
        # 각 구역에서 극값 찾기 및 라벨링
        for zone in zones:
            zone_start = zone['start']
            zone_end = zone['end']
            zone_histogram = histogram[zone_start:zone_end+1]
            
            if zone['type'] == 'negative':
                # 음수 구역: 가장 낮은 값(L값) 찾기
                min_idx = np.argmin(zone_histogram)
                actual_idx = zone_start + min_idx
                labels[actual_idx] = 1  # 매수 라벨
                
            elif zone['type'] == 'positive':
                # 양수 구역: 가장 높은 값(H값) 찾기  
                max_idx = np.argmax(zone_histogram)
                actual_idx = zone_start + max_idx
                labels[actual_idx] = -1  # 매도 라벨
        
        df['label'] = labels
        return df
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """전체 라벨링 프로세스"""
        # MACD 계산
        df_with_macd = self.calculate_macd(df)
        
        # 구역 탐지 및 라벨링
        df_labeled = self.detect_zones_and_extremes(df_with_macd)
        
        return df_labeled

def process_all_timeframes(target_timeframe: str | None = None):
    """15개 타임프레임 모두 처리"""
    
    # 🔥 경로 수정 - 실제 파이프라인과 일치
    input_dir = Path('data/processed/btc_usdt_kst/resampled_ohlcv')
    output_dir = Path('data/processed/btc_usdt_kst/labeled')  # 🔥 수정됨
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 입력 경로: {input_dir}")
    print(f"📁 출력 경로: {output_dir}")
    
    # 타임프레임 파일명 매핑 (실제 파일명과 일치)
    timeframe_files = {
        '1min': '1min.parquet',
        '3min': '3min.parquet',
        '5min': '5min.parquet',
        '10min': '10min.parquet',
        '15min': '15min.parquet',
        '30min': '30min.parquet',
        '1h': '1h.parquet',
        '2h': '2h.parquet',
        '4h': '4h.parquet',
        '6h': '6h.parquet',
        '8h': '8h.parquet',
        '12h': '12h.parquet',
        '1d': '1day.parquet',      # 🔥 파일명 수정
        '2d': '2day.parquet',      # 🔥 추가
        '3d': '3day.parquet',      # 🔥 파일명 수정
        '1w': '1week.parquet'      # 🔥 파일명 수정
    }
    
    # 🔥 실제 존재하는 파일 확인
    available_files = list(input_dir.glob('*.parquet'))
    print(f"📊 사용 가능한 파일: {len(available_files)}개")
    for file in sorted(available_files):
        print(f"  - {file.name}")
    
    # 타임프레임 목록 (실제 파일 기준)
    timeframes: list[str] = list(timeframe_files.keys())
    
    if target_timeframe:
        if target_timeframe in timeframes:
            timeframes = [target_timeframe]
        else:
            print(f"❌ 오류: '{target_timeframe}'은 유효한 타임프레임이 아닙니다.")
            print(f"✅ 유효한 값: {', '.join(timeframes)}")
            return

    labeler = MACDZoneLabeler()
    successful_count = 0
    failed_count = 0
    
    for tf in tqdm(timeframes, desc="🔄 타임프레임 처리중"):
        try:
            # 🔥 실제 파일명 매핑 사용
            filename = timeframe_files[tf]
            input_file = input_dir / filename
            
            if not input_file.exists():
                print(f"⚠️ '{tf}': 파일이 존재하지 않습니다 - {input_file}")
                failed_count += 1
                continue
                
            df = pd.read_parquet(input_file)
            print(f"✅ '{tf}': 데이터 로드 완료 ({len(df):,}개 캔들)")
            
            # 인덱스 타입 확인 및 자동 복구
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"⚠️ 경고: '{tf}' 인덱스가 DatetimeIndex가 아닙니다. (타입: {type(df.index)})")
                print(f"   🔧 인덱스 자동 복구 시도...")
                
                try:
                    # timestamp 컬럼 찾기
                    if 'timestamp' in df.columns:
                        ts_col = 'timestamp'
                    elif df.index.name and isinstance(df.index.name, str) and 'time' in df.index.name.lower():
                        # 이미 시간 관련 인덱스인 경우
                        df.index = pd.to_datetime(df.index)
                        ts_col = None # 이미 인덱스 처리됨
                    else:
                        # 첫 번째 컬럼 시도
                        ts_col = df.columns[0]
                        df[ts_col] = pd.to_datetime(df[ts_col])
                        df = df.set_index(ts_col)
                    
                    if isinstance(df.index, pd.DatetimeIndex):
                        print(f"   ✅ 인덱스 복구 성공!")
                    else:
                        raise ValueError("복구 후에도 DatetimeIndex가 아닙니다.")
                        
                except Exception as e:
                    print(f"   ❌ 인덱스 복구 실패: {e}")
                    print(f"   ⏭️ '{tf}' 건너뛰기")
                    failed_count += 1
                    continue

            # 라벨링 수행
            print(f"🔄 '{tf}': MACD 라벨링 수행중...")
            df_labeled = labeler.create_labels(df)
            
            # 라벨 분포 확인
            label_counts = df_labeled['label'].value_counts().sort_index()
            total = len(df_labeled)
            
            print(f"📊 '{tf}' 라벨 분포:")
            for label, count in zip(label_counts.index, label_counts.values):
                pct = count / total * 100
                label_int = int(label)
                label_name = {0: "관망", 1: "매수", -1: "매도"}.get(label_int, f"라벨{label_int}")
                print(f"   {label_name} ({label_int}): {count:,}개 ({pct:.1f}%)")
            
            # 🔥 출력 파일명 수정 (일관성 있게)
            output_file = output_dir / f"{tf}_macd_labeled.parquet"
            df_labeled.to_parquet(output_file, index=True)
            print(f"💾 '{tf}': 저장 완료 - {output_file}")
            print("-" * 60)
            
            successful_count += 1
            
        except Exception as e:
            print(f"❌ '{tf}': 처리 중 오류 발생 - {e}")
            failed_count += 1
    
    # 최종 결과 요약
    print("\n" + "="*60)
    print("📋 MACD 라벨링 결과 요약")
    print("="*60)
    print(f"✅ 성공: {successful_count}개 타임프레임")
    print(f"❌ 실패: {failed_count}개 타임프레임")
    print(f"📊 성공률: {successful_count/(successful_count+failed_count)*100:.1f}%")
    
    if successful_count > 0:
        print(f"\n📁 생성된 라벨 파일 위치: {output_dir}")
        print("🚀 다음 단계: Phase 2.5 라벨 분석을 다시 실행하세요!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MACD 구역 라벨링 스크립트")
    parser.add_argument(
        "--timeframe", 
        type=str,
        default=None,
        help="특정 타임프레임만 처리 (예: '1min', '5min'). 미지정시 모두 처리"
    )
    args = parser.parse_args()

    print("🚀 MACD 구역 기반 라벨링 시작")
    print("="*60)

    if args.timeframe:
        print(f"🎯 타겟: '{args.timeframe}' 타임프레임만 처리")
        process_all_timeframes(target_timeframe=args.timeframe)
    else:
        print("🎯 타겟: 모든 타임프레임 처리")
        process_all_timeframes()
    
    print("\n🏁 MACD 라벨링 완료!")