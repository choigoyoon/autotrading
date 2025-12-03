"""
차트 패턴 성공률 백테스트

각 패턴이 완성된 후 실제로 예상대로 움직였는가?
"""

import pandas as pd
import numpy as np

class PatternSuccessRateAnalyzer:
    def __init__(self, csv_file='analysis_15m.csv'):
        print("=" * 80)
        print("Chart Pattern Success Rate Analysis")
        print("=" * 80)
        
        self.df_raw = pd.read_csv(csv_file)
        self.df_raw['datetime'] = pd.to_datetime(self.df_raw['datetime'])
        self.df_raw = self.df_raw.sort_values('datetime').reset_index(drop=True)
        self.df_recent = self.df_raw[self.df_raw['datetime'] >= '2020-01-01'].copy()
        
        self.prepare_data()
        
    def prepare_data(self):
        """데이터 준비"""
        df = self.df_recent.set_index('datetime')
        
        # 1시간봉으로 리샘플링
        self.df_1h = df.resample('1H').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        
        # MACD 계산
        exp1 = self.df_1h['close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df_1h['close'].ewm(span=26, adjust=False).mean()
        self.df_1h['macd'] = exp1 - exp2
        self.df_1h['signal'] = self.df_1h['macd'].ewm(span=9, adjust=False).mean()
        self.df_1h['hist'] = self.df_1h['macd'] - self.df_1h['signal']
        
        print(f"\nData loaded: {len(self.df_1h):,} 1H candles")
    
    def extract_hl_points(self):
        """MACD 구간별 H/L 변곡점 추출"""
        hist = self.df_1h['hist'].values
        high = self.df_1h['high'].values
        low = self.df_1h['low'].values
        timestamps = self.df_1h['datetime'].values
        
        n = len(hist)
        points = []
        
        i = 0
        while i < n:
            if hist[i] > 0:
                start = i
                while i < n and hist[i] > 0:
                    i += 1
                
                segment_highs = high[start:i]
                max_idx = start + np.argmax(segment_highs)
                points.append({
                    'type': 'H',
                    'price': high[max_idx],
                    'time': timestamps[max_idx],
                    'idx': max_idx
                })
            
            elif hist[i] < 0:
                start = i
                while i < n and hist[i] < 0:
                    i += 1
                
                segment_lows = low[start:i]
                min_idx = start + np.argmin(segment_lows)
                points.append({
                    'type': 'L',
                    'price': low[min_idx],
                    'time': timestamps[min_idx],
                    'idx': min_idx
                })
            else:
                i += 1
        
        return points
    
    def detect_double_bottom(self, points, tolerance=0.02):
        """Double Bottom 패턴 감지"""
        patterns = []
        
        for i in range(len(points) - 2):
            if points[i]['type'] == 'L' and points[i+2]['type'] == 'L':
                L1 = points[i]['price']
                L2 = points[i+2]['price']
                
                diff_pct = abs(L2 - L1) / L1
                
                if diff_pct < tolerance:
                    patterns.append({
                        'L1': points[i],
                        'H': points[i+1],
                        'L2': points[i+2],
                        'neckline': points[i+1]['price']
                    })
        
        return patterns
    
    def test_double_bottom_success(self, pattern, lookforward_hours=48):
        """
        Double Bottom 성공 여부 테스트
        
        성공 = L2 이후 Neckline 돌파하고 H 이상 도달
        """
        L2_idx = pattern['L2']['idx']
        neckline = pattern['neckline']
        target_H = pattern['H']['price']
        L2_price = pattern['L2']['price']
        
        # L2 이후 lookforward_hours 시간 동안 추적
        future_data = self.df_1h.iloc[L2_idx:L2_idx+lookforward_hours]
        
        if len(future_data) < 2:
            return None  # 데이터 부족
        
        # Neckline 돌파했는가?
        breakout = future_data[future_data['high'] > neckline]
        if len(breakout) == 0:
            return {
                'success': False,
                'reason': 'No breakout',
                'max_reached': future_data['high'].max(),
                'target': target_H,
                'L2_price': L2_price
            }
        
        # H 이상 도달했는가?
        max_price = future_data['high'].max()
        min_price = future_data['low'].min()
        
        # 손절 체크: L2 아래로 떨어졌는가?
        if min_price < L2_price * 0.98:  # L2 대비 2% 이상 하락
            return {
                'success': False,
                'reason': 'Stop loss hit',
                'max_reached': max_price,
                'min_reached': min_price,
                'target': target_H,
                'L2_price': L2_price
            }
        
        if max_price >= target_H:
            return {
                'success': True,
                'max_reached': max_price,
                'target': target_H,
                'L2_price': L2_price,
                'gain_pct': (max_price - L2_price) / L2_price * 100
            }
        else:
            return {
                'success': False,
                'reason': 'Failed to reach target',
                'max_reached': max_price,
                'target': target_H,
                'L2_price': L2_price
            }
    
    def detect_ascending_triangle(self, points, tolerance=0.02):
        """상승 삼각형 패턴 감지 (H는 비슷, L은 상승)"""
        patterns = []
        
        for i in range(len(points) - 4):
            # H1, L1, H2, L2 패턴
            if (points[i]['type'] == 'H' and 
                points[i+1]['type'] == 'L' and
                points[i+2]['type'] == 'H' and
                points[i+3]['type'] == 'L'):
                
                H1 = points[i]['price']
                L1 = points[i+1]['price']
                H2 = points[i+2]['price']
                L2 = points[i+3]['price']
                
                # H는 비슷 (tolerance 이내)
                h_diff = abs(H2 - H1) / H1
                
                # L은 상승 (L2 > L1)
                l_rising = L2 > L1 * 1.005  # 0.5% 이상 상승
                
                if h_diff < tolerance and l_rising:
                    patterns.append({
                        'H1': points[i],
                        'L1': points[i+1],
                        'H2': points[i+2],
                        'L2': points[i+3],
                        'resistance': (H1 + H2) / 2
                    })
        
        return patterns
    
    def test_ascending_triangle_success(self, pattern, lookforward_hours=72):
        """상승 삼각형 성공 여부"""
        L2_idx = pattern['L2']['idx']
        resistance = pattern['resistance']
        L2_price = pattern['L2']['price']
        
        future_data = self.df_1h.iloc[L2_idx:L2_idx+lookforward_hours]
        
        if len(future_data) < 2:
            return None
        
        max_price = future_data['high'].max()
        min_price = future_data['low'].min()
        
        # 저항선 돌파?
        if max_price > resistance * 1.01:  # 1% 이상 돌파
            gain_pct = (max_price - L2_price) / L2_price * 100
            return {
                'success': True,
                'reason': 'Breakout',
                'max_reached': max_price,
                'resistance': resistance,
                'gain_pct': gain_pct
            }
        
        # 지지선 이탈?
        support = pattern['L1']['price']
        if min_price < support * 0.99:
            return {
                'success': False,
                'reason': 'Support broken',
                'min_reached': min_price,
                'support': support
            }
        
        return {
            'success': False,
            'reason': 'No breakout',
            'max_reached': max_price,
            'resistance': resistance
        }
    
    def analyze_hl_sequence(self, points):
        """HL 시퀀스 패턴 분석 - 연속 상승/하락"""
        results = []
        
        # L값들만 추출
        lows = [p for p in points if p['type'] == 'L']
        
        for i in range(2, len(lows)):
            L_prev2 = lows[i-2]['price']
            L_prev1 = lows[i-1]['price']
            L_curr = lows[i]['price']
            
            # 변화율
            change1 = (L_prev1 - L_prev2) / L_prev2 * 100
            change2 = (L_curr - L_prev1) / L_prev1 * 100
            
            # 패턴 분류
            if change1 < -0.5 and change2 > 0.5:
                pattern_type = 'V_REVERSAL'  # 하락 후 상승 (V자 반전)
            elif change1 > 0.5 and change2 > 0.5:
                pattern_type = 'RISING'  # 연속 상승
            elif change1 < -0.5 and change2 < -0.5:
                pattern_type = 'FALLING'  # 연속 하락
            elif change1 > 0.5 and change2 < -0.5:
                pattern_type = 'PEAK'  # 상승 후 하락
            else:
                pattern_type = 'SIDEWAYS'  # 횡보
            
            # 이후 성과 측정
            curr_idx = lows[i]['idx']
            future_data = self.df_1h.iloc[curr_idx:curr_idx+24]  # 24시간 후
            
            if len(future_data) > 1:
                entry_price = self.df_1h.iloc[curr_idx]['close']
                max_price = future_data['high'].max()
                min_price = future_data['low'].min()
                
                max_gain = (max_price - entry_price) / entry_price * 100
                max_loss = (min_price - entry_price) / entry_price * 100
                
                results.append({
                    'time': lows[i]['time'],
                    'pattern': pattern_type,
                    'change1': change1,
                    'change2': change2,
                    'max_gain': max_gain,
                    'max_loss': max_loss,
                    'net': max_gain + max_loss
                })
        
        return results
    
    def analyze_all_patterns(self):
        """전체 기간 모든 패턴 성공률 분석"""
        
        # 전체 HL 점 추출
        all_points = self.extract_hl_points()
        
        print(f"\nTotal HL points: {len(all_points)}")
        
        # 1. Double Bottom 테스트
        print(f"\n{'='*80}")
        print(f"1. Double Bottom Success Rate")
        print(f"{'='*80}")
        
        db_patterns = self.detect_double_bottom(all_points)
        print(f"Total Double Bottom patterns: {len(db_patterns)}")
        
        db_results = []
        for pattern in db_patterns:
            result = self.test_double_bottom_success(pattern)
            if result:
                db_results.append(result)
        
        if db_results:
            success_count = sum(1 for r in db_results if r['success'])
            success_rate = success_count / len(db_results) * 100
            
            print(f"Tested: {len(db_results)} patterns")
            print(f"Success: {success_count} ({success_rate:.1f}%)")
            print(f"Failed: {len(db_results) - success_count}")
            
            # 실패 이유 분석
            failed = [r for r in db_results if not r['success']]
            if failed:
                reasons = {}
                for r in failed:
                    reason = r.get('reason', 'Unknown')
                    reasons[reason] = reasons.get(reason, 0) + 1
                print(f"\nFailure reasons:")
                for reason, count in reasons.items():
                    print(f"  - {reason}: {count}")
            
            # 성공한 패턴들의 평균 수익
            successful = [r for r in db_results if r['success']]
            if successful:
                avg_gain = np.mean([r['gain_pct'] for r in successful])
                print(f"\nAverage gain (successful): {avg_gain:.2f}%")
        
        # 2. 상승 삼각형 테스트
        print(f"\n{'='*80}")
        print(f"2. Ascending Triangle Success Rate")
        print(f"{'='*80}")
        
        at_patterns = self.detect_ascending_triangle(all_points)
        print(f"Total Ascending Triangle patterns: {len(at_patterns)}")
        
        at_results = []
        for pattern in at_patterns:
            result = self.test_ascending_triangle_success(pattern)
            if result:
                at_results.append(result)
        
        if at_results:
            success_count = sum(1 for r in at_results if r['success'])
            success_rate = success_count / len(at_results) * 100
            
            print(f"Tested: {len(at_results)} patterns")
            print(f"Success: {success_count} ({success_rate:.1f}%)")
            
            successful = [r for r in at_results if r['success']]
            if successful:
                avg_gain = np.mean([r['gain_pct'] for r in successful])
                print(f"Average gain (successful): {avg_gain:.2f}%")
        
        # 3. HL 시퀀스 패턴 분석
        print(f"\n{'='*80}")
        print(f"3. HL Sequence Pattern Analysis")
        print(f"{'='*80}")
        
        seq_results = self.analyze_hl_sequence(all_points)
        
        if seq_results:
            df_seq = pd.DataFrame(seq_results)
            
            print(f"\nTotal sequences: {len(seq_results)}")
            print(f"\nPerformance by pattern type:")
            print("-" * 60)
            
            for pattern_type in df_seq['pattern'].unique():
                subset = df_seq[df_seq['pattern'] == pattern_type]
                avg_gain = subset['max_gain'].mean()
                avg_loss = subset['max_loss'].mean()
                win_rate = (subset['max_gain'] > abs(subset['max_loss'])).mean() * 100
                
                print(f"{pattern_type:12}: {len(subset):>4} cases, "
                      f"Avg Gain: {avg_gain:>5.2f}%, "
                      f"Avg Loss: {avg_loss:>6.2f}%, "
                      f"Win Rate: {win_rate:>5.1f}%")
        
        return {
            'double_bottom': {
                'total': len(db_patterns),
                'tested': len(db_results) if db_results else 0,
                'results': db_results
            },
            'ascending_triangle': {
                'total': len(at_patterns),
                'tested': len(at_results) if at_results else 0,
                'results': at_results
            },
            'hl_sequence': seq_results
        }

if __name__ == "__main__":
    analyzer = PatternSuccessRateAnalyzer()
    results = analyzer.analyze_all_patterns()
