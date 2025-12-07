"""
나우캐스트 준수 백테스트 - 순수 OHLCV 데이터 기반
미래 데이터 참조 완전 제거
"""

import pandas as pd
import numpy as np
from datetime import datetime


class NowcastBacktest:
    """나우캐스트 준수 백테스트 시스템"""
    
    def __init__(self, macd_fast=12, macd_slow=26, macd_signal=9):
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        
    def load_data(self, filepath):
        """OHLCV 데이터 로드"""
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 숫자형 변환
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna().reset_index(drop=True)
        return df
    
    def calculate_macd(self, df):
        """MACD 계산"""
        df = df.copy()
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        return df
    
    def generate_hl_labels_nowcast(self, df):
        """
        나우캐스트 준수 H/L 라벨링
        - MACD 히스토그램 부호 전환 시점에 라벨 확정
        - 미래 데이터 참조 없음
        """
        df = df.copy()
        df['label'] = None
        df['label_price'] = np.nan
        
        hist = df['macd_hist'].values
        current_sign = None
        segment_start = 0
        
        for i in range(len(df)):
            if pd.isna(hist[i]):
                continue
            
            sign = 1 if hist[i] >= 0 else -1
            
            if current_sign is not None and sign != current_sign:
                # 이전 구간 (과거 데이터만)
                segment = df.iloc[segment_start:i]
                
                if len(segment) > 0:
                    if current_sign == 1:  # 양수→음수: H
                        max_idx = segment['high'].idxmax()
                        df.loc[i, 'label'] = 'H'
                        df.loc[i, 'label_price'] = segment.loc[max_idx, 'high']
                    else:  # 음수→양수: L
                        min_idx = segment['low'].idxmin()
                        df.loc[i, 'label'] = 'L'
                        df.loc[i, 'label_price'] = segment.loc[min_idx, 'low']
                
                segment_start = i
            
            current_sign = sign
        
        return df
    
    def get_trendlines_at(self, df, current_idx, max_age=200):
        """
        현재 시점에서 유효한 추세선 계산
        - current_idx 이전의 확정된 H/L만 사용
        - 미래 참조 완전 제거
        """
        # 현재 시점까지의 데이터만
        past_df = df.iloc[:current_idx]
        
        h_labels = past_df[past_df['label'] == 'H']
        l_labels = past_df[past_df['label'] == 'L']
        
        trendlines = []
        
        # 하락추세선 (H 연결) - 최근 2개 H로만
        recent_h = h_labels[h_labels.index >= current_idx - max_age]
        if len(recent_h) >= 2:
            h1_idx = recent_h.index[-2]
            h2_idx = recent_h.index[-1]
            h1_price = recent_h.loc[h1_idx, 'label_price']
            h2_price = recent_h.loc[h2_idx, 'label_price']
            
            # 하락 조건: H2 < H1
            if h2_price < h1_price and h2_idx > h1_idx:
                slope = (h2_price - h1_price) / (h2_idx - h1_idx)
                trendlines.append({
                    'type': 'down',
                    'start_idx': h1_idx,
                    'start_price': h1_price,
                    'slope': slope
                })
        
        # 상승추세선 (L 연결) - 최근 2개 L로만
        recent_l = l_labels[l_labels.index >= current_idx - max_age]
        if len(recent_l) >= 2:
            l1_idx = recent_l.index[-2]
            l2_idx = recent_l.index[-1]
            l1_price = recent_l.loc[l1_idx, 'label_price']
            l2_price = recent_l.loc[l2_idx, 'label_price']
            
            # 상승 조건: L2 > L1
            if l2_price > l1_price and l2_idx > l1_idx:
                slope = (l2_price - l1_price) / (l2_idx - l1_idx)
                trendlines.append({
                    'type': 'up',
                    'start_idx': l1_idx,
                    'start_price': l1_price,
                    'slope': slope
                })
        
        return trendlines
    
    def get_tl_price(self, tl, idx):
        """특정 인덱스에서 추세선 가격"""
        return tl['start_price'] + tl['slope'] * (idx - tl['start_idx'])
    
    def detect_breakout(self, df, idx, trendlines):
        """
        현재 봉에서 돌파 감지
        - 이전 봉 close vs 추세선
        - 현재 봉 close vs 추세선
        """
        if idx < 1 or not trendlines:
            return None
        
        curr_close = df.iloc[idx]['close']
        prev_close = df.iloc[idx - 1]['close']
        
        for tl in trendlines:
            tl_curr = self.get_tl_price(tl, idx)
            tl_prev = self.get_tl_price(tl, idx - 1)
            
            # 하락추세선 상향 돌파 (롱)
            if tl['type'] == 'down':
                if prev_close <= tl_prev and curr_close > tl_curr:
                    return {'type': 'long', 'price': curr_close, 'tl_price': tl_curr}
            
            # 상승추세선 하향 돌파 (숏)
            elif tl['type'] == 'up':
                if prev_close >= tl_prev and curr_close < tl_curr:
                    return {'type': 'short', 'price': curr_close, 'tl_price': tl_curr}
        
        return None
    
    def detect_fvg(self, df, idx):
        """FVG (Fair Value Gap) 감지"""
        if idx < 2:
            return False
        return df.iloc[idx]['low'] > df.iloc[idx - 2]['high']
    
    def run_backtest(self, df, tp=2.0, sl=2.0, interval=10, long_only=True,
                     use_fvg_filter=False, use_dynamic_tp=False, verbose=True):
        """
        백테스트 실행
        
        Args:
            df: OHLCV DataFrame
            tp: Take Profit %
            sl: Stop Loss %
            interval: 최소 거래 간격 (봉)
            long_only: 롱만 거래
            use_fvg_filter: FVG 필터 사용
            use_dynamic_tp: 동적 TP (FVG 있으면 2.5%, 없으면 1.5%)
        """
        if verbose:
            print("MACD 계산 중...")
        df = self.calculate_macd(df)
        
        if verbose:
            print("H/L 라벨링 중...")
        df = self.generate_hl_labels_nowcast(df)
        
        # 라벨 통계
        h_count = (df['label'] == 'H').sum()
        l_count = (df['label'] == 'L').sum()
        if verbose:
            print(f"  H: {h_count}개, L: {l_count}개")
        
        trades = []
        last_entry = -interval - 1
        
        # MACD 웜업 기간
        start_idx = max(self.macd_slow + self.macd_signal + 50, 100)
        
        if verbose:
            print(f"백테스트 실행 중... ({start_idx} ~ {len(df)-50})")
        
        progress_step = (len(df) - start_idx) // 10
        
        for i in range(start_idx, len(df) - 50):
            # 진행 표시
            if verbose and progress_step > 0 and (i - start_idx) % progress_step == 0:
                pct = (i - start_idx) / (len(df) - start_idx - 50) * 100
                print(f"  진행: {pct:.0f}%")
            
            # 간격 필터
            if i - last_entry < interval:
                continue
            
            # 현재 시점의 추세선
            tls = self.get_trendlines_at(df, i)
            if not tls:
                continue
            
            # 돌파 감지
            brk = self.detect_breakout(df, i, tls)
            if not brk:
                continue
            
            # 롱 전용
            if long_only and brk['type'] != 'long':
                continue
            
            # FVG 필터
            has_fvg = self.detect_fvg(df, i)
            if use_fvg_filter and not has_fvg:
                continue
            
            # TP 설정
            if use_dynamic_tp:
                actual_tp = 2.5 if has_fvg else 1.5
            else:
                actual_tp = tp
            
            # 진입 (다음 봉 시가)
            entry_idx = i + 1
            entry_price = df.iloc[entry_idx]['open']
            direction = brk['type']
            
            # TP/SL 레벨
            if direction == 'long':
                tp_level = entry_price * (1 + actual_tp / 100)
                sl_level = entry_price * (1 - sl / 100)
            else:
                tp_level = entry_price * (1 - actual_tp / 100)
                sl_level = entry_price * (1 + sl / 100)
            
            # 결과 시뮬레이션 (최대 50봉)
            result = 'TIMEOUT'
            exit_price = df.iloc[min(entry_idx + 49, len(df) - 1)]['close']
            exit_idx = min(entry_idx + 49, len(df) - 1)
            
            for j in range(entry_idx + 1, min(entry_idx + 50, len(df))):
                high = df.iloc[j]['high']
                low = df.iloc[j]['low']
                
                if direction == 'long':
                    # 같은 봉에서 TP/SL 둘 다 터치 시 - 시가 기준 판단
                    if high >= tp_level and low <= sl_level:
                        open_price = df.iloc[j]['open']
                        if open_price >= tp_level:
                            result, exit_price, exit_idx = 'TP', tp_level, j
                        elif open_price <= sl_level:
                            result, exit_price, exit_idx = 'SL', sl_level, j
                        else:
                            # 시가가 중간이면 SL 먼저 가정 (보수적)
                            result, exit_price, exit_idx = 'SL', sl_level, j
                        break
                    elif high >= tp_level:
                        result, exit_price, exit_idx = 'TP', tp_level, j
                        break
                    elif low <= sl_level:
                        result, exit_price, exit_idx = 'SL', sl_level, j
                        break
                else:  # short
                    if low <= tp_level and high >= sl_level:
                        open_price = df.iloc[j]['open']
                        if open_price <= tp_level:
                            result, exit_price, exit_idx = 'TP', tp_level, j
                        elif open_price >= sl_level:
                            result, exit_price, exit_idx = 'SL', sl_level, j
                        else:
                            result, exit_price, exit_idx = 'SL', sl_level, j
                        break
                    elif low <= tp_level:
                        result, exit_price, exit_idx = 'TP', tp_level, j
                        break
                    elif high >= sl_level:
                        result, exit_price, exit_idx = 'SL', sl_level, j
                        break
            
            # 수익 계산
            if direction == 'long':
                pnl = (exit_price - entry_price) / entry_price * 100
            else:
                pnl = (entry_price - exit_price) / entry_price * 100
            
            trades.append({
                'entry_idx': entry_idx,
                'entry_time': df.iloc[entry_idx]['datetime'],
                'entry_price': entry_price,
                'direction': direction,
                'has_fvg': has_fvg,
                'tp_pct': actual_tp,
                'sl_pct': sl,
                'result': result,
                'exit_price': exit_price,
                'exit_idx': exit_idx,
                'pnl': pnl,
                'hold_bars': exit_idx - entry_idx
            })
            
            last_entry = i
        
        trades_df = pd.DataFrame(trades)
        
        # 통계
        if len(trades_df) > 0:
            stats = {
                'total_trades': len(trades_df),
                'win_rate': (trades_df['pnl'] > 0).mean() * 100,
                'avg_pnl': trades_df['pnl'].mean(),
                'total_pnl': trades_df['pnl'].sum(),
                'max_win': trades_df['pnl'].max(),
                'max_loss': trades_df['pnl'].min(),
                'avg_hold': trades_df['hold_bars'].mean(),
                'tp_count': (trades_df['result'] == 'TP').sum(),
                'sl_count': (trades_df['result'] == 'SL').sum(),
                'timeout_count': (trades_df['result'] == 'TIMEOUT').sum(),
            }
        else:
            stats = {'total_trades': 0}
        
        return trades_df, stats


def main():
    print("=" * 70)
    print("나우캐스트 준수 백테스트 (미래 참조 제거)")
    print("=" * 70)
    
    # 데이터 로드
    bt = NowcastBacktest()
    df = bt.load_data('btc_15m_ohlcv.csv')
    
    print(f"\n데이터: {len(df)}봉")
    print(f"기간: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    
    # 기본 테스트
    print("\n" + "=" * 70)
    print("[1] 기본 설정 (TP:2%, SL:2%, 간격:10봉)")
    print("=" * 70)
    
    trades1, stats1 = bt.run_backtest(df.copy(), tp=2.0, sl=2.0, interval=10)
    
    print(f"\n결과:")
    print(f"  총 거래: {stats1['total_trades']}회")
    print(f"  승률: {stats1['win_rate']:.1f}%")
    print(f"  평균 수익: {stats1['avg_pnl']:.3f}%")
    print(f"  총 수익: {stats1['total_pnl']:.1f}%")
    print(f"  TP: {stats1['tp_count']}, SL: {stats1['sl_count']}, TIMEOUT: {stats1['timeout_count']}")
    
    # 파라미터 최적화
    print("\n" + "=" * 70)
    print("[2] 파라미터 최적화")
    print("=" * 70)
    
    results = []
    test_params = [
        # 기본
        (1.0, 1.0, 'TP1_SL1'),
        (1.5, 1.5, 'TP1.5_SL1.5'),
        (2.0, 2.0, 'TP2_SL2'),
        (2.5, 2.5, 'TP2.5_SL2.5'),
        (3.0, 3.0, 'TP3_SL3'),
        # 넓은 SL (기존 발견)
        (1.0, 2.0, 'TP1_SL2'),
        (1.5, 2.0, 'TP1.5_SL2'),
        (1.0, 3.0, 'TP1_SL3'),
        (1.5, 3.0, 'TP1.5_SL3'),
        (2.0, 3.0, 'TP2_SL3'),
        # 빠른 TP
        (0.5, 1.0, 'TP0.5_SL1'),
        (0.5, 2.0, 'TP0.5_SL2'),
        (0.7, 1.5, 'TP0.7_SL1.5'),
        (0.7, 2.0, 'TP0.7_SL2'),
    ]
    
    for tp, sl, name in test_params:
        trades, stats = bt.run_backtest(df.copy(), tp=tp, sl=sl, interval=10, verbose=False)
        if stats['total_trades'] > 0:
            net_pnl = stats['avg_pnl'] - 0.11  # 수수료
            results.append({
                'name': name,
                'tp': tp,
                'sl': sl,
                'trades': stats['total_trades'],
                'win_rate': stats['win_rate'],
                'avg_pnl': stats['avg_pnl'],
                'net_pnl': net_pnl,
                'total_pnl': stats['total_pnl']
            })
            print(f"  {name}: 거래={stats['total_trades']}, 승률={stats['win_rate']:.1f}%, 평균={stats['avg_pnl']:.3f}%")
    
    # 결과 정렬
    results_df = pd.DataFrame(results).sort_values('net_pnl', ascending=False)
    
    print("\n상위 5개 (순수익 기준):")
    print(results_df.head(5).to_string(index=False))
    
    print("\n상위 5개 (승률 기준):")
    print(results_df.sort_values('win_rate', ascending=False).head(5).to_string(index=False))
    
    # FVG 필터 테스트
    print("\n" + "=" * 70)
    print("[3] FVG 필터 적용")
    print("=" * 70)
    
    trades_fvg, stats_fvg = bt.run_backtest(
        df.copy(), tp=2.0, sl=2.0, interval=10, 
        use_fvg_filter=True, verbose=False
    )
    
    print(f"FVG 필터:")
    print(f"  거래: {stats_fvg['total_trades']}회")
    print(f"  승률: {stats_fvg['win_rate']:.1f}%")
    print(f"  평균: {stats_fvg['avg_pnl']:.3f}%")
    
    # 동적 TP 테스트
    print("\n" + "=" * 70)
    print("[4] 동적 TP (FVG: 2.5%, 없음: 1.5%)")
    print("=" * 70)
    
    trades_dyn, stats_dyn = bt.run_backtest(
        df.copy(), tp=2.0, sl=2.0, interval=10,
        use_dynamic_tp=True, verbose=False
    )
    
    print(f"동적 TP:")
    print(f"  거래: {stats_dyn['total_trades']}회")
    print(f"  승률: {stats_dyn['win_rate']:.1f}%")
    print(f"  평균: {stats_dyn['avg_pnl']:.3f}%")
    
    # 최적 설정 상세
    if len(results_df) > 0:
        best = results_df.iloc[0]
        print("\n" + "=" * 70)
        print(f"[5] 최적 설정 상세: {best['name']}")
        print("=" * 70)
        
        trades_best, stats_best = bt.run_backtest(
            df.copy(), tp=best['tp'], sl=best['sl'], interval=10, verbose=False
        )
        
        # 월간 추정
        total_days = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days
        months = total_days / 30
        
        print(f"\n최적 파라미터: TP={best['tp']}%, SL={best['sl']}%")
        print(f"  총 거래: {stats_best['total_trades']}회")
        print(f"  승률: {stats_best['win_rate']:.1f}%")
        print(f"  평균 수익: {stats_best['avg_pnl']:.3f}%")
        print(f"  수수료 후: {stats_best['avg_pnl'] - 0.11:.3f}%")
        print(f"  총 수익: {stats_best['total_pnl']:.1f}%")
        print(f"\n월간 추정:")
        print(f"  월 거래: {stats_best['total_trades'] / months:.1f}회")
        print(f"  월 수익: {stats_best['total_pnl'] / months:.1f}%")
        
        # 거래 저장
        trades_best.to_csv('nowcast_trades_result.csv', index=False)
        print(f"\n거래 내역 저장: nowcast_trades_result.csv")
    
    # 비교 요약
    print("\n" + "=" * 70)
    print("기존 전략 vs 나우캐스트 비교")
    print("=" * 70)
    print("""
기존 전략 (미래 참조):
  - 승률: 84-90%
  - 월 수익: 14%+
  - 문제: 미래 데이터로 추세선 완성 (실전 불가)

나우캐스트 전략 (수정):
  - 미래 참조 완전 제거
  - 현재 시점 데이터만 사용
  - 실전 적용 가능
""")
    
    if len(results_df) > 0:
        best_wr = results_df.sort_values('win_rate', ascending=False).iloc[0]
        print(f"나우캐스트 최고 승률: {best_wr['win_rate']:.1f}% ({best_wr['name']})")
        print(f"나우캐스트 최고 수익: {results_df.iloc[0]['net_pnl']:.3f}% ({results_df.iloc[0]['name']})")
    
    # 결과 저장
    results_df.to_csv('optimization_results.csv', index=False)
    print(f"\n최적화 결과 저장: optimization_results.csv")
    
    print("\n" + "=" * 70)
    print("백테스트 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
