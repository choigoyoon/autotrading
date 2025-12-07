#!/usr/bin/env python3
"""
L값이 내려가는 상황 분석
"L값이 내려간다" = Higher Low (HL) 형성 = 저점이 높아지는 = 상승 전환 신호
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("L값 하락(HL 형성) 상황 분석")
print("=" * 80)

# 1. L패턴 데이터 로드
try:
    l_patterns = pd.read_csv('L_patterns_analysis.csv')
    print(f"\n✅ L패턴 데이터 로드: {len(l_patterns):,}개 레코드")
    print(f"컬럼: {l_patterns.columns.tolist()}")
except Exception as e:
    print(f"❌ L패턴 데이터 로드 실패: {e}")
    l_patterns = None

# 2. 모든 L값 데이터 로드
try:
    all_l_values = pd.read_csv('all_L_values.csv')
    print(f"\n✅ 모든 L값 데이터 로드: {len(all_l_values):,}개 레코드")
    print(f"컬럼: {all_l_values.columns.tolist()}")
except Exception as e:
    print(f"❌ 모든 L값 데이터 로드 실패: {e}")
    all_l_values = None

# 3. L값 시퀀스 단계 분석 데이터 로드
try:
    l_sequence = pd.read_csv('L_sequence_stages_analysis.csv')
    print(f"\n✅ L시퀀스 단계 데이터 로드: {len(l_sequence):,}개 레코드")
    print(f"컬럼: {l_sequence.columns.tolist()}")
except Exception as e:
    print(f"❌ L시퀀스 데이터 로드 실패: {e}")
    l_sequence = None

print("\n" + "=" * 80)
print("1단계: L값 하락(HL 형성) 빈도 분석")
print("=" * 80)

if all_l_values is not None:
    # L값 변화율 계산
    if 'L_value' in all_l_values.columns:
        all_l_values['L_prev'] = all_l_values['L_value'].shift(1)
        all_l_values['L_change_pct'] = ((all_l_values['L_value'] - all_l_values['L_prev']) / all_l_values['L_prev'] * 100).round(3)
        
        # L값 변화 방향 분류
        all_l_values['L_direction'] = 'Same'
        all_l_values.loc[all_l_values['L_change_pct'] > 0, 'L_direction'] = 'Higher (HL)'  # L값이 올라감 = HL
        all_l_values.loc[all_l_values['L_change_pct'] < 0, 'L_direction'] = 'Lower (LL)'   # L값이 내려감 = LL
        
        # 방향별 통계
        direction_stats = all_l_values['L_direction'].value_counts()
        direction_pct = (direction_stats / len(all_l_values) * 100).round(2)
        
        print("\n📊 L값 변화 방향 분포:")
        for direction, count in direction_stats.items():
            pct = direction_pct[direction]
            print(f"  • {direction}: {count:,}개 ({pct}%)")
        
        # L값 하락(HL) 케이스만 필터링
        hl_cases = all_l_values[all_l_values['L_direction'] == 'Higher (HL)'].copy()
        ll_cases = all_l_values[all_l_values['L_direction'] == 'Lower (LL)'].copy()
        
        print(f"\n📈 Higher Low (HL) 케이스: {len(hl_cases):,}개")
        print(f"   - L값 평균 상승률: {hl_cases['L_change_pct'].mean():.3f}%")
        print(f"   - L값 중간 상승률: {hl_cases['L_change_pct'].median():.3f}%")
        print(f"   - L값 최대 상승률: {hl_cases['L_change_pct'].max():.3f}%")
        
        print(f"\n📉 Lower Low (LL) 케이스: {len(ll_cases):,}개")
        print(f"   - L값 평균 하락률: {ll_cases['L_change_pct'].mean():.3f}%")
        print(f"   - L값 중간 하락률: {ll_cases['L_change_pct'].median():.3f}%")
        print(f"   - L값 최대 하락률: {ll_cases['L_change_pct'].min():.3f}%")

print("\n" + "=" * 80)
print("2단계: HL 발생 시점의 시장 상황 분석")
print("=" * 80)

if all_l_values is not None and 'L_direction' in all_l_values.columns:
    # HL 케이스에서 중요 컬럼 분석
    hl_cases = all_l_values[all_l_values['L_direction'] == 'Higher (HL)'].copy()
    
    # 시간 관련 분석이 있다면
    if 'timestamp' in hl_cases.columns or 'datetime' in hl_cases.columns:
        time_col = 'timestamp' if 'timestamp' in hl_cases.columns else 'datetime'
        hl_cases[time_col] = pd.to_datetime(hl_cases[time_col])
        hl_cases['year'] = hl_cases[time_col].dt.year
        hl_cases['month'] = hl_cases[time_col].dt.month
        hl_cases['hour'] = hl_cases[time_col].dt.hour
        
        print("\n📅 HL 발생 연도별 분포:")
        year_dist = hl_cases['year'].value_counts().sort_index()
        for year, count in year_dist.items():
            pct = count / len(hl_cases) * 100
            print(f"  • {year}: {count:,}개 ({pct:.2f}%)")
        
        print("\n📅 HL 발생 월별 분포 (상위 5개월):")
        month_dist = hl_cases['month'].value_counts().sort_values(ascending=False).head(5)
        for month, count in month_dist.items():
            pct = count / len(hl_cases) * 100
            print(f"  • {month}월: {count:,}개 ({pct:.2f}%)")
    
    # L값 크기별 분석
    if 'L_value' in hl_cases.columns:
        print("\n💰 HL 발생 시 L값 범위 분포:")
        l_ranges = pd.cut(hl_cases['L_value'], bins=5)
        l_range_dist = l_ranges.value_counts().sort_index()
        for range_val, count in l_range_dist.items():
            pct = count / len(hl_cases) * 100
            print(f"  • {range_val}: {count:,}개 ({pct:.2f}%)")

print("\n" + "=" * 80)
print("3단계: HL 발생 전후 가격 움직임 분석")
print("=" * 80)

if all_l_values is not None and 'L_direction' in all_l_values.columns:
    # OHLCV 데이터 로드
    try:
        ohlcv = pd.read_csv('btc_15m_ohlcv.csv')
        ohlcv['datetime'] = pd.to_datetime(ohlcv['datetime'])
        print(f"✅ OHLCV 데이터 로드: {len(ohlcv):,}개 캔들")
        
        # HL 케이스와 조인
        time_col = 'timestamp' if 'timestamp' in all_l_values.columns else 'datetime'
        if time_col in all_l_values.columns:
            all_l_values[time_col] = pd.to_datetime(all_l_values[time_col])
            
            # HL만 필터링
            hl_with_price = all_l_values[all_l_values['L_direction'] == 'Higher (HL)'].copy()
            
            # 각 HL 시점의 전후 가격 움직임 분석
            hl_analysis = []
            
            for idx, row in hl_with_price.head(1000).iterrows():  # 샘플 1000개만
                hl_time = row[time_col]
                
                # HL 시점 캔들 찾기
                hl_candle = ohlcv[ohlcv['datetime'] == hl_time]
                if len(hl_candle) == 0:
                    continue
                    
                hl_candle = hl_candle.iloc[0]
                hl_idx = hl_candle.name
                
                # 이전 5개 캔들 (HL 발생 전)
                before_candles = ohlcv.iloc[max(0, hl_idx-5):hl_idx]
                
                # 이후 10개 캔들 (HL 발생 후)
                after_candles = ohlcv.iloc[hl_idx+1:min(len(ohlcv), hl_idx+11)]
                
                if len(before_candles) < 3 or len(after_candles) < 5:
                    continue
                
                # 가격 변화율 계산
                before_change = ((before_candles['close'].iloc[-1] - before_candles['close'].iloc[0]) / 
                                before_candles['close'].iloc[0] * 100)
                
                after_change_5 = ((after_candles['close'].iloc[4] - hl_candle['close']) / 
                                 hl_candle['close'] * 100) if len(after_candles) >= 5 else None
                
                after_change_10 = ((after_candles['close'].iloc[-1] - hl_candle['close']) / 
                                  hl_candle['close'] * 100) if len(after_candles) >= 10 else None
                
                # 최고점/최저점
                after_high = after_candles['high'].max() if len(after_candles) > 0 else hl_candle['high']
                after_high_pct = ((after_high - hl_candle['close']) / hl_candle['close'] * 100)
                
                after_low = after_candles['low'].min() if len(after_candles) > 0 else hl_candle['low']
                after_low_pct = ((after_low - hl_candle['close']) / hl_candle['close'] * 100)
                
                hl_analysis.append({
                    'timestamp': hl_time,
                    'L_change_pct': row['L_change_pct'],
                    'before_5_change': before_change,
                    'after_5_change': after_change_5,
                    'after_10_change': after_change_10,
                    'after_high_pct': after_high_pct,
                    'after_low_pct': after_low_pct
                })
            
            if len(hl_analysis) > 0:
                hl_df = pd.DataFrame(hl_analysis)
                
                print(f"\n📊 HL 발생 전후 가격 변화 (샘플 {len(hl_df):,}개):")
                print(f"\n  이전 5캔들 변화:")
                print(f"    • 평균: {hl_df['before_5_change'].mean():.3f}%")
                print(f"    • 중간값: {hl_df['before_5_change'].median():.3f}%")
                
                print(f"\n  이후 5캔들 변화:")
                print(f"    • 평균: {hl_df['after_5_change'].mean():.3f}%")
                print(f"    • 중간값: {hl_df['after_5_change'].median():.3f}%")
                print(f"    • 상승 비율: {(hl_df['after_5_change'] > 0).sum() / len(hl_df) * 100:.2f}%")
                
                print(f"\n  이후 10캔들 변화:")
                print(f"    • 평균: {hl_df['after_10_change'].mean():.3f}%")
                print(f"    • 중간값: {hl_df['after_10_change'].median():.3f}%")
                print(f"    • 상승 비율: {(hl_df['after_10_change'] > 0).sum() / len(hl_df) * 100:.2f}%")
                
                print(f"\n  이후 최대 상승:")
                print(f"    • 평균: {hl_df['after_high_pct'].mean():.3f}%")
                print(f"    • 중간값: {hl_df['after_high_pct'].median():.3f}%")
                
                print(f"\n  이후 최대 하락:")
                print(f"    • 평균: {hl_df['after_low_pct'].mean():.3f}%")
                print(f"    • 중간값: {hl_df['after_low_pct'].median():.3f}%")
                
                # 저장
                hl_df.to_csv('hl_price_movement_analysis.csv', index=False)
                print(f"\n💾 분석 결과 저장: hl_price_movement_analysis.csv")
                
    except Exception as e:
        print(f"❌ 가격 움직임 분석 실패: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("4단계: HL 발생과 매매 성과 상관관계")
print("=" * 80)

try:
    # 백테스트 결과 로드
    backtest = pd.read_csv('backtest_confirmation_space_results.csv')
    print(f"✅ 백테스트 결과 로드: {len(backtest):,}개 거래")
    
    if 'entry_time' in backtest.columns and all_l_values is not None:
        backtest['entry_time'] = pd.to_datetime(backtest['entry_time'])
        time_col = 'timestamp' if 'timestamp' in all_l_values.columns else 'datetime'
        all_l_values[time_col] = pd.to_datetime(all_l_values[time_col])
        
        # 각 거래에 대해 가장 가까운 HL 찾기
        trade_hl_analysis = []
        
        for idx, trade in backtest.head(100).iterrows():  # 샘플 100개
            entry_time = trade['entry_time']
            
            # 진입 전 가장 최근 HL 찾기
            recent_hl = all_l_values[
                (all_l_values['L_direction'] == 'Higher (HL)') & 
                (all_l_values[time_col] < entry_time)
            ].tail(1)
            
            if len(recent_hl) == 0:
                continue
            
            recent_hl = recent_hl.iloc[0]
            time_diff_hours = (entry_time - recent_hl[time_col]).total_seconds() / 3600
            
            trade_hl_analysis.append({
                'entry_time': entry_time,
                'exit_reason': trade['exit_reason'],
                'pnl_pct': trade['pnl_pct'],
                'hl_time_diff_hours': time_diff_hours,
                'hl_L_change': recent_hl['L_change_pct']
            })
        
        if len(trade_hl_analysis) > 0:
            trade_hl_df = pd.DataFrame(trade_hl_analysis)
            
            print(f"\n📊 HL 발생 후 거래 진입 시간 차이:")
            print(f"  • 평균: {trade_hl_df['hl_time_diff_hours'].mean():.2f}시간")
            print(f"  • 중간값: {trade_hl_df['hl_time_diff_hours'].median():.2f}시간")
            
            # 시간 차이별 성과
            print(f"\n📊 HL 발생 후 시간별 거래 성과:")
            
            # 시간 구간별 분류
            trade_hl_df['time_group'] = pd.cut(
                trade_hl_df['hl_time_diff_hours'],
                bins=[0, 2, 6, 12, 24, 1000],
                labels=['0-2h', '2-6h', '6-12h', '12-24h', '24h+']
            )
            
            for group in ['0-2h', '2-6h', '6-12h', '12-24h', '24h+']:
                group_data = trade_hl_df[trade_hl_df['time_group'] == group]
                if len(group_data) == 0:
                    continue
                    
                avg_pnl = group_data['pnl_pct'].mean()
                tp2_rate = (group_data['exit_reason'] == 'TP2_Full').sum() / len(group_data) * 100
                
                print(f"  • {group}: {len(group_data)}건, 평균 PNL {avg_pnl:.3f}%, TP2 성공률 {tp2_rate:.1f}%")
            
            trade_hl_df.to_csv('trade_hl_correlation.csv', index=False)
            print(f"\n💾 거래-HL 상관관계 저장: trade_hl_correlation.csv")
            
except Exception as e:
    print(f"❌ 거래 성과 상관관계 분석 실패: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
