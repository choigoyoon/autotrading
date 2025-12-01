import pandas as pd

trades = pd.read_csv('trades_4h_15m.csv')

print("=" * 70)
print("정확한 승률 계산 (TP1 일부익절 포함)")
print("=" * 70)

# 현재 데이터에 TP1 도달 정보가 있는지 확인
print(f"\n컬럼: {list(trades.columns)}")
print(f"\n첫 5개 샘플:")
print(trades.head())

# 결과별 평균 PNL 확인
print("\n결과별 평균 PNL:")
for result in trades['result'].unique():
    result_trades = trades[trades['result'] == result]
    avg_pnl = result_trades['pnl'].mean()
    min_pnl = result_trades['pnl'].min()
    max_pnl = result_trades['pnl'].max()
    count = len(result_trades)
    print(f"  {result}: {count}건, 평균 {avg_pnl:.3f}%, 범위 [{min_pnl:.3f}% ~ {max_pnl:.3f}%]")

# 승률 계산 방식들
print("\n" + "=" * 70)
print("승률 계산 방식 비교")
print("=" * 70)

total = len(trades)

# 방식 1: PNL > 0
win_pnl = (trades['pnl'] > 0).sum()
winrate_1 = win_pnl / total * 100
print(f"\n[방식 1] PNL > 0만 승리:")
print(f"  승: {win_pnl}건 / 패: {total - win_pnl}건")
print(f"  승률: {winrate_1:.1f}%")
print(f"  문제: BE(본절)와 일부익절을 패배로 취급")

# 방식 2: PNL >= 0 (BE 포함)
win_pnl_zero = (trades['pnl'] >= 0).sum()
winrate_2 = win_pnl_zero / total * 100
print(f"\n[방식 2] PNL >= 0 (BE 포함):")
print(f"  승: {win_pnl_zero}건 / 패: {total - win_pnl_zero}건")
print(f"  승률: {winrate_2:.1f}%")
print(f"  문제: BE를 승리로 봐야 하나?")

# 방식 3: SL이 아니면 승리
win_not_sl = (trades['result'] != 'SL').sum()
winrate_3 = win_not_sl / total * 100
print(f"\n[방식 3] SL이 아니면 승리:")
print(f"  승: {win_not_sl}건 / 패: {total - win_not_sl}건")
print(f"  승률: {winrate_3:.1f}%")
print(f"  문제: TIME 음수도 승리로 취급")

# TIME의 PNL 분포 확인
time_trades = trades[trades['result'] == 'TIME']
time_positive = (time_trades['pnl'] > 0).sum()
time_zero = (time_trades['pnl'] == 0).sum()
time_negative = (time_trades['pnl'] < 0).sum()

print(f"\n[TIME 결과 상세 분석]:")
print(f"  양수: {time_positive}건 ({time_positive/len(time_trades)*100:.1f}%)")
print(f"  0: {time_zero}건")
print(f"  음수: {time_negative}건 ({time_negative/len(time_trades)*100:.1f}%)")

# 방식 4: TP2 + BE + TIME(양수) = 승리 (가장 정확)
win_accurate = ((trades['result'] == 'TP2') | 
                (trades['result'] == 'BE') | 
                ((trades['result'] == 'TIME') & (trades['pnl'] > 0))).sum()
winrate_4 = win_accurate / total * 100

print(f"\n[방식 4] TP2 + BE + TIME(양수) = 승리 ✅:")
print(f"  승: {win_accurate}건 / 패: {total - win_accurate}건")
print(f"  승률: {winrate_4:.1f}%")
print(f"  설명: TP1 도달 후 본절/수익은 승리, 손실만 패배")

# 패배 구성
losses = total - win_accurate
sl_count = (trades['result'] == 'SL').sum()
time_neg_count = time_negative

print(f"\n[패배 구성]:")
print(f"  SL: {sl_count}건")
print(f"  TIME(음수): {time_neg_count}건")
print(f"  합계: {losses}건")

print("\n" + "=" * 70)
print("✅ 최종 권장: 방식 4 (TP1 도달 효과 반영)")
print("=" * 70)
print(f"\n승률: {winrate_4:.1f}%")
print("이유: TP1 도달 시 손절을 본절로 올렸으므로,")
print("     BE와 TIME(양수)는 '손실 방어 성공' = 승리로 봐야 함")
