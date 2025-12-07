import pandas as pd

# 방금 생성한 거래 내역 확인
trades = pd.read_csv('trades_4h_15m.csv')

print("=" * 60)
print("승률 계산 검증")
print("=" * 60)

# 방식 1: PNL > 0
win_pnl = (trades['pnl'] > 0).sum()
total = len(trades)
winrate_pnl = win_pnl / total * 100

print(f"\n[방식 1] PNL > 0 기준:")
print(f"  승: {win_pnl}건")
print(f"  패: {total - win_pnl}건")
print(f"  승률: {winrate_pnl:.1f}%")

# 방식 2: BE를 승리로 포함
win_with_be = ((trades['pnl'] > 0) | (trades['result'] == 'BE')).sum()
winrate_with_be = win_with_be / total * 100

print(f"\n[방식 2] PNL > 0 또는 BE 기준:")
print(f"  승+BE: {win_with_be}건")
print(f"  패: {total - win_with_be}건")
print(f"  승률: {winrate_with_be:.1f}%")

# 결과 분포
print(f"\n결과 분포:")
result_dist = trades['result'].value_counts()
for result, count in result_dist.items():
    pct = count / total * 100
    avg_pnl = trades[trades['result'] == result]['pnl'].mean()
    print(f"  {result}: {count}건 ({pct:.1f}%), 평균 PNL {avg_pnl:.3f}%")
