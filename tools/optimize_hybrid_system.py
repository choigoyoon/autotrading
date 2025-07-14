import torch
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import sys
import json
import time
from tqdm import tqdm

# 프로젝트 설정
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings('ignore')

# === 🚀 디버깅 강화된 설정 ===
class DebugOptimizationConfig:
    """디버깅 강화된 최적화 설정"""
    
    PROJECT_ROOT = project_root
    MODEL_PATH = project_root / "models" / "pytorch26_transformer_v1.pth"
    SEQUENCES_DIR = project_root / "data" / "sequences_macd" / "test"
    OUTPUT_DIR = project_root / "results" / "hybrid_debug"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    MAX_SAMPLES = 1000
    
    # 🔥 수정된 임계값 (더 관대하게)
    AI_BUY_THRESHOLD = 0.3      # 매수 확률 임계값
    AI_SELL_THRESHOLD = 0.7     # 매도 확률 임계값
    CONFIDENCE_THRESHOLDS = [0.4, 0.5, 0.6]
    
    # 백테스트 설정
    INITIAL_CAPITAL = 10000
    POSITION_SIZE = 0.95  # 95% 투자

# === 🤖 디버깅 강화된 AI 예측기 ===
class DebugAIPredictor:
    """디버깅 강화된 AI 예측기"""
    
    def __init__(self, model_path, device):
        self.device = device
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        """모델 로드 및 검증"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 간소화된 모델 구조
            class SimpleTransformer(torch.nn.Module):
                def __init__(self, input_dim=8, d_model=64, nhead=4, num_encoder_layers=2, 
                             dim_feedforward=256, num_classes=2, dropout=0.1):
                    super().__init__()
                    
                    self.input_projection = torch.nn.Linear(input_dim, d_model)
                    self.positional_encoding = torch.nn.Parameter(torch.randn(500, d_model))
                    
                    encoder_layer = torch.nn.TransformerEncoderLayer(
                        d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                        dropout=dropout, batch_first=True
                    )
                    
                    self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
                    self.signal_head = torch.nn.Sequential(
                        torch.nn.Linear(d_model, d_model // 2),
                        torch.nn.ReLU(),
                        torch.nn.Linear(d_model // 2, num_classes)
                    )
                    self.dropout = torch.nn.Dropout(dropout)
                    
                def forward(self, x):
                    batch_size, seq_len, _ = x.shape
                    x = self.input_projection(x)
                    pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0)
                    x = x + pos_enc
                    x = self.transformer_encoder(x)
                    x = x.mean(dim=1)
                    x = self.dropout(x)
                    signal_pred = self.signal_head(x)
                    return signal_pred, torch.zeros(batch_size, 1), torch.zeros(batch_size, 1)
            
            model_config = checkpoint.get('config', {})
            model = SimpleTransformer(
                input_dim=model_config.get('input_dim', 8),
                d_model=model_config.get('d_model', 64)
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            print(f"✅ 모델 로드 완료 (정확도: {checkpoint.get('val_acc', 0):.4f})")
            return model
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return None
    
    def predict_batch_with_debug(self, features_batch):
        """디버깅 정보를 포함한 배치 예측"""
        if self.model is None:
            return None, None, None
            
        with torch.no_grad():
            signal_pred, _, _ = self.model(features_batch)
            probabilities = torch.softmax(signal_pred, dim=1)
            
            # 매수 확률, 신뢰도, 예측 클래스
            buy_probs = probabilities[:, 1].cpu().numpy()
            confidences = probabilities.max(dim=1)[0].cpu().numpy()
            predictions = probabilities.argmax(dim=1).cpu().numpy()
            
            return buy_probs, confidences, predictions

# === 📈 강화된 백테스트 엔진 ===
class RobustBacktester:
    """강화된 백테스트 엔진"""
    
    def __init__(self, initial_capital=10000, position_size=0.95):
        self.initial_capital = initial_capital
        self.position_size = position_size
        
    def run_backtest_with_debug(self, signals, prices, strategy_name="Unknown"):
        """디버깅 정보를 포함한 백테스트"""
        
        # 입력 검증
        if len(signals) != len(prices):
            print(f"❌ {strategy_name}: 신호와 가격 길이 불일치 ({len(signals)} vs {len(prices)})")
            return self.empty_result()
        
        # 신호 통계
        buy_signals = np.sum(signals == 1)
        sell_signals = np.sum(signals == -1)
        hold_signals = np.sum(signals == 0)
        
        print(f"📊 {strategy_name}: 매수 {buy_signals}, 매도 {sell_signals}, 관망 {hold_signals}")
        
        if buy_signals == 0 and sell_signals == 0:
            print(f"⚠️ {strategy_name}: 거래 신호 없음")
            return self.empty_result()
        
        # 백테스트 실행
        capital = float(self.initial_capital)
        position = 0.0
        trades = []
        equity_curve = [capital]
        
        for i in range(1, len(signals)):
            signal = signals[i]
            price = float(prices.iloc[i] if hasattr(prices, 'iloc') else prices[i])
            
            current_equity = capital + (position * price if position > 0 else 0)
            
            # 매수 신호
            if signal == 1 and position <= 0:
                if position < 0:  # 공매도 청산
                    capital += abs(position) * price
                    position = 0
                
                # 매수
                invest_amount = current_equity * self.position_size
                shares_to_buy = invest_amount / price
                position += shares_to_buy
                capital -= invest_amount
                
                trades.append(('BUY', price, shares_to_buy, current_equity))
            
            # 매도 신호
            elif signal == -1 and position > 0:
                # 매도
                capital += position * price
                trades.append(('SELL', price, position, current_equity))
                position = 0
            
            # 현재 자산가치 계산
            current_equity = capital + (position * price if position > 0 else 0)
            equity_curve.append(current_equity)
        
        # 최종 청산
        if position > 0:
            final_price = float(prices.iloc[-1] if hasattr(prices, 'iloc') else prices[-1])
            capital += position * final_price
            trades.append(('FINAL_SELL', final_price, position, capital))
        
        return self.calculate_performance_robust(equity_curve, trades, strategy_name)
    
    def calculate_performance_robust(self, equity_curve, trades, strategy_name):
        """강화된 성과 계산"""
        try:
            equity_series = pd.Series(equity_curve)
            
            if len(equity_series) < 2:
                return self.empty_result()
            
            # 기본 지표
            initial_value = equity_series.iloc[0]
            final_value = equity_series.iloc[-1]
            total_return = ((final_value / initial_value) - 1) * 100
            
            # 수익률 계산
            returns = equity_series.pct_change().dropna()
            
            if len(returns) == 0:
                return self.empty_result()
            
            # 연율화 가정 (일간 데이터라고 가정)
            annual_return = total_return
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
            
            # Sharpe Ratio (위험 없는 수익률 = 0으로 가정)
            sharpe_ratio = (annual_return / 100) / volatility if volatility > 0 else 0
            
            # 최대 낙폭
            cummax = equity_series.cummax()
            drawdown = (equity_series - cummax) / cummax * 100
            max_drawdown = drawdown.min()
            
            # 거래 통계
            num_trades = len(trades)
            
            # 승률 계산 (간소화)
            if num_trades > 2:
                profit_trades = 0
                for i in range(1, len(trades)):
                    if trades[i][0] == 'SELL' and i > 0:
                        buy_price = trades[i-1][1]
                        sell_price = trades[i][1]
                        if sell_price > buy_price:
                            profit_trades += 1
                win_rate = (profit_trades / (num_trades // 2)) * 100 if num_trades > 0 else 0
            else:
                win_rate = 0
            
            result = {
                'Total Return (%)': round(total_return, 4),
                'Annual Return (%)': round(annual_return, 4),
                'Volatility (%)': round(volatility * 100, 4),
                'Sharpe Ratio': round(sharpe_ratio, 4),
                'Max Drawdown (%)': round(max_drawdown, 4),
                'Number of Trades': num_trades,
                'Win Rate (%)': round(win_rate, 2),
                'Final Equity': round(final_value, 2),
                'Equity Curve': equity_series
            }
            
            print(f"📈 {strategy_name}: 수익률 {total_return:.2f}%, 거래 {num_trades}회, Sharpe {sharpe_ratio:.3f}")
            return result
            
        except Exception as e:
            print(f"❌ {strategy_name} 성과 계산 오류: {e}")
            return self.empty_result()
    
    def empty_result(self):
        """빈 결과 반환"""
        return {
            'Total Return (%)': 0.0,
            'Annual Return (%)': 0.0,
            'Volatility (%)': 0.0,
            'Sharpe Ratio': 0.0,
            'Max Drawdown (%)': 0.0,
            'Number of Trades': 0,
            'Win Rate (%)': 0.0,
            'Final Equity': 10000.0,
            'Equity Curve': pd.Series([10000.0])
        }

# === 🔍 통합 분석기 ===
class DebugHybridAnalyzer:
    """디버깅 강화된 하이브리드 분석기"""
    
    def __init__(self, config):
        self.config = config
        self.ai_predictor = DebugAIPredictor(config.MODEL_PATH, config.DEVICE)
        self.backtester = RobustBacktester(config.INITIAL_CAPITAL, config.POSITION_SIZE)
        
    def run_complete_analysis(self):
        """완전한 분석 실행"""
        
        print("🚀 디버깅 강화된 하이브리드 분석 시작!")
        
        # 1. 데이터 로드
        features_tensor, price_series = self.load_and_validate_data()
        if features_tensor is None:
            return
        
        # 2. AI 예측
        buy_probs, confidences, predictions = self.ai_predictor.predict_batch_with_debug(
            features_tensor.to(self.config.DEVICE)
        )
        
        if buy_probs is None:
            print("❌ AI 예측 실패")
            return
        
        # 3. 예측 결과 분석
        self.analyze_predictions(buy_probs, confidences, predictions)
        
        # 4. 신호 생성 및 분석
        strategies = self.create_robust_strategies(buy_probs, confidences, price_series)
        
        # 5. 백테스트
        results = self.run_comprehensive_backtest(strategies, price_series)
        
        # 6. 결과 분석 및 저장
        self.analyze_and_save_results(results)
        
        return results
    
    def load_and_validate_data(self):
        """데이터 로드 및 검증"""
        print("📊 데이터 로드 및 검증...")
        
        sequence_files = list(self.config.SEQUENCES_DIR.glob("*.pt"))[:self.config.MAX_SAMPLES]
        
        if not sequence_files:
            print(f"❌ 시퀀스 파일 없음: {self.config.SEQUENCES_DIR}")
            return None, None
        
        features_list = []
        prices_list = []
        timestamps = []
        
        for i, file_path in enumerate(tqdm(sequence_files, desc="데이터 로딩")):
            try:
                data = torch.load(file_path, map_location='cpu', weights_only=False)
                features_list.append(data['features'])
                
                # 마지막 종가를 가격으로 사용
                last_close = data['features'][-1, 3].item()  # close price
                prices_list.append(last_close)
                
                # 타임스탬프 (가상)
                timestamps.append(pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=i))
                
            except Exception as e:
                print(f"⚠️ 파일 로드 실패: {file_path}")
                continue
        
        if not features_list:
            print("❌ 유효한 데이터 없음")
            return None, None
        
        features_tensor = torch.stack(features_list)
        price_series = pd.Series(prices_list, index=timestamps)
        
        print(f"✅ 데이터 로드 완료: {len(features_list)}개 샘플")
        print(f"   가격 범위: ${price_series.min():.2f} - ${price_series.max():.2f}")
        
        return features_tensor, price_series
    
    def analyze_predictions(self, buy_probs, confidences, predictions):
        """AI 예측 결과 분석"""
        print("\n🤖 AI 예측 분석:")
        print(f"   매수 확률 범위: {buy_probs.min():.4f} - {buy_probs.max():.4f}")
        print(f"   평균 매수 확률: {buy_probs.mean():.4f}")
        print(f"   신뢰도 범위: {confidences.min():.4f} - {confidences.max():.4f}")
        print(f"   예측 분포: 매도 {np.sum(predictions == 0)}개, 매수 {np.sum(predictions == 1)}개")
        
        # 임계값별 신호 수 분석
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            buy_signals = np.sum(buy_probs > threshold)
            print(f"   매수 확률 > {threshold}: {buy_signals}개 ({buy_signals/len(buy_probs)*100:.1f}%)")
    
    def create_robust_strategies(self, buy_probs, confidences, price_series):
        """강화된 전략 생성"""
        print("\n⚡ 강화된 전략 생성...")
        
        strategies = {}
        
        # 1. AI 단독 전략 (다양한 임계값)
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        for threshold in thresholds:
            signals = np.where(buy_probs > threshold, 1, 
                              np.where(buy_probs < (1-threshold), -1, 0))
            strategies[f'AI_Threshold_{threshold}'] = signals
        
        # 2. 신뢰도 기반 전략
        for conf_thresh in self.config.CONFIDENCE_THRESHOLDS:
            signals = np.where(
                (buy_probs > 0.5) & (confidences > conf_thresh), 1,
                np.where((buy_probs < 0.5) & (confidences > conf_thresh), -1, 0)
            )
            strategies[f'AI_Confidence_{conf_thresh}'] = signals
        
        # 3. 간단한 모멘텀 전략 (MACD 대신)
        returns = price_series.pct_change().fillna(0)
        momentum_signals = np.where(returns > 0, 1, -1)
        strategies['Momentum'] = momentum_signals
        
        # 4. 하이브리드 전략
        ai_signals = np.where(buy_probs > 0.4, 1, -1)
        hybrid_signals = np.where(
            (ai_signals == momentum_signals) & (confidences > 0.5), 
            ai_signals, 0
        )
        strategies['Hybrid_Consensus'] = hybrid_signals
        
        # 전략 요약
        for name, signals in strategies.items():
            buy_count = np.sum(signals == 1)
            sell_count = np.sum(signals == -1)
            hold_count = np.sum(signals == 0)
            print(f"   {name}: 매수 {buy_count}, 매도 {sell_count}, 관망 {hold_count}")
        
        return strategies
    
    def run_comprehensive_backtest(self, strategies, price_series):
        """포괄적 백테스트 실행"""
        print("\n🔍 백테스트 실행...")
        
        results = {}
        
        for name, signals in strategies.items():
            try:
                result = self.backtester.run_backtest_with_debug(signals, price_series, name)
                results[name] = result
            except Exception as e:
                print(f"❌ {name} 백테스트 실패: {e}")
        
        return results
    
    def analyze_and_save_results(self, results):
        """결과 분석 및 저장"""
        if not results:
            print("❌ 분석할 결과 없음")
            return
        
        # 결과 DataFrame 생성
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('Sharpe Ratio', ascending=False)
        
        print("\n" + "="*60)
        print("📈 최종 분석 결과")
        print("="*60)
        
        print("\n🏆 상위 5개 전략:")
        display_df = results_df.drop('Equity Curve', axis=1).head()
        print(display_df.round(4))
        
        # 최고 성과 전략
        best_strategy = results_df.index[0]
        best_performance = results_df.iloc[0]
        
        print(f"\n🥇 최고 성과 전략: {best_strategy}")
        for key, value in best_performance.items():
            if key != 'Equity Curve':
                print(f"   {key}: {value}")
        
        # 결과 저장
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        results_path = self.config.OUTPUT_DIR / "comprehensive_results.csv"
        results_df.drop('Equity Curve', axis=1).to_csv(results_path)
        
        summary = {
            'best_strategy': best_strategy,
            'best_sharpe_ratio': float(best_performance['Sharpe Ratio']),
            'best_total_return': float(best_performance['Total Return (%)']),
            'total_strategies': len(results),
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_path = self.config.OUTPUT_DIR / "analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📁 결과 저장: {results_path}")
        print(f"📁 요약 저장: {summary_path}")

def main():
    """메인 실행"""
    print("🚀 디버깅 강화된 하이브리드 최적화!")
    
    config = DebugOptimizationConfig()
    analyzer = DebugHybridAnalyzer(config)
    
    results = analyzer.run_complete_analysis()
    
    if results:
        print("\n🎉 분석 완료!")
    else:
        print("\n❌ 분석 실패")

if __name__ == '__main__':
    main()
