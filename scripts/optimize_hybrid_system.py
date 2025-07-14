import torch
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import sys
import json
import time
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Tuple, TypedDict
import numpy.typing as npt
from torch import nn
import matplotlib.pyplot as plt

# 프로젝트 설정
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

warnings.filterwarnings('ignore')

# --- 타입 정의 ---

class ModelCheckpoint(TypedDict):
    config: Dict[str, Any]
    model_state_dict: Dict[str, torch.Tensor]
    val_acc: float

class BacktestResult(TypedDict, total=False):
    total_return_pct: float
    annual_return_pct: float
    volatility_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    num_trades: int
    win_rate_pct: float
    final_equity: float
    equity_curve: pd.Series

# --- 모델 클래스 ---

class SimpleTransformer(nn.Module):
    """간소화된 트랜스포머 모델 구조"""
    def __init__(self, input_dim: int = 8, d_model: int = 64, nhead: int = 4, 
                 num_encoder_layers: int = 2, dim_feedforward: int = 256, 
                 num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(500, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.signal_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        x_proj = self.input_projection(x)
        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0)
        encoded = x_proj + pos_enc
        encoded = self.transformer_encoder(encoded)
        features = encoded.mean(dim=1)
        features = self.dropout(features)
        signal_pred = self.signal_head(features)
        
        # 더미 반환값 추가 (기존 모델 구조와 호환성을 위해)
        dummy_return = torch.zeros(batch_size, 1, device=x.device)
        dummy_confidence = torch.zeros(batch_size, 1, device=x.device)
        
        return signal_pred, dummy_return, dummy_confidence

# === 🚀 디버깅 강화된 설정 ===
class DebugOptimizationConfig:
    """디버깅 강화된 최적화 설정"""
    
    PROJECT_ROOT: Path = project_root
    MODEL_PATH: Path = project_root / "models" / "pytorch26_transformer_v1.pth"
    SEQUENCES_DIR: Path = project_root / "data" / "sequences_macd" / "test"
    OUTPUT_DIR: Path = project_root / "results" / "hybrid_debug"
    
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE: int = 64
    MAX_SAMPLES: int = 1000
    
    # 🔥 수정된 임계값 (더 관대하게)
    AI_BUY_THRESHOLD: float = 0.3      # 매수 확률 임계값
    AI_SELL_THRESHOLD: float = 0.7     # 매도 확률 임계값
    CONFIDENCE_THRESHOLDS: List[float] = [0.4, 0.5, 0.6]
    
    # 백테스트 설정
    INITIAL_CAPITAL: float = 10000.0
    POSITION_SIZE: float = 0.95  # 95% 투자

# === 🤖 디버깅 강화된 AI 예측기 ===
class DebugAIPredictor:
    """디버깅 강화된 AI 예측기"""
    
    def __init__(self, model_path: Path, device: torch.device):
        self.device = device
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path: Path) -> Optional[SimpleTransformer]:
        """모델 로드 및 검증"""
        if not model_path.exists():
            print(f"❌ 모델 파일 없음: {model_path}")
            return None
        try:
            checkpoint: ModelCheckpoint = torch.load(model_path, map_location=self.device, weights_only=False) # type: ignore
            
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
    
    def predict_batch_with_debug(self, features_batch: torch.Tensor) -> Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int_]]]:
        """디버깅 정보를 포함한 배치 예측"""
        if self.model is None:
            return None
            
        with torch.no_grad():
            signal_pred, _, _ = self.model(features_batch)
            probabilities = torch.softmax(signal_pred, dim=1)
            
            # 매수 확률, 신뢰도, 예측 클래스
            buy_probs: npt.NDArray[np.float64] = probabilities[:, 1].cpu().numpy()
            confidences: npt.NDArray[np.float64] = probabilities.max(dim=1)[0].cpu().numpy()
            predictions: npt.NDArray[np.int_] = probabilities.argmax(dim=1).cpu().numpy()
            
            return buy_probs, confidences, predictions

# === 📈 강화된 백테스트 엔진 ===
class RobustBacktester:
    """강화된 백테스트 엔진"""
    
    def __init__(self, initial_capital: float = 10000.0, position_size: float = 0.95):
        self.initial_capital = initial_capital
        self.position_size = position_size
        
    def run_backtest_with_debug(self, signals: npt.NDArray[np.int_], prices: pd.Series, strategy_name: str = "Unknown") -> BacktestResult:
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
        capital = self.initial_capital
        position: float = 0.0
        trades: List[Tuple[str, float, float, float]] = []
        equity_curve = [capital]
        
        price_values = prices.values
        for i in range(1, len(signals)):
            signal = signals[i]
            price = price_values[i]
            
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
            final_price = price_values[-1]
            capital += position * final_price
            trades.append(('FINAL_SELL', final_price, position, capital))
        
        return self.calculate_performance_robust(equity_curve, trades, strategy_name)
    
    def calculate_performance_robust(self, equity_curve: List[float], trades: List[Tuple[str, float, float, float]], strategy_name: str) -> BacktestResult:
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
            
            if returns.empty:
                return self.empty_result()
            
            # 연율화 가정 (일간 데이터라고 가정)
            annual_return = total_return
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
            
            # Sharpe Ratio (위험 없는 수익률 = 0으로 가정)
            sharpe_ratio = (annual_return / 100) / volatility if volatility > 0 else 0.0
            
            # 최대 낙폭
            cummax = equity_series.cummax()
            drawdown = (equity_series - cummax) / cummax * 100
            max_drawdown = drawdown.min()
            
            # 거래 통계
            num_trades = len(trades)
            
            # 승률 계산 (간소화)
            win_rate = 0.0
            if num_trades > 1:
                profit_trades = 0
                trade_count_for_win_rate = 0
                for i in range(len(trades)):
                    if trades[i][0] == 'SELL':
                        # 가장 가까운 이전 BUY를 찾음
                        for j in range(i - 1, -1, -1):
                            if trades[j][0] == 'BUY':
                                buy_price = trades[j][1]
                                sell_price = trades[i][1]
                                if sell_price > buy_price:
                                    profit_trades += 1
                                trade_count_for_win_rate += 1
                                break # 다음 SELL을 위해 내부 루프 탈출
                if trade_count_for_win_rate > 0:
                    win_rate = (profit_trades / trade_count_for_win_rate) * 100

            result: BacktestResult = {
                'total_return_pct': round(total_return, 4),
                'annual_return_pct': round(annual_return, 4),
                'volatility_pct': round(volatility * 100, 4),
                'sharpe_ratio': round(sharpe_ratio, 4),
                'max_drawdown_pct': round(max_drawdown, 4),
                'num_trades': num_trades,
                'win_rate_pct': round(win_rate, 2),
                'final_equity': round(final_value, 2),
                'equity_curve': equity_series
            }
            
            print(f"📈 {strategy_name}: 수익률 {total_return:.2f}%, 거래 {num_trades}회, Sharpe {sharpe_ratio:.3f}")
            return result
        except Exception as e:
            print(f"❌ 성과 계산 오류 ({strategy_name}): {e}")
            return self.empty_result()
            
    def empty_result(self) -> BacktestResult:
        """빈 결과 반환"""
        return {
            'total_return_pct': 0.0,
            'annual_return_pct': 0.0,
            'volatility_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'num_trades': 0,
            'win_rate_pct': 0.0,
            'final_equity': self.initial_capital,
            'equity_curve': pd.Series([self.initial_capital])
        }

# === 🧩 하이브리드 전략 분석기 ===
class DebugHybridAnalyzer:
    """디버깅 강화된 하이브리드 전략 분석기"""
    
    def __init__(self, config: DebugOptimizationConfig):
        self.config = config
        self.predictor = DebugAIPredictor(config.MODEL_PATH, config.DEVICE)
        self.backtester = RobustBacktester(config.INITIAL_CAPITAL, config.POSITION_SIZE)
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def run_complete_analysis(self) -> None:
        """전체 분석 파이프라인 실행"""
        print("\n" + "="*20 + " 하이브리드 전략 분석 시작 " + "="*20)
        start_time = time.time()
        
        # 1. 데이터 로드
        features_dict, price_series = self.load_and_validate_data()
        if not features_dict:
            return

        # 2. AI 예측
        features_tensor = torch.tensor(np.array(list(features_dict.values())), dtype=torch.float32).to(self.config.DEVICE)
        prediction_result = self.predictor.predict_batch_with_debug(features_tensor)
        if prediction_result is None:
            print("❌ AI 예측 실패. 분석을 중단합니다.")
            return
        
        buy_probs, confidences, _ = prediction_result
        
        # 3. 예측 결과 분석 및 신호 생성
        all_signals = self.analyze_predictions(buy_probs, confidences, price_series.index)
        
        # 4. 전략 생성
        strategies = self.create_robust_strategies(all_signals, price_series)
        
        # 5. 백테스팅
        results = self.run_comprehensive_backtest(strategies, price_series)
        
        # 6. 결과 저장 및 시각화
        self.analyze_and_save_results(results)
        
        print(f"\n✅ 전체 분석 완료. (소요 시간: {time.time() - start_time:.2f}초)")
        print("="*60 + "\n")

    def load_and_validate_data(self) -> Tuple[Dict[str, npt.NDArray[np.float32]], pd.Series]:
        """시퀀스 데이터와 가격 데이터 로드 및 검증"""
        print("\n--- 1. 데이터 로드 및 검증 ---")
        sequence_files = sorted(list(self.config.SEQUENCES_DIR.glob("*.pt")))
        if not sequence_files:
            print(f"❌ 시퀀스 파일 없음: {self.config.SEQUENCES_DIR}")
            return {}, pd.Series(dtype=np.float64)

        if self.config.MAX_SAMPLES > 0:
            sequence_files = sequence_files[:self.config.MAX_SAMPLES]
        
        features_dict: Dict[str, npt.NDArray[np.float32]] = {}
        timestamps: List[pd.Timestamp] = []

        for file in tqdm(sequence_files, desc="데이터 로드 중"):
            try:
                data = torch.load(file, map_location='cpu')
                features_dict[file.stem] = data['features'].numpy().astype(np.float32)
                timestamp_str = file.stem.split('_')[0]
                timestamps.append(pd.to_datetime(timestamp_str))
            except Exception as e:
                print(f"⚠️ 파일 로드 오류 {file.name}: {e}")

        if not features_dict:
            print("❌ 유효한 데이터를 로드하지 못했습니다.")
            return {}, pd.Series(dtype=np.float64)

        # 가격 데이터 생성 (임시. 실제 데이터 소스와 연결 필요)
        price_data = 20000 + np.random.randn(len(timestamps)).cumsum()
        price_series = pd.Series(price_data, index=pd.Index(timestamps)).sort_index()

        print(f"✅ 데이터 로드 완료: {len(features_dict)}개 샘플")
        return features_dict, price_series

    def analyze_predictions(self, buy_probs: npt.NDArray[np.float64], confidences: npt.NDArray[np.float64], index: pd.Index) -> pd.DataFrame:
        """AI 예측을 기반으로 다양한 신호 생성"""
        print("\n--- 2. AI 예측 분석 및 신호 생성 ---")
        df = pd.DataFrame(index=index)
        df['buy_prob'] = buy_probs
        df['confidence'] = confidences
        
        df['ai_signal'] = np.where(df['buy_prob'] >= self.config.AI_SELL_THRESHOLD, -1, 
                                 np.where(df['buy_prob'] <= self.config.AI_BUY_THRESHOLD, 1, 0))

        # MACD 신호 (임시. 실제 데이터와 연동 필요)
        df['macd_signal'] = np.random.choice([-1, 0, 1], size=len(df), p=[0.05, 0.9, 0.05])
        
        print(f"   - AI 신호: 매수 {np.sum(df['ai_signal'] == 1)}, 매도 {np.sum(df['ai_signal'] == -1)}")
        print(f"   - MACD 신호: 매수 {np.sum(df['macd_signal'] == 1)}, 매도 {np.sum(df['macd_signal'] == -1)}")
        
        return df

    def create_robust_strategies(self, all_signals: pd.DataFrame, price_series: pd.Series) -> Dict[str, npt.NDArray[np.int_]]:
        """다양한 하이브리드 전략 생성"""
        print("\n--- 3. 하이브리드 전략 생성 ---")
        strategies: Dict[str, npt.NDArray[np.int_]] = {
            "AI_Only": all_signals['ai_signal'].to_numpy(dtype=np.int_),
            "MACD_Only": all_signals['macd_signal'].to_numpy(dtype=np.int_),
            "Hybrid_AND": np.where(all_signals['ai_signal'] == all_signals['macd_signal'], all_signals['ai_signal'], 0).astype(np.int_),
            "Hybrid_OR": np.where(all_signals['ai_signal'] != 0, all_signals['ai_signal'], all_signals['macd_signal']).astype(np.int_)
        }

        # 신뢰도 기반 필터링 전략
        for conf in self.config.CONFIDENCE_THRESHOLDS:
            strategy_name = f"AI_CONF_{conf:.1f}+"
            strategies[strategy_name] = np.where(all_signals['confidence'] >= conf, all_signals['ai_signal'], 0).astype(np.int_)

        print(f"✅ {len(strategies)}개 전략 생성 완료.")
        return strategies

    def run_comprehensive_backtest(self, strategies: Dict[str, npt.NDArray[np.int_]], price_series: pd.Series) -> Dict[str, BacktestResult]:
        """모든 전략에 대한 백테스트 실행"""
        print("\n--- 4. 종합 백테스팅 실행 ---")
        results: Dict[str, BacktestResult] = {}
        for name, signals in tqdm(strategies.items(), desc="백테스팅 중"):
            results[name] = self.backtester.run_backtest_with_debug(signals, price_series, name)
        return results

    def analyze_and_save_results(self, results: Dict[str, BacktestResult]) -> None:
        """결과 분석, 저장 및 시각화"""
        print("\n--- 5. 결과 분석 및 저장 ---")
        
        # DataFrame으로 변환 (Equity Curve 제외)
        report_data = {name: {k: v for k, v in res.items() if k != 'equity_curve'} for name, res in results.items()}
        report_df = pd.DataFrame(report_data).T
        
        # Sharpe Ratio 기준으로 정렬
        report_df = report_df.sort_values(by='sharpe_ratio', ascending=False)
        
        print("\n=== 백테스트 결과 요약 ===")
        print(report_df.to_string(float_format="%.3f"))
        
        # CSV 파일로 저장
        report_path = self.config.OUTPUT_DIR / "hybrid_performance_report.csv"
        report_df.to_csv(report_path)
        print(f"\n💾 리포트 저장 완료: {report_path}")

        # 시각화
        fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
        for name, result in results.items():
            equity_curve = result.get('equity_curve')
            if equity_curve is not None and not equity_curve.empty:
                sharpe = result.get('sharpe_ratio', 0.0)
                ret = result.get('total_return_pct', 0.0)
                label = f"{name} (Sharpe: {sharpe:.2f}, Return: {ret:.1f}%)"
                ax.plot(equity_curve.index, equity_curve.to_numpy(), label=label, lw=1.5)

        ax.set_title("Hybrid Strategy Performance Comparison", fontsize=16, weight='bold')
        ax.set_ylabel("Portfolio Value")
        ax.set_xlabel("Date")
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()

        plot_path = self.config.OUTPUT_DIR / "hybrid_equity_curves.png"
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"📊 플롯 저장 완료: {plot_path}")

def main() -> None:
    """메인 실행 함수"""
    config = DebugOptimizationConfig()
    analyzer = DebugHybridAnalyzer(config)
    analyzer.run_complete_analysis()

if __name__ == '__main__':
    main()
