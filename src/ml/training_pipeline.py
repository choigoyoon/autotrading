# src/ml/training_pipeline.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split # type: ignore
import numpy as np
from typing import Dict, Any, List

from .data_preprocessor import MultiTimeframeDataPreprocessor
from .multi_timeframe_predictor import MultiTimeframePredictor

class TrainingPipeline:
    """
    데이터 전처리, 모델 학습, 검증, 저장을 포함하는 전체 학습 파이프라인.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        timeframes: List[str] = self.config.get("timeframes", [])
        self.model = MultiTimeframePredictor(
            num_timeframes=len(timeframes),
            input_features=self.config.get("input_features", 5)
        ).to(self.device)

    def run(self):
        """파이프라인을 실행합니다."""
        # 1. 데이터 준비
        print("Preparing data...")
        preprocessor = MultiTimeframeDataPreprocessor(self.config)
        features, labels, returns = preprocessor.run()
        
        # 데이터가 없는 경우 중단
        if not features:
            print("No data to train on. Exiting.")
            return
        
        # stratify를 위해 labels가 비어있지 않은지 확인
        if not labels:
            print("No labels to stratify on. Splitting without stratification.")
            X_train, X_test, y_train, y_test, returns_train, returns_test = train_test_split(
                features, labels, returns, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test, returns_train, returns_test = train_test_split(
                features, labels, returns, test_size=0.2, random_state=42, stratify=labels
            )

        # PyTorch DataLoader 생성
        train_loader = self._create_dataloader(X_train, y_train, returns_train, self.config.get("batch_size", 32))
        test_loader = self._create_dataloader(X_test, y_test, returns_test, self.config.get("batch_size", 32))

        # 2. 손실 함수 및 옵티마이저 정의
        loss_weights = self.config.get("loss_weights", {'classification': 0.7, 'regression': 0.3})
        criterion_class = nn.CrossEntropyLoss()
        criterion_reg = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.get("learning_rate", 0.001))

        # 3. 모델 학습
        print("Starting model training...")
        self._train_model(train_loader, test_loader, criterion_class, criterion_reg, optimizer, loss_weights)

        # 4. 모델 저장
        model_path = self.config.get("model_save_path", "models/multi_timeframe_predictor.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def _create_dataloader(self, features: list, labels: list, returns: list, batch_size: int) -> DataLoader:
        """데이터로부터 PyTorch DataLoader를 생성합니다."""
        features_tensor = torch.tensor(np.array(features), dtype=torch.float32)
        labels_tensor = torch.tensor(np.array(labels), dtype=torch.long)
        returns_tensor = torch.tensor(np.array(returns), dtype=torch.float32).unsqueeze(1)
        timeframe_ids = torch.arange(features_tensor.shape[1]).unsqueeze(0).repeat(features_tensor.shape[0], 1)
        
        dataset = TensorDataset(features_tensor, timeframe_ids, labels_tensor, returns_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def _train_model(self, train_loader, test_loader, crit_class, crit_reg, optimizer, loss_weights):
        """실제 학습 루프."""
        epochs = self.config.get("epochs", 10)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for features, tf_ids, labels, returns in train_loader:
                features, tf_ids, labels, returns = features.to(self.device), tf_ids.to(self.device), labels.to(self.device), returns.to(self.device)
                
                optimizer.zero_grad()
                
                prob, ret, _ = self.model(features, tf_ids)
                
                # 멀티태스크 손실 계산
                loss_c = crit_class(prob, labels)
                loss_r = crit_reg(ret, returns)
                
                loss = (loss_weights['classification'] * loss_c) + (loss_weights['regression'] * loss_r)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

            # TODO: Add validation loop and save best model logic

if __name__ == '__main__':
    config = {
        "label_base_path": "data/labels/btc_usdt_kst",
        "holding_trades_path": "results/top_trades/top_100_holding_trades.csv",
        "breakeven_trades_path": "results/top_trades/top_100_breakeven_trades.csv",
        "timeframes": [
            '1min', '3min', '5min', '10min', '15min', '30min', '1h', '2h', '4h', 
            '6h', '8h', '12h', '1day', '3day', '1week', '1month'
        ],
        "input_features": 5,
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": 0.001,
        "model_save_path": "models/multi_timeframe_predictor.pth"
    }
    
    pipeline = TrainingPipeline(config)
    pipeline.run() 