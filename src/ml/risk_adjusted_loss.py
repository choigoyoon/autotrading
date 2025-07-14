import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class LossConfig:
    """Configuration for risk-adjusted loss calculation"""
    # Loss weights
    direction_weight: float = 0.3
    magnitude_weight: float = 0.2
    duration_weight: float = 0.1
    confidence_weight: float = 0.2
    risk_weight: float = 0.2
    
    # Risk adjustment parameters
    max_drawdown_penalty: float = 1.0
    sharpe_ratio_weight: float = 0.5
    sortino_ratio_weight: float = 0.5
    
    # Margin parameters
    margin: float = 0.1
    
    # Label smoothing
    label_smoothing: float = 0.1
    
    # New attributes for risk aversion
    loss_type: str = 'sharpe'  # 'sharpe', 'sortino', 'calmar'
    risk_free_rate: float = 0.01
    target_sharpe: float = 1.0
    risk_aversion: float = 1.0
    
    def to_dict(self):
        return self.__dict__
    
    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)

class RiskAdjustedLoss(nn.Module):
    """
    Risk-adjusted loss function that combines multiple objectives:
    1. Direction classification (cross-entropy)
    2. Magnitude regression (smooth L1)
    3. Duration regression (smooth L1)
    4. Confidence calibration (binary cross-entropy)
    5. Risk adjustment (custom)
    """
    
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        
        # Base losses
        self.direction_loss = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.confidence_loss = nn.BCEWithLogitsLoss()
        
        # Smooth L1 loss is more robust to outliers than MSE
        self.regression_loss = nn.SmoothL1Loss(reduction='none')
        
        # Risk metrics
        self.sharpe_ratio = self._sharpe_ratio
        self.sortino_ratio = self._sortino_ratio
    
    def _sharpe_ratio(self, returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate
        return torch.mean(excess_returns) / (torch.std(returns) + 1e-6)
    
    def _sortino_ratio(self, returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate
        downside_returns = torch.min(returns, torch.zeros_like(returns))
        downside_std = torch.std(downside_returns)
        return torch.mean(excess_returns) / (downside_std + 1e-6)
    
    def _max_drawdown(self, returns: torch.Tensor) -> torch.Tensor:
        """Calculate maximum drawdown"""
        cum_returns = torch.cumsum(returns, dim=0)
        peak = torch.cummax(cum_returns, dim=0)[0]
        drawdown = (peak - cum_returns) / (peak + 1e-6)
        return torch.max(drawdown)
    
    def _calculate_risk_metrics(
        self,
        returns: torch.Tensor,
        risk_free_rate: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """Calculate various risk metrics"""
        metrics = {
            'sharpe_ratio': self.sharpe_ratio(returns, risk_free_rate),
            'sortino_ratio': self.sortino_ratio(returns, risk_free_rate),
            'max_drawdown': self._max_drawdown(returns),
            'return': torch.mean(returns),
            'volatility': torch.std(returns),
            'downside_risk': torch.std(torch.min(returns, torch.zeros_like(returns)))
        }
        return metrics
    
    def _calculate_risk_adjusted_return(
        self,
        returns: torch.Tensor,
        confidence: torch.Tensor,
        risk: torch.Tensor
    ) -> torch.Tensor:
        """Calculate risk-adjusted return"""
        # Calculate weighted return based on confidence
        weighted_return = returns * confidence
        
        # Calculate risk penalty
        risk_penalty = risk * self.config.max_drawdown_penalty
        
        # Combine return and risk
        risk_adjusted_return = weighted_return - risk_penalty
        
        return risk_adjusted_return
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        returns: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the total loss with risk adjustment
        
        Args:
            outputs: Dictionary with model outputs
                - direction: [batch_size, 3] (log probabilities for long/neutral/short)
                - magnitude: [batch_size] (0-1)
                - duration: [batch_size] (0-1)
                - confidence: [batch_size] (0-1)
                - risk: [batch_size] (0-1)
            targets: Dictionary with target values
                - direction: [batch_size] (0=long, 1=neutral, 2=short)
                - magnitude: [batch_size] (0-1)
                - duration: [batch_size] (0-1)
                - confidence: [batch_size] (0-1)
            returns: Optional tensor of realized returns [batch_size]
            
        Returns:
            Dictionary with individual loss components and total loss
        """
        # Direction classification loss
        direction_loss = self.direction_loss(
            outputs['direction'],
            targets['direction'].long()
        )
        
        # Magnitude regression loss with margin
        magnitude_diff = torch.abs(outputs['magnitude'] - targets['magnitude'])
        magnitude_loss = torch.where(
            magnitude_diff < self.config.margin,
            0.5 * magnitude_diff ** 2 / self.config.margin,
            magnitude_diff - 0.5 * self.config.margin
        ).mean()
        
        # Duration regression loss with margin
        duration_diff = torch.abs(outputs['duration'] - targets['duration'])
        duration_loss = torch.where(
            duration_diff < self.config.margin,
            0.5 * duration_diff ** 2 / self.config.margin,
            duration_diff - 0.5 * self.config.margin
        ).mean()
        
        # Confidence calibration loss
        confidence_loss = F.binary_cross_entropy(
            outputs['confidence'],
            targets['confidence'],
            reduction='mean'
        )
        
        # Risk adjustment
        if returns is not None and len(returns) > 0:
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(returns)
            
            # Calculate risk-adjusted return
            risk_adjusted_return = self._calculate_risk_adjusted_return(
                returns,
                outputs['confidence'].detach(),
                outputs['risk']
            )
            
            # Risk penalty based on drawdown
            risk_penalty = self.config.max_drawdown_penalty * risk_metrics['max_drawdown']
            
            # Combine risk metrics into a single score
            risk_score = (
                -self.config.sharpe_ratio_weight * risk_metrics['sharpe_ratio']
                -self.config.sortino_ratio_weight * risk_metrics['sortino_ratio']
                + risk_penalty
            )
        else:
            # If no returns provided, just use the model's risk prediction
            risk_score = outputs['risk'].mean()
        
        # Weighted sum of losses
        total_loss = (
            self.config.direction_weight * direction_loss +
            self.config.magnitude_weight * magnitude_loss +
            self.config.duration_weight * duration_loss +
            self.config.confidence_weight * confidence_loss +
            self.config.risk_weight * risk_score
        )
        
        # Prepare loss dictionary
        loss_dict = {
            'total': total_loss,
            'direction': direction_loss,
            'magnitude': magnitude_loss,
            'duration': duration_loss,
            'confidence': confidence_loss,
            'risk': risk_score,
        }
        
        # Add risk metrics if available
        if returns is not None and len(returns) > 0:
            loss_dict.update({
                'sharpe_ratio': risk_metrics['sharpe_ratio'],
                'sortino_ratio': risk_metrics['sortino_ratio'],
                'max_drawdown': risk_metrics['max_drawdown'],
                'return': risk_metrics['return'],
                'volatility': risk_metrics['volatility'],
                'downside_risk': risk_metrics['downside_risk'],
            })
        
        return loss_dict

# Example usage
if __name__ == "__main__":
    # Create loss config
    config = LossConfig(
        direction_weight=0.3,
        magnitude_weight=0.2,
        duration_weight=0.1,
        confidence_weight=0.2,
        risk_weight=0.2,
        max_drawdown_penalty=1.0,
        sharpe_ratio_weight=0.5,
        sortino_ratio_weight=0.5,
        margin=0.1,
        label_smoothing=0.1
    )
    
    # Create loss function
    loss_fn = RiskAdjustedLoss(config)
    
    # Create dummy data
    batch_size = 32
    
    # Model outputs
    outputs = {
        'direction': torch.randn(batch_size, 3),  # logits for 3 classes
        'magnitude': torch.sigmoid(torch.randn(batch_size)),
        'duration': torch.sigmoid(torch.randn(batch_size)),
        'confidence': torch.sigmoid(torch.randn(batch_size)),
        'risk': torch.sigmoid(torch.randn(batch_size)),
    }
    
    # Targets
    targets = {
        'direction': torch.randint(0, 3, (batch_size,)),
        'magnitude': torch.rand(batch_size),
        'duration': torch.rand(batch_size),
        'confidence': torch.rand(batch_size),
    }
    
    # Returns (optional)
    returns = torch.randn(batch_size) * 0.01  # Daily returns
    
    # Calculate loss
    loss_dict = loss_fn(outputs, targets, returns)
    
    # Print results
    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"{k}: {v.item():.4f}")
