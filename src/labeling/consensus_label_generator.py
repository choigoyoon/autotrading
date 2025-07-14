import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from dataclasses import dataclass
import json

@dataclass
class TimeframeWeights:
    """Timeframe weights for consensus calculation"""
    WEIGHTS = {
        '1w': 2.0, '1d': 1.8, '12h': 1.5, '8h': 1.3,  # Upper TFs
        '4h': 1.2, '2h': 1.0, '1h': 0.8,             # Middle TFs
        '30m': 0.6, '15m': 0.4, '5m': 0.2            # Lower TFs
    }

class ConsensusLabelGenerator:
    def __init__(self, data_dir: str):
        """
        Initialize the Consensus Label Generator
        
        Args:
            data_dir: Directory containing label data
        """
        self.data_dir = Path(data_dir)
        self.timeframes = list(TimeframeWeights.WEIGHTS.keys())
        self.weights = np.array(list(TimeframeWeights.WEIGHTS.values()))
        self.weights /= self.weights.sum()  # Normalize weights
        
    def load_labels(self, symbol: str = 'btc_usdt_kst') -> Dict[str, pd.DataFrame]:
        """Load labels for all timeframes"""
        labels = {}
        for tf in self.timeframes:
            # 수정: 파일 경로를 라벨 디렉토리 구조에 맞게 조정
            file_path = self.data_dir / symbol / f'macd_zone_{tf}.parquet'
            print(f"Checking {file_path}")
            if file_path.exists():
                print(f"Loading {tf} labels from {file_path}")
                labels[tf] = pd.read_parquet(file_path)
                print(f"Loaded {len(labels[tf])} rows for {tf}")
            else:
                print(f"Warning: {file_path} not found")
        return labels
    
    def calculate_consensus(self, labels: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate consensus scores from multiple timeframe labels"""
        if not labels:
            raise ValueError("No label data provided")
            
        # Align all dataframes on index
        aligned = pd.concat(
            [df['label'] for df in labels.values()], 
            axis=1, 
            keys=labels.keys(),
            join='outer'
        )
        
        # Calculate weighted consensus
        consensus = (aligned * self.weights[:len(aligned.columns)]).sum(axis=1)
        
        # Calculate signal strength (standard deviation across timeframes)
        signal_strength = aligned.std(axis=1)
        
        # Create final dataframe
        result = pd.DataFrame({
            'consensus_score': consensus,
            'signal_strength': signal_strength,
            'label': self._classify_labels(consensus, signal_strength)
        }, index=aligned.index)
        
        return result
    
    def _classify_labels(self, consensus: pd.Series, signal_strength: pd.Series) -> pd.Series:
        """Classify labels into S/A/B/C grades"""
        labels = pd.Series('C', index=consensus.index)
        
        # S-grade
        mask = (consensus > 8.0) & (signal_strength > 3.0)
        labels[mask] = 'S'
        
        # A-grade
        mask = (consensus > 6.0) & (signal_strength > 2.0) & (labels != 'S')
        labels[mask] = 'A'
        
        # B-grade
        mask = (consensus > 4.0) & (signal_strength > 1.0) & (labels.isin(['C']))
        labels[mask] = 'B'
        
        return labels
    
    def save_consensus_labels(self, consensus_df: pd.DataFrame, output_dir: str):
        """Save consensus labels to file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save labels
        labels_file = output_path / 'consensus_labels.parquet'
        consensus_df.to_parquet(labels_file)
        print(f"Saved consensus labels to {labels_file}")
        
        # Also save metadata
        metadata = {
            'timeframes': self.timeframes,
            'weights': {k: float(v) for k, v in zip(self.timeframes, self.weights)},
            'label_distribution': consensus_df['label'].value_counts().to_dict()
        }
        
        metadata_file = output_path / 'consensus_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_file}")

def generate_consensus_labels(data_dir: str, output_dir: str, symbol: str = 'btc_usdt_kst'):
    """Generate and save consensus labels"""
    print(f"Starting consensus label generation for {symbol}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    generator = ConsensusLabelGenerator(data_dir)
    labels = generator.load_labels(symbol)
    
    if not labels:
        print("No label data found")
        return
    
    print(f"\nTimeframes loaded: {list(labels.keys())}")
    
    try:
        consensus = generator.calculate_consensus(labels)
        print(f"\nConsensus calculation complete. Shape: {consensus.shape}")
        
        # Save results
        generator.save_consensus_labels(consensus, output_dir)
        return True
    except Exception as e:
        print(f"\n❌ Error generating consensus labels: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    # 기본 경로 설정 (프로젝트 루트 기준)
    project_root = Path(__file__).parent.parent.parent
    default_data_dir = str(project_root / 'data' / 'labels')
    default_output_dir = str(project_root / 'data' / 'processed' / 'consensus_labels')
    
    print(f"Project root: {project_root}")
    print(f"Default data directory: {default_data_dir}")
    print(f"Default output directory: {default_output_dir}")
    
    parser = argparse.ArgumentParser(description='Generate consensus trading labels from multiple timeframes')
    parser.add_argument('--data-dir', type=str, default=default_data_dir,
                      help=f'Directory containing label data (default: {default_data_dir})')
    parser.add_argument('--output-dir', type=str, default=default_output_dir,
                      help=f'Directory to save consensus labels (default: {default_output_dir})')
    parser.add_argument('--symbol', type=str, default='btc_usdt_kst',
                      help='Trading symbol (default: btc_usdt_kst)')
    
    args = parser.parse_args()
    print("\n" + "="*50)
    print(f"Running with settings:")
    print(f"- Data directory: {args.data_dir}")
    print(f"- Output directory: {args.output_dir}")
    print(f"- Symbol: {args.symbol}")
    print("="*50 + "\n")
    
    success = generate_consensus_labels(args.data_dir, args.output_dir, args.symbol)
    
    if success:
        print("\n✅ Consensus label generation completed successfully!")
    else:
        print("\n❌ Consensus label generation failed.")
        exit(1)
