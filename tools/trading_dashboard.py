from flask import Flask, render_template
from flask_socketio import SocketIO
import pandas as pd
from pathlib import Path
import sys
import warnings

# --- 프로젝트 설정 ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# 가상 트레이더 및 관련 모듈 임포트
from tools.live_ai_predictions import LiveConfig, load_latest_model_and_scaler, get_current_data, get_features_from_data, predict_with_confidence
from tools.portfolio_optimizer import get_rule_based_exposure
from tools.virtual_trading_simulator import VirtualTrader, TraderConfig

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Flask 앱 및 SocketIO 설정 ---
app = Flask(__name__, template_folder='../templates')
socketio = SocketIO(app, async_mode='threading')

# --- 데이터 시뮬레이션 및 거래 로직 ---
def dashboard_background_thread():
    """백그라운드에서 가상 거래를 시뮬레이션하고 데이터를 전송합니다."""
    print("🚀 대시보드 백그라운드 스레드 시작")
    
    # 설정 및 초기화
    config = LiveConfig()
    trader_config = TraderConfig()
    model, scaler, device = load_latest_model_and_scaler(config)
    trader = VirtualTrader(initial_balance=10000, config=trader_config)
    
    iteration = 0
    balance_history = []

    while True:
        try:
            # 1. 데이터 가져오기 (시뮬레이션)
            current_sequence_df, iteration = get_current_data(config, iteration)
            if current_sequence_df.empty:
                socketio.sleep(1)
                continue

            # 2. 피처 및 신호 생성
            features_df = get_features_from_data(current_sequence_df.copy())
            ai_signal, ai_confidence = predict_with_confidence(model, scaler, features_df, device)
            macd_signal = 1 if features_df['macd_hist'].iloc[-1] > 0 else -1

            # 3. 포지션 결정 (규칙 기반)
            signal_data = pd.DataFrame([{
                'ai_signal': ai_signal,
                'macd_signal': macd_signal,
                'ai_confidence': ai_confidence
            }])
            exposure = get_rule_based_exposure(signal_data).iloc[0]

            # 4. 가상 거래 실행 및 관리
            current_price = current_sequence_df['close'].iloc[-1]
            current_time = current_sequence_df.index[-1]
            trader.manage_positions(current_price, current_time)
            trader.execute_trade(exposure, current_price, current_time)
            
            # 5. 데이터 준비 및 전송
            portfolio_report = trader.generate_report()
            balance_history.append({'time': current_time, 'balance': portfolio_report['Balance']})
            
            # 프론트엔드로 보낼 데이터 패키징
            data_to_send = {
                'signals': format_signal_data(ai_signal, ai_confidence, macd_signal),
                'portfolio': {
                    'balance': portfolio_report['Balance'],
                    'return_pct': portfolio_report['Total Return (%)'],
                    'trades': portfolio_report['Trades'],
                    'win_rate': portfolio_report['Win Rate (%)']
                },
                'chart': {
                    'x': [item['time'].strftime('%Y-%m-%d %H:%M:%S') for item in balance_history],
                    'y': [item['balance'] for item in balance_history]
                }
            }
            socketio.emit('update', data_to_send)

            iteration += 1
            socketio.sleep(config.REFRESH_INTERVAL_SECONDS) # 1분 대기
        
        except Exception as e:
            print(f"백그라운드 스레드 오류: {e}")
            socketio.sleep(10)

def format_signal_data(ai_signal, confidence, macd_signal):
    """프론트엔드 표시용 신호 데이터 포맷팅"""
    ai_text = f"BUY ({confidence:.1%})" if ai_signal == 1 else f"SELL ({confidence:.1%})"
    macd_text = "BUY" if macd_signal == 1 else "SELL"
    
    hybrid_text = "신호 분석 중..."
    hybrid_class = ""
    
    if ai_signal == macd_signal:
        if confidence > 0.85: 
            hybrid_text = f"강력한 {ai_text.split(' ')[0]} 신호!"
            hybrid_class = "strong-buy" if ai_signal == 1 else "strong-sell"
        else:
            hybrid_text = f"일치된 {ai_text.split(' ')[0]} 신호"
            hybrid_class = "buy" if ai_signal == 1 else "sell"
    else:
        hybrid_text = "신호 충돌, 관망 권장"
        hybrid_class = "conflict"

    return {
        'ai_text': ai_text,
        'ai_signal_class': 'buy' if ai_signal == 1 else 'sell',
        'macd_text': macd_text,
        'macd_signal_class': 'buy' if macd_signal == 1 else 'sell',
        'hybrid_text': hybrid_text,
        'hybrid_class': hybrid_class,
    }


@app.route('/')
def index():
    """대시보드 페이지를 렌더링합니다."""
    return render_template('dashboard.html')

@socketio.on('connect')
def on_connect():
    """클라이언트 연결 시 백그라운드 스레드 시작."""
    print('클라이언트 연결됨')
    global thread
    thread = socketio.start_background_task(dashboard_background_thread)


if __name__ == '__main__':
    print("대시보드 서버 시작: http://127.0.0.1:5000")
    socketio.run(app, debug=False) 