# 표준 라이브러리
import sys
import warnings
from pathlib import Path
from threading import Lock

# 서드파티 라이브러리
import pandas as pd
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# --- 프로젝트 경로 설정 (로컬 모듈 임포트 직전) ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- 로컬 모듈 임포트 (수정됨) ---
from scripts.live_ai_predictions import LiveConfig, AIPredictor
from scripts.portfolio_optimizer import get_rule_based_exposure
from scripts.virtual_trading_simulator import TraderConfig, VirtualTrader

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Flask 앱 및 SocketIO 설정 ---
app = Flask(__name__, template_folder='../templates')
socketio = SocketIO(app, async_mode='threading')

# --- 백그라운드 스레드 관리 ---
thread = None
thread_lock = Lock()

def dashboard_background_thread() -> None:
    """백그라운드에서 가상 거래를 시뮬레이션하고 데이터를 전송합니다."""
    print("🚀 대시보드 백그라운드 스레드 시작")
    
    # 설정 및 초기화 (수정됨)
    live_config = LiveConfig()
    trader_config = TraderConfig()
    predictor = AIPredictor(live_config)
    trader = VirtualTrader(initial_balance=10000.0, config=trader_config)
    
    balance_history: list[dict[str, object]] = []

    while True:
        try:
            # 1. 데이터 가져오기 (수정됨)
            current_sequence_df = predictor.get_latest_sequence_from_exchange()
            if current_sequence_df is None or current_sequence_df.empty:
                socketio.sleep(live_config.REFRESH_INTERVAL_SECONDS)
                continue

            # 2. 피처 및 신호 생성 (수정됨)
            features_df = predictor.create_features(current_sequence_df.copy())
            if features_df.empty: continue
            
            ai_signal_str, ai_confidence = predictor.predict(features_df)
            ai_signal = 1 if ai_signal_str == "BUY" else -1
            
            macd_hist_last = features_df['macdhist'].iloc[-1]
            macd_signal = 1 if macd_hist_last > 0 else -1

            # 3. 포지션 결정 (기존 로직 유지)
            signal_data = pd.DataFrame([{
                'ai_signal': ai_signal,
                'macd_signal': macd_signal,
                'ai_confidence': ai_confidence
            }])
            exposure = get_rule_based_exposure(signal_data).iloc[0]

            # 4. 가상 거래 실행 및 관리 (기존 로직 유지)
            current_price = current_sequence_df['close'].iloc[-1]
            current_time = current_sequence_df.index[-1]
            trader.manage_positions(current_price, current_time)
            trader.execute_trade(exposure, current_price, current_time)
            
            # 5. 데이터 준비 및 전송 (수정됨)
            portfolio_report = trader.generate_report()
            balance_history.append({'time': pd.to_datetime(current_time), 'balance': portfolio_report['Balance']})
            
            # 프론트엔드로 보낼 데이터 패키징
            data_to_send = {
                'signals': format_signal_data(ai_signal, ai_confidence, macd_signal),
                'portfolio': {
                    'balance': f"{portfolio_report['Balance']:,.2f}",
                    'return_pct': f"{portfolio_report['Total_Return_pct']:.2f}",
                    'trades': portfolio_report['Trades'],
                    'win_rate': f"{portfolio_report['Win_Rate_pct']:.2f}"
                },
                'chart': {
                    'x': [item['time'].strftime('%Y-%m-%d %H:%M:%S') for item in balance_history[-100:] if isinstance(item['time'], pd.Timestamp)],
                    'y': [item['balance'] for item in balance_history[-100:]]
                }
            }
            socketio.emit('update', data_to_send)

            socketio.sleep(live_config.REFRESH_INTERVAL_SECONDS)
        
        except Exception as e:
            print(f"백그라운드 스레드 오류: {e}")
            socketio.sleep(10)

def format_signal_data(ai_signal: int, confidence: float, macd_signal: int) -> dict[str, str]:
    """프론트엔드 표시용 신호 데이터 포맷팅"""
    ai_text = f"BUY ({confidence:.1%})" if ai_signal == 1 else f"SELL ({confidence:.1%})"
    macd_text = "BUY" if macd_signal == 1 else "SELL"
    
    hybrid_text: str
    hybrid_class: str
    
    if ai_signal == macd_signal:
        action_text = "BUY" if ai_signal == 1 else "SELL"
        if confidence > 0.85: 
            hybrid_text = f"강력한 {action_text} 신호!"
            hybrid_class = "strong-buy" if ai_signal == 1 else "strong-sell"
        else:
            hybrid_text = f"일치된 {action_text} 신호"
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
def index() -> str:
    """대시보드 페이지를 렌더링합니다."""
    return render_template('dashboard.html')

@socketio.on('connect')
def on_connect() -> None:
    """클라이언트 연결 시 백그라운드 스레드 시작."""
    global thread
    with thread_lock:
        if thread is None:
            print('클라이언트 연결됨: 백그라운드 스레드 시작')
            thread = socketio.start_background_task(dashboard_background_thread)
        else:
            print('클라이언트 재연결됨: 스레드는 이미 실행 중')
    emit('update', {'status': 'Connected'})


if __name__ == '__main__':
    print("대시보드 서버 시작: http://127.0.0.1:5000")
    socketio.run(app, debug=False, allow_unsafe_werkzeug=True) 