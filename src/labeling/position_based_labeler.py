# type: ignore
# pylint: disable-all
from enum import Enum

import pandas as pd

import numpy as np

from typing import Tuple, List, Optional



class TradingState(Enum):

    WAITING = 0      # ?ÄÍ∏?(?¨Ï????ÜÏùå)

    LONG_ENTRY = 1   # Î°?ÏßÑÏûÖ ?†Ìò∏

    LONG_HOLD = 2    # Î°?Î≥¥Ïú† Ï§?

    LONG_EXIT = 3    # Î°?Ï≤?Ç∞ ?†Ìò∏

    SHORT_ENTRY = -1 # ??ÏßÑÏûÖ ?†Ìò∏

    SHORT_HOLD = -2  # ??Î≥¥Ïú† Ï§?

    SHORT_EXIT = -3  # ??Ï≤?Ç∞ ?†Ìò∏



class Position:

    def __init__(self):

        self.state: TradingState = TradingState.WAITING

        self.entry_price: Optional[float] = None

        self.entry_time: Optional[str] = None

        self.stop_loss: Optional[float] = None

        self.take_profit: Optional[float] = None

        self.size: float = 0.0



class PositionBasedLabeler:

    def __init__(self, stop_loss_pct: float = 0.03, lookback_period: int = 5):

        self.stop_loss_pct = stop_loss_pct

        self.lookback_period = lookback_period

    

    def is_bottom_reversal(self, df: pd.DataFrame, idx: int) -> bool:

        """ÏßÑÏßú Î∞îÎã• Î∞òÏ†Ñ?∏Ï? ?ïÏù∏"""

        # Íµ¨ÌòÑ ?¥Ïö©

        return False

    

    def is_top_reversal(self, df: pd.DataFrame, idx: int) -> bool:

        """ÏßÑÏßú Ï≤úÏû• Î∞òÏ†Ñ?∏Ï? ?ïÏù∏"""

        # Íµ¨ÌòÑ ?¥Ïö©

        return False

    

    def generate_position_labels(self, df: pd.DataFrame) -> Tuple[List[int], List[int], List[float]]:

        """?¨Ï???Í∏∞Î∞ò ?ºÎ≤® ?ùÏÑ± (?§Ï†ú Îß§Îß§ ?úÎ??àÏù¥??"""

        close = df['close'].values

        macd_hist = df['macd_histogram'].values if 'macd_histogram' in df.columns else None

        if macd_hist is None:

            raise ValueError('macd_histogram Ïª¨Îüº???ÑÏöî?©Îãà??')

        n = len(df)

        actions = [0] * n

        positions = [0] * n

        entry_prices = [np.nan] * n

        state = TradingState.WAITING

        entry_price: Optional[float] = None

        for i in range(1, n):

            # MACD Î≥ÄÍ≥°Ï†ê Í∞êÏ?

            prev_hist = macd_hist[i-1]

            curr_hist = macd_hist[i]

            action = 0

            # ÏßÑÏûÖ Ï°∞Í±¥: Î∞îÎã• Î∞òÏ†Ñ(Î°?, Ï≤úÏû• Î∞òÏ†Ñ(??

            if state == TradingState.WAITING:

                if not (np.isnan(prev_hist) or np.isnan(curr_hist)):

                    if prev_hist < 0 and curr_hist >= 0 and self.is_bottom_reversal(df, i):

                        # Î°?ÏßÑÏûÖ

                        state = TradingState.LONG_HOLD

                        entry_price = float(close[i])

                        action = 1

                    elif prev_hist > 0 and curr_hist <= 0 and self.is_top_reversal(df, i):

                        # ??ÏßÑÏûÖ

                        state = TradingState.SHORT_HOLD

                        entry_price = float(close[i])

                        action = -1

            elif state == TradingState.LONG_HOLD and entry_price is not None:

                # Î°??¨Ï???Î≥¥Ïú† Ï§? ?êÏ†à/?µÏ†à/Î∞òÎ? ?†Ìò∏

                stop_loss = entry_price * (1 - self.stop_loss_pct)

                take_profit = entry_price * (1 + self.stop_loss_pct)

                if close[i] <= stop_loss:

                    # ?êÏ†à

                    state = TradingState.WAITING

                    action = 2

                    entry_price = None

                elif close[i] >= take_profit:

                    # ?µÏ†à

                    state = TradingState.WAITING

                    action = -2

                    entry_price = None

                elif prev_hist > 0 and curr_hist <= 0 and self.is_top_reversal(df, i):

                    # Î∞òÎ? ?†Ìò∏(Ï≤?Ç∞)

                    state = TradingState.WAITING

                    action = -1

                    entry_price = None

            elif state == TradingState.SHORT_HOLD and entry_price is not None:

                # ???¨Ï???Î≥¥Ïú† Ï§? ?êÏ†à/?µÏ†à/Î∞òÎ? ?†Ìò∏

                stop_loss = entry_price * (1 + self.stop_loss_pct)

                take_profit = entry_price * (1 - self.stop_loss_pct)

                if close[i] >= stop_loss:

                    # ?êÏ†à

                    state = TradingState.WAITING

                    action = 2

                    entry_price = None

                elif close[i] <= take_profit:

                    # ?µÏ†à

                    state = TradingState.WAITING

                    action = -2

                    entry_price = None

                elif prev_hist < 0 and curr_hist >= 0 and self.is_bottom_reversal(df, i):

                    # Î∞òÎ? ?†Ìò∏(Ï≤?Ç∞)

                    state = TradingState.WAITING

                    action = 1

                    entry_price = None

            # Í∏∞Î°ù

            actions[i] = action

            if state == TradingState.LONG_HOLD and entry_price is not None:

                positions[i] = 1

                entry_prices[i] = entry_price

            elif state == TradingState.SHORT_HOLD and entry_price is not None:

                positions[i] = -1

                entry_prices[i] = entry_price

            else:

                positions[i] = 0

                entry_prices[i] = np.nan

        return actions, positions, entry_prices 

