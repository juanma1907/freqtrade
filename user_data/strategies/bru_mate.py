# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
import os
import json
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
from freqtrade.strategy import IStrategy, merge_informative_pair, stoploss_from_open, stoploss_from_absolute
import requests as requests
from freqtrade.persistence import Trade

class bru_mate(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '1m'

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".

    #response = requests.get('https://api.coindesk.com/v1/bpi/currentprice.json')
    #data = response.json()
    #a = 500/data["bpi"]["USD"]["rate_float"]    

    minimal_roi = {
        "0": 0.07
    }

    #b = 1000/data["bpi"]["USD"]["rate_float"]


    ## Optimal stoploss designed for the strategy.
    ## This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.07
    use_custom_stoploss = True

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 10

    # Strategy parameters
    buy_rsi = IntParameter(10, 40, default=29, space="buy")
    sell_rsi = IntParameter(60, 90, default=71, space="sell")
    print('------------------------------')
    print('------------------------------')
    print('------------------------------')
    print(buy_rsi)
    print(sell_rsi)
    print('------------------------------')
    print('------------------------------')
    print('------------------------------')

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }
    
    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            'main_plot': {
                'tema': {},
                'sar': {'color': 'white'},
            },
            'subplots': {
                # Subplots - each dict defines one additional plot
                "MACD": {
                    'macd': {'color': 'blue'},
                    'macdsignal': {'color': 'orange'},
                },
                "RSI": {
                    'rsi': {'color': 'red'},
                }
            }
        }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        
        return [("BTC/USDT", "15m")]
        """
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '15m') for pair in pairs]
        # Optionally Add additional "static" pairs
        informative_pairs += [
                              ("BTC/USDT:USDT", "15m"),
                              ("BTC/USDT:USDT", "1h"),
                              ("BTC/USDT:USDT", "4h")
                            ]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        
        inf_tf = '15m'
        inf_tf2 = '1h'
        inf_tf3 = '4h'
        # Get the informative pair
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        informative2 = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf2)
        informative3 = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf3)
        # Get the 20, 100 y 200 ema on 15m
        informative['ema20'] = ta.EMA(informative, timeperiod=20)
        informative['ema100'] = ta.EMA(informative, timeperiod=100)
        informative['ema200'] = ta.EMA(informative, timeperiod=200)
        # Get the 20, 100 y 200 ema on 1h
        informative2['ema20'] = ta.EMA(informative2, timeperiod=20)
        informative2['ema100'] = ta.EMA(informative2, timeperiod=100)
        informative2['ema200'] = ta.EMA(informative2, timeperiod=200)
        # Get the 20, 100 y 200 ema on 4h
        informative3['ema20'] = ta.EMA(informative3, timeperiod=20)
        informative3['ema100'] = ta.EMA(informative3, timeperiod=100)
        informative3['ema200'] = ta.EMA(informative3, timeperiod=200)

        # Use the helper function merge_informative_pair to safely merge the pair
        # Automatically renames the columns and merges a shorter timeframe dataframe and a longer timeframe informative pair
        # use ffill to have the 1d value available in every row throughout the day.
        # Without this, comparisons between columns of the original and the informative pair would only work once per day.
        # Full documentation of this method, see below
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)
        dataframe = merge_informative_pair(dataframe, informative2, self.timeframe, inf_tf2, ffill=True)
        dataframe = merge_informative_pair(dataframe, informative3, self.timeframe, inf_tf3, ffill=True)

        # Momentum Indicators
        # ------------------------------------

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

        # # Plus Directional Indicator / Movement
        # dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        # dataframe['plus_di'] = ta.PLUS_DI(dataframe)

        # # Minus Directional Indicator / Movement
        # dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        # dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        # # Aroon, Aroon Oscillator
        # aroon = ta.AROON(dataframe)
        # dataframe['aroonup'] = aroon['aroonup']
        # dataframe['aroondown'] = aroon['aroondown']
        # dataframe['aroonosc'] = ta.AROONOSC(dataframe)

        # # Awesome Oscillator
        # dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        # # Keltner Channel
        # keltner = qtpylib.keltner_channel(dataframe)
        # dataframe["kc_upperband"] = keltner["upper"]
        # dataframe["kc_lowerband"] = keltner["lower"]
        # dataframe["kc_middleband"] = keltner["mid"]
        # dataframe["kc_percent"] = (
        #     (dataframe["close"] - dataframe["kc_lowerband"]) /
        #     (dataframe["kc_upperband"] - dataframe["kc_lowerband"])
        # )
        # dataframe["kc_width"] = (
        #     (dataframe["kc_upperband"] - dataframe["kc_lowerband"]) / dataframe["kc_middleband"]
        # )

        # # Ultimate Oscillator
        # dataframe['uo'] = ta.ULTOSC(dataframe)

        # # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        # dataframe['cci'] = ta.CCI(dataframe)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        # rsi = 0.1 * (dataframe['rsi'] - 50)
        # dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # # Inverse Fisher transform on RSI normalized: values [0.0, 100.0] (https://goo.gl/2JGGoy)
        # dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

        # # Stochastic Slow
        # stoch = ta.STOCH(dataframe)
        # dataframe['slowd'] = stoch['slowd']
        # dataframe['slowk'] = stoch['slowk']

        # Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # # Stochastic RSI
        # Please read https://github.com/freqtrade/freqtrade/issues/2961 before using this.
        # STOCHRSI is NOT aligned with tradingview, which may result in non-expected results.
        # stoch_rsi = ta.STOCHRSI(dataframe)
        # dataframe['fastd_rsi'] = stoch_rsi['fastd']
        # dataframe['fastk_rsi'] = stoch_rsi['fastk']

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # # ROC
        # dataframe['roc'] = ta.ROC(dataframe)

        # Overlap Studies
        # ------------------------------------

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )

        # Bollinger Bands - Weighted (EMA based instead of SMA)
        # weighted_bollinger = qtpylib.weighted_bollinger_bands(
        #     qtpylib.typical_price(dataframe), window=20, stds=2
        # )
        # dataframe["wbb_upperband"] = weighted_bollinger["upper"]
        # dataframe["wbb_lowerband"] = weighted_bollinger["lower"]
        # dataframe["wbb_middleband"] = weighted_bollinger["mid"]
        # dataframe["wbb_percent"] = (
        #     (dataframe["close"] - dataframe["wbb_lowerband"]) /
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"])
        # )
        # dataframe["wbb_width"] = (
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"]) / dataframe["wbb_middleband"]
        # )

        # # EMA - Exponential Moving Average
        # dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        # dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        # dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        # dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        # dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        # dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # # SMA - Simple Moving Average
        # dataframe['sma3'] = ta.SMA(dataframe, timeperiod=3)
        # dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
        # dataframe['sma10'] = ta.SMA(dataframe, timeperiod=10)
        # dataframe['sma21'] = ta.SMA(dataframe, timeperiod=21)
        # dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        # dataframe['sma100'] = ta.SMA(dataframe, timeperiod=100)

        # Parabolic SAR
        dataframe['sar'] = ta.SAR(dataframe)

        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        # Cycle Indicator
        # ------------------------------------
        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']

        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------
        # # Hammer: values [0, 100]
        # dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
        # # Inverted Hammer: values [0, 100]
        # dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
        # # Dragonfly Doji: values [0, 100]
        # dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
        # # Piercing Line: values [0, 100]
        # dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]
        # # Morningstar: values [0, 100]
        # dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]
        # # Three White Soldiers: values [0, 100]
        # dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]

        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        # # Hanging Man: values [0, 100]
        # dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
        # # Shooting Star: values [0, 100]
        # dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
        # # Gravestone Doji: values [0, 100]
        # dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
        # # Dark Cloud Cover: values [0, 100]
        # dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
        # # Evening Doji Star: values [0, 100]
        # dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
        # # Evening Star: values [0, 100]
        # dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)

        # Pattern Recognition - Bullish/Bearish candlestick patterns
        # ------------------------------------
        # # Three Line Strike: values [0, -100, 100]
        # dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
        # # Spinning Top: values [0, -100, 100]
        # dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]
        # # Engulfing: values [0, -100, 100]
        # dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]
        # # Harami: values [0, -100, 100]
        # dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]
        # # Three Outside Up/Down: values [0, -100, 100]
        # dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]
        # # Three Inside Up/Down: values [0, -100, 100]
        # dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]

        # # Chart type
        # # ------------------------------------
        # # Heikin Ashi Strategy
        # heikinashi = qtpylib.heikinashi(dataframe)
        # dataframe['ha_open'] = heikinashi['open']
        # dataframe['ha_close'] = heikinashi['close']
        # dataframe['ha_high'] = heikinashi['high']
        # dataframe['ha_low'] = heikinashi['low']

        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """

        close = float(dataframe['close'].iloc[-1])
        low = float(dataframe['low'].iloc[-1])
        high = float(dataframe['high'].iloc[-1])
        #ema20_15m = float(dataframe['ema20_15m'].iloc[-1])
        #ema100_15m = float(dataframe['ema100_15m'].iloc[-1])
        #ema200_15m = float(dataframe['ema200_15m'].iloc[-1])
        #ema20_1h = float(dataframe['ema20_1h'].iloc[-1])
        #ema100_1h = float(dataframe['ema100_1h'].iloc[-1])
        #ema200_1h = float(dataframe['ema200_1h'].iloc[-1])
        #ema20_4h = float(dataframe['ema20_4h'].iloc[-1])
        #ema100_4h = float(dataframe['ema100_4h'].iloc[-1])
        #ema200_4h = float(dataframe['ema200_4h'].iloc[-1])
        rsi = float(dataframe['rsi'].iloc[-1])
        cruza_rsi_por_debajo = False
        cruza_rsi_por_encima = False
        if rsi < self.buy_rsi.value:
            cruza_rsi_por_debajo = True
        elif rsi > self.sell_rsi.value:
            cruza_rsi_por_encima = True

        # Algoritmo seleccionador de ema mas cercana dependiendo si hay sobrecompra o sobreventa 

        # emas_str = self.dp.current_blacklist() 

        emas_str = ["ema200_15m","ema20_1h","ema100_4h"]

        emas = list([])

        ema_mas_cercana = list([])

        for ema in emas_str:
            emas.append(float(dataframe[ema].iloc[-1]))
            ema_mas_cercana.append(abs(close-float(dataframe[ema].iloc[-1])))

        print(emas)


        #emas_str = list(['ema20 15m', 'ema100 15m', 'ema200 15m', 'ema20 1h', 'ema100 1h', 'ema200 1h', 'ema20 4h', 'ema100 4h', 'ema200 4h'])

        #emas = list([ema20_15m, ema100_15m, ema200_15m, ema20_1h, ema100_1h, ema200_1h, ema20_4h, ema100_4h, ema200_4h])
                
        #ema_mas_cercana = list([abs(close-ema20_15m), abs(close-ema100_15m), abs(close-ema200_15m), abs(close-ema20_1h), abs(close-ema100_1h), abs(close-ema200_1h), abs(close-ema20_4h), abs(close-ema100_4h), abs(close-ema200_4h)])

        ema_entrada_str = str(' ')
        
        ema_entrada = close

        if cruza_rsi_por_debajo or cruza_rsi_por_encima:
            while True:
                indice_ema_mas_cercana = int(ema_mas_cercana.index(min(ema_mas_cercana)))
                if cruza_rsi_por_debajo and low > emas[indice_ema_mas_cercana]:
                    ema_entrada_str = emas_str[indice_ema_mas_cercana]
                    ema_entrada = emas[indice_ema_mas_cercana]
                    break
                elif cruza_rsi_por_encima and high < emas[indice_ema_mas_cercana]:
                    ema_entrada_str = emas_str[indice_ema_mas_cercana]
                    ema_entrada = emas[indice_ema_mas_cercana]
                    break
                else:
                    emas.pop(indice_ema_mas_cercana)
                    ema_mas_cercana.pop(indice_ema_mas_cercana)
                    emas_str.pop(indice_ema_mas_cercana)
                    if len(emas) == 0:
                        break
        
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['rsi'], self.buy_rsi.value)) &  
                (low > ema_entrada)   
            ),
            ['enter_long', 'enter_tag']] = (1, ema_entrada_str)

        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi.value)) &  
                (high < ema_entrada)   
            ),
            ['enter_short', 'enter_tag']] = (1, ema_entrada_str)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi.value)) &  # Signal: RSI crosses above sell_rsi
                (dataframe['tema'] > dataframe['bb_middleband']) &  # Guard: tema above BB middle
                (dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard: tema is falling
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_long'] = 1
        # Uncomment to use shorts (Only used in futures/margin mode. Check the documentation for more info)
        """
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], s.iloc[-1]elf.buy_rsi.value)) &  # Signal: RSI crosses above buy_rsi
                (dataframe['tema'] <= dataframe['bb_middleband']) &  # Guard: tema below BB middle
                (dataframe['tema'] > dataframe['tema'].shift(1)) &  # Guard: tema is raising
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_short'] = 1
        """
        return dataframe
    
    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)
        #ema20_15m = dataframe['ema20_15m'].iloc[-1]
        #ema100_15m = dataframe['ema100_15m'].iloc[-1]
        #ema200_15m = dataframe['ema200_15m'].iloc[-1]
        #ema20_1h = dataframe['ema20_1h'].iloc[-1]
        #ema100_1h = dataframe['ema100_1h'].iloc[-1]
        #ema200_1h = dataframe['ema200_1h'].iloc[-1]
        #ema20_4h = dataframe['ema20_4h'].iloc[-1]
        #ema100_4h = dataframe['ema100_4h'].iloc[-1]
        #ema200_4h = dataframe['ema200_4h'].iloc[-1]

        ema_entrada = dataframe['enter_tag'].iloc[-1]

        #emas = list([ema20_15m, ema100_15m, ema200_15m, ema20_1h, ema100_1h, ema200_1h, ema20_4h, ema100_4h, ema200_4h])
                
        #emas_str = list(['ema20 15m', 'ema100 15m', 'ema200 15m', 'ema20 1h', 'ema100 1h', 'ema200 1h', 'ema20 4h', 'ema100 4h', 'ema200 4h'])

        #indice_ema_mas_cercana = int(emas_str.index(ema_entrada))

        new_entryprice = float(dataframe[ema_entrada].iloc[-1]) 
        print('-------------')
        print('-------------')
        print(ema_entrada, new_entryprice)
        print('-------------')
        print('-------------')
        #path_archivo_ema = os.path.join(os.getcwd(), "ema.py")
        #init = 'ema = ' + "'" + ema_entrada + "'"
        #with open(path_archivo_ema, 'w') as archivo_ema:
        #    archivo_ema.write(init)

        return new_entryprice

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 10.0

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # stop = 1000/trade.open_rate
        stoploss_price = trade.open_rate - 1000

        # return stoploss_from_open(-stop, current_profit, is_short=trade.is_short)
        return stoploss_from_absolute(stoploss_price, current_rate, is_short=trade.is_short)


    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):

        tp_signal_price_long = trade.open_rate + 10
        tp_signal_price_short = trade.open_rate - 10

        if not trade.is_short and current_rate >= tp_signal_price_long:

            return True

        if trade.is_short and current_rate <= tp_signal_price_short:

            return True

    def custom_exit_price(self, pair: str, trade: Trade,
                          current_time: datetime, proposed_rate: float,
                          current_profit: float, exit_tag: Optional[str], **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)

        tp_signal_price_long = trade.open_rate + 100
        tp_signal_price_short = trade.open_rate - 100

        print('-------------')
        print('-------------')
        print(trade.open_rate)
        print('-------------')
        print('-------------')

        close = float(dataframe['close'].iloc[-1])

        if not trade.is_short:
            new_exitprice = tp_signal_price_long
            print('-------------')
            print('-------------')
            print(new_exitprice)
            print('-------------')
            print('-------------')
            return new_exitprice

        if trade.is_short:
            new_exitprice = tp_signal_price_short
            print('-------------')
            print('-------------')
            print(new_exitprice)
            print('-------------')
            print('-------------')
            return new_exitprice
