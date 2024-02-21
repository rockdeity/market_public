
import datetime
import importlib
import unittest
import pandas as pd
import trader
from trader import DataHandler, Order, Portfolio, TradeSimulator, StrategySimulator, InsufficientCashException, InsufficientHoldingsException, InvalidGroupingDimensions

class TestDataHandlerFromDF(unittest.TestCase):
    def setUp(self):
        self.grouping_dimensions = ['symbol', 'model_type']
        self.forecast_dimensions = ['High', 'Low']
        self.date_col = 'date'
        self.forecast_leading_col = 'forecast_leading'
        self.forecast_target_col = 'forecast_target'
        self.actual_col_name = 'actual_column_name'
        self.forecast_leading_days_col = 'forecast_leading_days'
        self.forecast_leading_days = [1, 2, 3, 4]
        self.file_path = None
        self.df = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'GOOG', 'GOOG'],
            'model_type': ['type1', 'type2', 'type1', 'type2'],
            'forecast': [100, 200, 300, 400],
            'date': pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04']),
            'forecast_leading': [1, 2, 3, 4],
            'forecast_target': [100, 200, 300, 400],
            'High': [90, 210, 290, 410],
            'Low': [80, 125, 145, 365],
            'actual_column_name': ['High', 'High', 'Low', 'Low'],
            'forecast_leading_days': [1, 2, 3, 4]
        })
        # self.test_df = pd.DataFrame({
        #     'symbol': ['AAPL', 'AAPL'],
        #     'model_type': ['type1', 'type2'],
        #     'forecast': [100, 200],
        #     'date': pd.to_datetime(['2022-01-01', '2022-01-02']),
        #     'forecast_leading': [1, 2],
        #     'forecast_target': [100, 200],
        #     'High': [90, 210],
        #     'Low': [80, 125],
        #     'actual_column_name': ['High', 'High'],
        #     'forecast_leading_days': [1, 2]
        # })
        self.data_handler = DataHandler(
            self.grouping_dimensions, 
            self.forecast_dimensions, 
            self.date_col,
            self.forecast_leading_col, 
            self.forecast_target_col,
            self.actual_col_name,
            self.forecast_leading_days_col,
            self.forecast_leading_days,
            self.file_path, 
            df=self.df
        )

    def test_init_with_df(self):
        self.assertTrue(isinstance(self.data_handler.dataframe, pd.DataFrame))
        self.assertEqual(len(self.data_handler.dataframe), 4)
        
    def test_get_stock_data_list_args(self):
        
        with self.assertRaises(ValueError):
            self.data_handler.get_stock_date_indexed(symbol=['AAPL'], model_type=['type1'])        
            
    def test_get_stock_data_single_match_symbol(self):
        
        with self.assertRaises(ValueError):
            self.data_handler.get_stock_date_indexed(symbol='AAPL')

    def test_get_stock_data_single_match_model_type(self):
        
        with self.assertRaises(ValueError):
            self.data_handler.get_stock_date_indexed(model_type='type1')
            
    def test_pivot_forecast_dims(self):
        
        df = self.data_handler.pivot_forecast_dimensions()
        self.assertTrue(isinstance(df, pd.DataFrame))
        
    def test_get_stock_data_double_match(self):
        
        stock_data = self.data_handler.get_stock_date_indexed(symbol='AAPL', model_type='type1')
        self.assertTrue(isinstance(stock_data, pd.DataFrame))
        self.assertEqual(len(stock_data), 1)

class TestDataHandlerFromFile(unittest.TestCase):
    
    def setUp(self):
        self.file_path = 'test/results/forecast_actuals_all_models.pkl'
        self.grouping_dimensions = ['symbol', 'model_type']
        self.forecast_dimensions = ['High', 'Low']
        self.date_col = 'Date'
        self.forecast_leading_col = 'forecast_leading'
        self.forecast_target_col = 'forecast_target'
        self.actual_col_name = 'actual_column_name'
        self.forecast_leading_days_col = 'forecast_leading_days'
        self.forecast_leading_days = [1, 3, 5]
        self.data_handler = DataHandler(
            grouping_dimensions=self.grouping_dimensions,
            forecast_dimensions=self.forecast_dimensions,
            date_col=self.date_col,
            forecast_leading_col=self.forecast_leading_col,
            forecast_target_col=self.forecast_target_col,
            actual_col_name=self.actual_col_name,
            forecast_leading_days_col=self.forecast_leading_days_col,
            forecast_leading_days=self.forecast_leading_days,
            file_path=self.file_path
        )

    def test_init_with_file_path(self):
        self.assertTrue(isinstance(self.data_handler.dataframe, pd.DataFrame))
        self.assertEqual(len(self.data_handler.dataframe), 4788)
        
    def test_pivot_forecast_dims(self):
        
        df = self.data_handler.pivot_forecast_dimensions()
        self.assertTrue(isinstance(df, pd.DataFrame))


class TestPortfolio(unittest.TestCase):
    
    def setUp(self):

        self.start_date = datetime.datetime(2024,1,1)

    def test_buy_stock(self):
        portfolio = Portfolio(initial_cash=10000)
        portfolio.buy_stock(symbol='AMZN', quantity=10, price=150, trade_date=self.start_date + datetime.timedelta(days=1))
        self.assertEqual(portfolio.get_cash(), 10000 - (10 * 150))
        self.assertEqual(portfolio.get_stock_average_cost(),  (10 * 150))

    def test_sell_stock(self):
        portfolio = Portfolio(initial_cash=10000)
        portfolio.buy_stock(symbol='AMZN', quantity=10, price=150, trade_date=self.start_date)
        portfolio.sell_stock(symbol='AMZN', quantity=5, price=160, trade_date=self.start_date)
        self.assertEqual(portfolio.get_cash(), 10000 - (10 * 150) + (5 * 160))
        self.assertEqual(portfolio.get_stock_average_cost(), (10 * 150) - (5 * 150))
                         
    def test_buy_stock_insufficient_cash(self):
        portfolio = Portfolio(initial_cash=10000)
        with self.assertRaises(InsufficientCashException):
            portfolio.buy_stock(symbol='AMZN', quantity=100, price=150, trade_date=self.start_date)
        self.assertEqual(portfolio.get_cash(), 10000)
        self.assertEqual(portfolio.get_stock_average_cost(), 0)
        
    def test_sell_stock_insufficient_holdings(self):       
        portfolio = Portfolio(initial_cash=10000)
        with self.assertRaises(InsufficientHoldingsException):
            portfolio.sell_stock(symbol='AAPL', quantity=5, price=160, trade_date=self.start_date)
        self.assertEqual(portfolio.get_cash(), 10000)
        self.assertEqual(portfolio.get_stock_average_cost(), 0)

    def test_get_transaction_log(self):
        portfolio = Portfolio(initial_cash=10000)
        portfolio.buy_stock(symbol='AAPL', quantity=10, price=150, trade_date=self.start_date)
        portfolio.sell_stock(symbol='AAPL', quantity=5, price=160, trade_date=self.start_date)
        transaction_log = portfolio.get_transaction_log()
        self.assertEqual(len(transaction_log), 2)
        self.assertEqual(transaction_log.dataframe.iloc[0]['symbol'], 'AAPL')
        self.assertEqual(transaction_log.dataframe.iloc[0]['operation'], 'buy')
        self.assertEqual(transaction_log.dataframe.iloc[0]['quantity'], 10)
        self.assertEqual(transaction_log.dataframe.iloc[0]['price'], 150)
        self.assertEqual(transaction_log.dataframe.iloc[1]['symbol'], 'AAPL')
        self.assertEqual(transaction_log.dataframe.iloc[1]['operation'], 'sell')
        self.assertEqual(transaction_log.dataframe.iloc[1]['quantity'], 5)
        self.assertEqual(transaction_log.dataframe.iloc[1]['price'], 160)

class TestTradeSimulator(unittest.TestCase):
    
    def setUp(self):
        
        self.data_handler = DataHandler(
            grouping_dimensions=['symbol', 'model_type'],
            forecast_dimensions=['high', 'low', 'close', 'open'],
            date_col='Date',
            forecast_leading_col='forecast_leading',
            forecast_target_col='forecast_target',
            actual_col_name='actual_column_name',
            forecast_leading_days_col='forecast_leading_days',
            forecast_leading_days=[1, 3, 5],
            file_path='test/results/forecast_actuals_all_models.pkl'
            )
        
        self.start_date = datetime.datetime.strptime('2024-01-01', '%Y-%m-%d')
        self.default_model_type = 'lin_regression'
        self.portfolio = Portfolio(initial_cash=10000)
        self.trade_simulator = TradeSimulator(portfolio=self.portfolio, data_handler=self.data_handler)
        
    def test_init(self):
        self.assertEqual(self.trade_simulator.portfolio.get_cash(), 10000)
        data = self.trade_simulator.data_handler.get_stock_data_raw()
        data = data[data.index.get_level_values(self.trade_simulator.data_handler.date_col) == self.start_date]
        self.assertTrue(data.shape[0] > 0)

    def test_submit_trade(self):
        order = Order('buy', 'AAPL', 10, 150)
        self.trade_simulator.submit_order(order, self.start_date)  
        self.assertEqual(order, self.trade_simulator.get_orders()[0][0])
        self.assertEqual(self.start_date, self.trade_simulator.get_orders()[0][1])
        self.assertEqual(self.trade_simulator.portfolio.get_cash(), 10000)
        # self.trade_simulator.simulate_trades
        
    def test_simulate_trade(self):
        
        self.assertEqual(self.trade_simulator.portfolio.get_cash(), 10000)
        data = self.trade_simulator.data_handler.get_stock_date_indexed(symbol='AMZN', model_type=self.default_model_type)
        data = data[data.index.get_level_values(self.trade_simulator.data_handler.date_col) == self.start_date]
        price = data['close'].values[0]
        order = Order('AMZN', 'buy', 10, price)
        self.trade_simulator.submit_order(order, self.start_date)  
        # print(f"order: {order} self.trade_simulator.get_orders()[0]:{self.trade_simulator.get_orders()[0]}")
        expiration_days = 1
        self.trade_simulator.simulate_trades(self.start_date + datetime.timedelta(days=1), expiration_days, model_type=self.default_model_type)
        print(f"avg stock cost: {self.trade_simulator.portfolio.get_stock_average_cost()}, cash: {self.trade_simulator.portfolio.get_cash()}")
        print(f"transation_log: {self.trade_simulator.get_transaction_log()}")
        self.assertEqual(self.trade_simulator.portfolio.get_cash(), 10000 - (10 * price))
        self.assertEqual(self.trade_simulator.portfolio.get_stock_average_cost(), 10 * price)
        self.assertEqual(len(self.trade_simulator.get_transaction_log()), 1)
        self.assertEqual(len(self.trade_simulator.get_orders()), 0)
        
        self.start_date = datetime.datetime.strptime('2024-01-01', '%Y-%m-%d')
        self.default_model_type = 'lin_regression'
        self.portfolio = Portfolio(initial_cash=10000)
        self.trade_simulator = TradeSimulator(portfolio=self.portfolio, data_handler=self.data_handler)
        
    def test_simulate_no_symbol_data(self): #use two different symbols
        
        self.assertEqual(self.trade_simulator.portfolio.get_cash(), 10000)
        data = self.trade_simulator.data_handler.get_stock_date_indexed(symbol='AMZN', model_type=self.default_model_type)
        data = data[data.index.get_level_values(self.trade_simulator.data_handler.date_col) == self.start_date]
        price = data['close'].values[0]
        order = Order('AAPL', 'buy', 10, price)
        self.trade_simulator.submit_order(order, self.start_date) 
        with self.assertRaises(InvalidGroupingDimensions) as igde:
            expiration_days = 1
            self.trade_simulator.simulate_trades(self.start_date + datetime.timedelta(days=1), expiration_days, model_type=self.default_model_type)
        print(f"igde: {igde.exception}")
        
        self.start_date = datetime.datetime.strptime('2024-01-01', '%Y-%m-%d')
        self.default_model_type = 'lin_regression'
        self.portfolio = Portfolio(initial_cash=10000)
        self.trade_simulator = TradeSimulator(portfolio=self.portfolio, data_handler=self.data_handler)
        
    def test_simulate_trade_bad_symbol(self): #use two different symbols
        # self.assertEqual(self.trade_simulator.portfolio.get_total_current_value(), 10000)
        with self.assertRaises(InvalidGroupingDimensions):
            self.trade_simulator.data_handler.get_stock_date_indexed(symbol='boogetyboogety', model_type=self.default_model_type)
        
    def test_simulate_trade_bad_model_type(self): #use two different symbols
        self.assertEqual(self.trade_simulator.portfolio.get_cash(), 10000)
        with self.assertRaises(InvalidGroupingDimensions):
            self.trade_simulator.data_handler.get_stock_date_indexed(symbol='AMZN', model_type='boogetyboogety')
    
    def test_execute_trade_not_in_daily_range(self):
        self.trade_simulator.execute_trade(Order(symbol='AMZN', operation='buy', quantity=10, price=150), self.start_date, model_type=self.default_model_type)
        self.assertEqual(self.portfolio.get_cash(), 10000)
        self.assertEqual(self.portfolio.get_stock_average_cost(), 0)
        self.assertEqual(len(self.trade_simulator.get_transaction_log()), 0)
        
    def test_execute_trade_in_daily_range(self):
        self.trade_simulator.execute_trade(Order(symbol='AMZN', operation='buy', quantity=10, price=152), self.start_date, model_type=self.default_model_type)
        self.assertEqual(self.portfolio.get_cash(), 10000 - (10 * 152))
        self.assertEqual(self.portfolio.get_stock_average_cost(), 10*152)
        self.assertEqual(len(self.trade_simulator.get_transaction_log()), 1)
        
    def test_execute_trade_insufficient_cash(self):
        # with self.assertRaises(InsufficientCashException):
        #     self.trade_simulator.execute_trade(Order(symbol='AMZN', operation='buy', quantity=100, price=152), self.start_date, model_type=self.default_model_type)
        self.assertEqual(self.portfolio.get_cash(), 10000)
        self.assertEqual(self.portfolio.get_stock_average_cost(), 0)
        self.assertEqual(len(self.trade_simulator.get_transaction_log()), 0)

    def test_execute_trade_insufficient_holdings(self):
        # with self.assertRaises(InsufficientHoldingsException):
        #     self.trade_simulator.execute_trade(Order(symbol='AMZN', operation='sell', quantity=10, price=152), self.start_date, model_type=self.default_model_type)
        self.trade_simulator.execute_trade(Order(symbol='AMZN', operation='sell', quantity=10, price=152), self.start_date, model_type=self.default_model_type)
        self.assertEqual(self.portfolio.get_cash(), 10000)
        self.assertEqual(self.portfolio.get_stock_average_cost(), 0)
        self.assertEqual(len(self.trade_simulator.get_transaction_log()), 0)

    def test_execute_trade(self):
        self.trade_simulator.execute_trade(Order(symbol='AMZN', operation='buy', quantity=10, price=152), self.start_date, model_type=self.default_model_type)
        self.trade_simulator.execute_trade(Order(symbol='AMZN', operation='sell', quantity=5, price=152.2), self.start_date, model_type=self.default_model_type)
        self.assertEqual(self.portfolio.get_cash(), 10000 - 10*152 + 5*152.2)
        self.assertEqual(self.portfolio.get_stock_average_cost(), (10*152) - (5*152))
        
    def test_get_orders(self):
        self.trade_simulator.submit_order(Order(symbol='AMZN', operation='buy', quantity=10, price=152), self.start_date)
        orders = self.trade_simulator.get_orders()
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0][0].symbol, 'AMZN')
        self.assertEqual(orders[0][0].operation, 'buy')
        self.assertEqual(orders[0][0].quantity, 10)
        self.assertEqual(orders[0][0].price, 152)
        self.assertEqual(orders[0][1], self.start_date)

    def test_get_transaction_log(self):
        self.trade_simulator.execute_trade(Order(symbol='AMZN', operation='buy', quantity=10, price=152), self.start_date, model_type=self.default_model_type)
        transaction_log = self.trade_simulator.get_transaction_log()
        self.assertEqual(len(transaction_log), 1)
        self.assertEqual(transaction_log.dataframe.iloc[0]['symbol'], 'AMZN')
        self.assertEqual(transaction_log.dataframe.iloc[0]['quantity'], 10)
        self.assertEqual(transaction_log.dataframe.iloc[0]['price'], 152)

class TestStrategySimulator(unittest.TestCase):
    
    def setUp(self):
        self.file_path = 'test/results/forecast_actuals_all_models.pkl'
        self.dimensions = ['symbol', 'model_type']
        self.data_handler = DataHandler(grouping_dimensions=self.dimensions, file_path=self.file_path)
        self.portfolio = Portfolio(initial_cash=10000)
        self.trade_simulator = TradeSimulator(portfolio=self.portfolio)
        self.strategy_simulator = StrategySimulator(data_handler=self.data_handler, trade_simulator=self.trade_simulator)

    def test_advanced_strategy(self):
        current_date = '2022-01-01'
        symbol = 'AAPL'
        low_target = 140
        high_target = 160
        leading_days = 5
        stock_data = self.data_handler.get_stock_data(['symbol'])
        filtered_data = stock_data[(stock_data['symbol'] == symbol) & (stock_data['date'] <= current_date)]
        strategy_data = self.strategy_simulator.advanced_strategy(current_date, symbol, low_target, high_target, leading_days)
        self.assertTrue(isinstance(strategy_data, pd.DataFrame))
        self.assertEqual(len(strategy_data), len(filtered_data))

    def test_simulate_strategy(self):
        initial_cash = 10000
        low_target = 140
        high_target = 160
        leading_days = 5
        quantity_per_trade = 1
        price_per_trade = 0
        self.strategy_simulator.simulate_strategy(initial_cash, low_target, high_target, leading_days, quantity_per_trade, price_per_trade)
        transaction_log = self.trade_simulator.get_transaction_log()
        self.assertTrue(isinstance(transaction_log, pd.DataFrame))
        self.assertEqual(len(transaction_log), 0)

class TestTradeSimulatorSimulateTrading(unittest.TestCase):
    
    def setUp(self):
        
        self.dataframe = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'GOOG', 'GOOG'],
            'model_type': ['type1', 'type1', 'type1', 'type1'],
            'forecast': [100, 200, 300, 400],
            'date': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-02', '2021-01-04']),
            'forecast_leading': [1, 2, 3, 4],
            'forecast_target': [100, 200, 300, 400],
            'High': [90, 210, 290, 410],
            'Low': [80, 125, 145, 365],
            'actual_column_name': ['high', 'high', 'low', 'low'],
            'forecast_leading_days': [1, 2, 3, 4]
        })
        self.data_handler = DataHandler(
            grouping_dimensions=['symbol', 'model_type'],
            forecast_dimensions=['high', 'low'],
            date_col='date',
            forecast_leading_col='forecast_leading',
            forecast_target_col='forecast_target',
            actual_col_name='actual_column_name',
            forecast_leading_days_col='forecast_leading_days',
            forecast_leading_days=[1, 3, 5],
            file_path=None, 
            df=self.dataframe,
        )
        
        self.start_date = datetime.datetime.strptime('2024-01-01', '%Y-%m-%d')
        
        self.portfolio = Portfolio(initial_cash=10000)
        self.trade_simulator = TradeSimulator(portfolio=self.portfolio, data_handler=self.data_handler)
        self.trade_date = datetime.datetime.strptime('2021-01-02', '%Y-%m-%d')
        self.trade_simulator.orders = [
            (Order(operation='buy', symbol='AAPL', quantity=10, price=150), datetime.datetime(2021, 1, 1)),
            (Order(operation='sell', symbol='AAPL', quantity=5, price=160), datetime.datetime(2021, 12, 1)),
            (Order(operation='buy', symbol='GOOG', quantity=8, price=200), datetime.datetime(2020, 1, 1))
        ]

    # def test_simulate_trading_execute_trade(self):
    #     expiration_days = 1
    #     self.trade_simulator.simulate_trades(self.trade_date, expiration_days, model_type='type1')
    #     transaction_log = self.trade_simulator.get_transaction_log()
    #     self.assertEqual(len(transaction_log), 1)
    #     self.assertEqual(transaction_log[0][0].symbol, 'AAPL')
    #     self.assertEqual(transaction_log[0][0].operation, 'buy')
    #     self.assertEqual(transaction_log[0][0].quantity, 10)
    #     self.assertEqual(transaction_log[0][0].price, 150)

class TestStrategySimulator(unittest.TestCase):
    
    def setUp(self):
        self.dataframe = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'GOOG', 'GOOG'],
            'model_type': ['type1', 'type1', 'type1', 'type1'],
            'forecast': [100, 200, 300, 400],
            'date': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04']),
            'forecast_leading': [1, 2, 3, 4],
            'forecast_target': [100, 200, 300, 400],
            'High': [90, 210, 290, 410],
            'Low': [80, 125, 145, 365],
            'Close': [85, 150, 180, 400],
            'actual_column_name': ['high', 'high', 'low', 'low'],
            'forecast_leading_days': [1, 2, 3, 4]
        })
        self.data_handler = DataHandler(
            grouping_dimensions=['symbol', 'model_type'],
            forecast_dimensions=['high', 'low', 'close'],
            date_col='date',
            forecast_leading_col='forecast_leading',
            forecast_target_col='forecast_target',
            actual_col_name='actual_column_name',
            forecast_leading_days_col='forecast_leading_days',
            forecast_leading_days=[1, 3, 5],
            file_path=None, 
            df=self.dataframe,
        )
        self.start_date = datetime.datetime(2021, 1, 1)
        self.default_model_type = 'lin_regression'
        self.portfolio = Portfolio(initial_cash=10000)
        self.trade_simulator = TradeSimulator(portfolio=self.portfolio, data_handler=self.data_handler)
        self.strategy_simulator = StrategySimulator(
            data_handler=self.data_handler,
            trade_simulator=self.trade_simulator, 
            model_type='type1',
            strategy=self.six_way_spread_weighted_outer)
        
    def six_way_spread_weighted_outer(self, symbol, symbol_data):
         
        # logger.info(f"clear_up_or_down: symbol: {symbol} symbol_data: {symbol_data}")
        orders = []
        dollars_to_spend = 1
        symbol_data = symbol_data.iloc[0]
        half_fan = 3
        # sell halfway between the two highs     
        for i in range (1, half_fan):
            price = (symbol_data['forecast_leading_High_1'] + symbol_data['High']) * (0.1 + (0.8 * i / half_fan))
            quantity = (dollars_to_spend + (i/3)) / (price + 0.0001)
            orders.append(Order(symbol=symbol, operation='sell', quantity=quantity, price=price))
        # buy halfway between the two lows
        for i in range (1, half_fan):
            price = (symbol_data['forecast_leading_Low_1'] + symbol_data['Low']) * (0.1 - (0.8 * i / half_fan))
            quantity = (dollars_to_spend + (i/3)) / (price + 0.0001)
            orders.append(Order(symbol=symbol, operation='buy', quantity=quantity, price=price))
        return orders

    def test_simulate_strategy_for_date_buy_no_day_data(self):
        date = datetime.datetime(2024, 1, 2)
        unique_symbols = ['AAPL']
        data_by_symbol = {
            'AAPL': pd.DataFrame({
                'symbol': ['AAPL'],
                'model_type': ['type1'],
                'date': [date], # DATE OFF BY ONE
                'forecast_leading_high_1': [200],
                'forecast_leading_low_1': [150],
                'high': [180],
                'low': [120],
                'close': [160]
            }).set_index(['symbol','model_type','date'])
        }
        

        # with(self.assertRaises(IndexError)):
        self.strategy_simulator.simulate_strategy_for_date(unique_symbols, data_by_symbol, date)
        self.strategy_simulator.simulate_trades(date) # DATE OFF BY ONE
        self.assertEqual(self.portfolio.get_cash(), 10000)
    
    def test_simulate_strategy_for_date_buy(self):
        date = datetime.datetime(2021, 1, 1)
        unique_symbols = ['AAPL']
        data_by_symbol = {
            'AAPL': 
                pd.DataFrame({
                'symbol': ['AAPL','AAPL'],
                'model_type': ['type1','type1'],
                'date': [date, date + datetime.timedelta(days=1)],
                'forecast_leading_high_1': [200, 200],
                'forecast_leading_low_1': [150, 150],
                'High': [180, 180],
                'Low': [120, 120],
                'Close': [160, 160]
            }).set_index(['symbol','model_type','date'])
        }  
        
        self.strategy_simulator.simulate_strategy_for_date(unique_symbols, data_by_symbol, date)
        self.strategy_simulator.simulate_trades(date + datetime.timedelta(days=1))
        self.assertEqual(self.portfolio.get_cash(), 10000)

class TestStragegyClassSimulator(unittest.TestCase):

    def setUp(self):

        self.symbol = 'ZION'
        self.start_date = datetime.datetime(2023, 10, 19)
        self.file_path = 'test/results/forecast_actuals_all_models_class.pkl'
        self.model_type = 'log_regression'
        self.grouping_dimensions = ['symbol', 'model_type']
        self.forecast_dimensions = ['LogClose_5_rolling_mean', 'close', 'low', 'high']
        self.date_col = 'Date'
        self.forecast_leading_col = 'forecast_leading'
        self.forecast_target_col = 'forecast_target'
        self.actual_col_name = 'actual_column_name'
        self.class_actual_col = 'class_actual'
        self.class_threshold_col = 'classify_threshold'
        self.forecast_leading_days_col = 'forecast_leading_days'
        self.forecast_leading_days = [1, 3, 5, 10]
        self.forecast_class_thresholds = [0.6]

        importlib.reload(trader)

        data_handler = trader.DataHandler(
            grouping_dimensions=self.grouping_dimensions,
            forecast_dimensions=self.forecast_dimensions,
            date_col=self.date_col,
            forecast_leading_col=self.forecast_leading_col,
            forecast_target_col=self.forecast_target_col,
            actual_col_name=self.actual_col_name,
            forecast_leading_days_col=self.forecast_leading_days_col,
            forecast_leading_days=self.forecast_leading_days,
            class_actual_col=self.class_actual_col,
            class_threshold_col=self.class_threshold_col,
            class_threshold_cols=self.forecast_class_thresholds,
            file_path=self.file_path
        )
        self.data_handler = data_handler
        self.portfolio = Portfolio(initial_cash=10000)
        self.trade_simulator = TradeSimulator(data_handler=data_handler, portfolio=self.portfolio)
        
        def avg_close_up_or_down(symbol, symbol_data):
            # logger.info(f"avg_close_up_or_down: symbol: {symbol} symbol_data: {symbol_data}")
            symbol_data = symbol_data.iloc[0]
            if symbol_data['forecast_leading_Logclose_5_rolling_mean_10_0.6'] > 0:
                return [Order(symbol=symbol, operation='buy', quantity=1, price=symbol_data['close'])]
            elif symbol_data['forecast_leading_Logclose_5_rolling_mean_10_0.6'] < 0:
                return [Order(symbol=symbol, operation='sell', quantity=1, price=symbol_data['close'])]
            return []
        
        self.strategy_simulator = StrategySimulator(model_type=self.model_type, data_handler=self.data_handler, 
                                                    trade_simulator=self.trade_simulator, strategy=avg_close_up_or_down)
    

    def test_simulate_strategy_for_date(self):
        
        symbol = self.symbol
           
        data_by_symbol = {}
        data_by_symbol[symbol] = self.strategy_simulator.data_handler.get_stock_date_indexed(symbol = symbol, model_type = self.model_type)
    
        self.strategy_simulator.simulate_strategy_for_date(unique_symbols=[symbol], data_by_symbol=data_by_symbol, date=self.start_date)
        self.strategy_simulator.simulate_trades(self.start_date, expiration_duration=10)
        transaction_log = self.trade_simulator.get_transaction_log()
        # self.assertTrue(isinstance(transaction_log, pd.DataFrame))
        # self.assertEqual(len(transaction_log), 0)

if __name__ == '__main__':
    unittest.main()