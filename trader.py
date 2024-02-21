
import datetime
import itertools
import logging
import numpy as np
import os
import pandas as pd
import pickle
from typing import Callable, List
import timing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not os.path.exists('logs'):
    os.makedirs('logs')
filename = f"logs/{__name__}_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log"
if not os.path.exists(filename):
    with open(filename, 'wt') as f:
        f.write(f'*** {filename} ***\n')
        
if len(logger.handlers) < 1:
    handler = logging.FileHandler(filename)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

slow_asserts = False

class InsufficientCashException(Exception):
    """Raised when there is not enough cash to buy"""
    pass

class InsufficientHoldingsException(Exception):
    """Raised when there are not enough shares to sell"""
    pass

class InvalidGroupingDimensions(Exception):
    """Raised when the grouping dimensions input do not match expected"""
    pass

class UnrecognizedTradeAction(Exception):
    """Raised when the desired trade action is unrecognized"""
    pass

class DataHandler:
    
    def __init__ (
        self, 
        grouping_dimensions, 
        forecast_dimensions, 
        date_col,
        forecast_leading_col, 
        forecast_target_col,
        actual_col_name,
        forecast_leading_days_col,
        class_actual_col=None,
        attribute_column_names=None,
        attribute_columns=None,
        # class_threshold_col=None,
        # class_threshold_cols=None,
        tickers_to_highlight=None,
        file_path=None, 
        df=None):
        
        if df is not None:
            self.dataframe = df.copy(deep=True)
        else:
            with open(file_path, 'rb') as file:
                self.dataframe = pickle.load(file)
            
        assert isinstance(self.dataframe, pd.DataFrame)
        for col in grouping_dimensions:
            assert col in self.dataframe.columns or col in self.dataframe.index.names, f"grouping_dimensions: {grouping_dimensions} not in dataframe columns {self.dataframe.columns}: or index: {self.dataframe.index.names}"
        assert 'symbol' in grouping_dimensions
        
        self.date_col = date_col
        self.grouping_dimensions = grouping_dimensions
        self.forecast_dimensions = forecast_dimensions
        self.forecast_leading_col = forecast_leading_col
        self.forecast_target_col = forecast_target_col
        self.actual_col_name = actual_col_name
        self.forecast_leading_days_col = forecast_leading_days_col
        self.class_actual_col = class_actual_col
        self.attribute_column_names = attribute_column_names
        self.attribute_columns = attribute_columns
        # self.class_threshold_col = class_threshold_col
        # self.class_threshold_cols = class_threshold_cols
        
        assert date_col in self.dataframe.columns or date_col in self.dataframe.index.names
        assert forecast_leading_col in self.dataframe.columns
        assert forecast_target_col in self.dataframe.columns
        assert actual_col_name in self.dataframe.columns
        assert forecast_leading_days_col in self.dataframe.columns
        if self.class_actual_col is not None:
            assert self.class_actual_col in self.dataframe.columns
            assert isinstance(self.actual_col_name, str)
        if self.attribute_column_names is not None:
            assert type(self.attribute_column_names) == list
            # for col in self.attribute_column_names:
            #     assert(col in self.dataframe.columns, f"{col} not in dataframe columns: {self.dataframe.columns}")
            # assert pd.api.types.is_numeric_dtype(self.dataframe[self.class_threshold_col]) or isinstance(self.class_threshold_col, str)
        
        assert self.dataframe.index.get_level_values(date_col).dtype == 'datetime64[ns]'
        assert pd.api.types.is_numeric_dtype(self.dataframe[forecast_leading_col])
        assert pd.api.types.is_numeric_dtype(self.dataframe[forecast_target_col])
        assert pd.api.types.is_numeric_dtype(self.dataframe[forecast_leading_days_col])
        assert isinstance(actual_col_name, str)
        
        assert set(forecast_dimensions).issubset(set(self.dataframe.columns))
        for col in self.forecast_dimensions:
            assert pd.api.types.is_numeric_dtype(self.dataframe[col])

        self.dataframe = self.dataframe.reset_index().drop('index', axis=1) # yuck. I hate indices. How can I guarantee that there is no index column?
        self.dataframe = self.dataframe.set_index(grouping_dimensions + [date_col])
        self.dataframe = self.dataframe.sort_index()

        # if tickers_to_highlight is not None:
        #     self.dataframe = self.dataframe[self.dataframe.index.get_level_values('symbol').isin(tickers_to_highlight)]
        
        # verify that we have one and only one row with actual_col_name to each member of actual_cols for each multi-index (dimensions + date_col)
        mask = True
        # if forecast_leading_days is not None:
        #     mask = mask & self.dataframe[self.forecast_leading_days_col].isin(self.forecast_leading_days)
        # mask = mask & (self.dataframe[self.actual_col_name].isin(self.forecast_dimensions))
        # if attribute_columns is None:
            # combos = itertools.product(self.forecast_dimensions, self.forecast_leading_days)
            # self.dataframe = self.dataframe[mask]
            # assert self.dataframe.groupby(level=self.grouping_dimensions + [self.date_col]).size().max() <= len(list(combos))
            # if slow_asserts:
            #     for forecast_dim, leading_days in combos:
            #         mask = (self.dataframe[self.forecast_leading_days_col] == leading_days)
            #         mask = mask & (self.dataframe[self.actual_col_name] == forecast_dim)
            #         df_permutation = self.dataframe[mask]
            #         assert df_permutation.groupby(level=self.grouping_dimensions + [self.date_col])[self.actual_col_name].count().eq(1).all()
        if attribute_columns is not None:
            assert type(attribute_columns) == list
            # combos = itertools.product(self.forecast_dimensions, self.forecast_leading_days, self.attribute_columns)
            for col in self.attribute_columns:
                mask = mask & self.dataframe[col].isin(self.attribute_columns)
            self.dataframe = self.dataframe[mask]
            largest_group = self.dataframe.groupby(level=self.grouping_dimensions + [self.date_col])[self.actual_col_name].count().max()
            # most_combos = len(list(combos))
            # logger.info(f"largest_group: {largest_group} most_combos: {most_combos}")
            # assert largest_group <= most_combos
            # if slow_asserts:
            #     for forecast_dim, leading_days, class_threshold in combos:
            #         # logger.debug(f"threshold checking: forecast_dim:{forecast_dim} leading_days:{leading_days} class_threshold:{class_threshold}")
            #         mask = (self.dataframe[self.forecast_leading_days_col] == leading_days)
            #         mask = mask & (self.dataframe[self.actual_col_name] == forecast_dim)
            #         mask = mask & (self.dataframe[self.class_threshold_col] == class_threshold)
            #         df_permutation = self.dataframe[mask]
            #         assert df_permutation.groupby(level=self.grouping_dimensions + [self.date_col])[self.actual_col_name].count().eq(1).all()
            
        # self.pivoted_forecast_dimensions_df = self.dataframe.groupby(self.grouping_dimensions).apply(call_wrapper)
        if self.attribute_column_names is not None:
            self.pivoted_forecast_dimensions_df = self.pivot_forecast_dimensions(attribute_column_names=[forecast_leading_days_col] + self.attribute_column_names)
        else:
            self.pivoted_forecast_dimensions_df = self.pivot_forecast_dimensions(attribute_column_names=[forecast_leading_days_col])

        # filename = 'pivoted_forecast_dimensions_df.csv'
        # logger.info(f"saving to file:{filename}")
        # self.pivoted_forecast_dimensions_df.to_csv(filename)

    def pivot_forecast_dimensions(self, 
                                  pivot_columns=['forecast_target', 'forecast_leading'], 
                                  attribute_column_names=['forecast_leading_days']):
        return self._pivot_forecast_dimensions(self.dataframe, pivot_columns, attribute_column_names)
    
    def _pivot_forecast_dimensions(self, df, pivot_columns, attribute_column_names):
        
        df = df.copy(deep=True).reset_index()
        # Creating a unique identifier for each combination
        df['combined_name'] = df['actual_column_name']
        for col in attribute_column_names:
            df['combined_name'] = df['combined_name'] + '_' + df[col].astype(str) + "_" + col
        # pivoted_dfs = []
        needed_columns = self.grouping_dimensions + self.forecast_dimensions + [self.date_col] 
        df_merged = df[needed_columns].drop_duplicates() # we will have sparseness in the pivoted columns, 
        # so get rid of those and deduplicate
        for pivot_col in pivot_columns:
            pivoted_df = df.pivot_table(index=self.grouping_dimensions + [self.date_col], 
                                    columns='combined_name', 
                                    values=pivot_col,
                                    aggfunc='sum')
            pivoted_df.columns = [f"{pivot_col}_{col}" for col in pivoted_df.columns]
            # pivoted_dfs.append(pivoted_df)
            df_merged = df_merged.merge(pivoted_df, on=self.grouping_dimensions + [self.date_col])
            needed_columns = needed_columns + list(pivoted_df.columns) 

        columns_to_drop = df_merged.columns.difference(needed_columns)
        # [self.forecast_target_col, self.forecast_leading_col, self.actual_col_name, 
        #                    self.forecast_leading_col, 'combined_name', 'index', 'label_date', 'label_date_reverse']
        df_merged = df_merged.drop(columns=columns_to_drop, errors='ignore').set_index(self.grouping_dimensions + [self.date_col])
        return df_merged

    def get_stock_data_raw(self):
        return self.dataframe.copy(deep=True)
    
    def get_stock_data_normalized(self):
        return self.pivoted_forecast_dimensions_df #.copy(deep=True)

    # def get_stock_date_indexed(self, **kwargs):

    #     with pd.option_context(
    #         # 'display.max_rows', None, 
    #         'display.max_columns', None,
    #         'display.max_colwidth', None):
    #         # verify that kwargs keys are equal to the contents of grouping_dimensions
    #         if set(kwargs.keys()) != set(self.grouping_dimensions):
    #             raise ValueError(f"get_stock_data_date_indexed: kwargs keys {kwargs.keys()} must match grouping_dimensions {self.grouping_dimensions}")

    #         mask = []
    #         for key, value in kwargs.items():
    #             # logger.debug(f"get_stock_data_date_indexed key:{key}, value:{value}")
    #             # verify that value is a simple type, not a list or other data structure
    #             if isinstance(value, list):
    #                 raise ValueError(f"get_stock_data_date_indexed: value {value} must be a simple type, not a list")
    #             mask.append((self.pivoted_forecast_dimensions_df.index.get_level_values(key) == value)) # e.g. self.dataframe.index.get_level_values('symbol') == 'AAPL'
    #         mask = np.all(mask, axis=0)
    #         # logger.debug(self.pivoted_forecast_dimensions_df)
    #         df = self.pivoted_forecast_dimensions_df[mask] #.copy(deep=True).reset_index().set_index([self.date_col])
    #         # logger.debug(df)
    #         if df.empty:
    #             raise InvalidGroupingDimensions(f"get_stock_data_date_indexed: no data for intersection {kwargs}")
    #         if slow_asserts and df.reset_index().groupby(self.grouping_dimensions + [self.date_col])[self.date_col].count().max() > 1:
    #             multi_count_df = df.reset_index().groupby(self.grouping_dimensions + [self.date_col])[self.date_col].count()
    #             multi_count_df = multi_count_df[multi_count_df > 1]
    #             max_count = multi_count_df.max() 
                
    #             logger.error(f"get_stock_data_date_indexed: invalid pivot (e.g. multiple rows per symbol-date) self.grouping_dimensions{self.grouping_dimensions} max_count:{max_count} on multi_count_df: {multi_count_df}")
    #             raise InvalidGroupingDimensions(f"get_stock_data_date_indexed: invalid pivot (e.g. multiple rows per symbol-date) self.grouping_dimensions{self.grouping_dimensions} max_count:{max_count} on multi_count_df: {multi_count_df}")
    #         return df
    
    def _join_to_other_dataframe(self, other_datahandler, df, other_df):
        assert self.grouping_dimensions == other_datahandler.grouping_dimensions
        assert self.date_col == other_datahandler.date_col
        join_columns = self.grouping_dimensions + [self.date_col]
        new_columns = df.columns.difference(other_df.columns)
        df_joined = df.join(other_df[new_columns], on=join_columns, how='outer')
        return df_joined
    
    def join_to_other_datahandler(self, other_datahandler):
        
        self.dataframe = self._join_to_other_dataframe(other_datahandler, self.dataframe, other_datahandler.dataframe)
        self.pivoted_forecast_dimensions_df = self._join_to_other_dataframe(other_datahandler, self.pivoted_forecast_dimensions_df, other_datahandler.pivoted_forecast_dimensions_df)
        self.forecast_dimensions = list(set(self.forecast_dimensions).union(set(other_datahandler.forecast_dimensions)))
        self.attribute_column_names = list(set(self.attribute_column_names).union(set(other_datahandler.attribute_column_names)))
        self.forecast_dimensions = list(set(self.forecast_dimensions).union(set(other_datahandler.forecast_dimensions)))
        
    
class Order:
    def __init__(self, symbol: str, operation: str, quantity, price: float):
        self.symbol = symbol
        self.operation = operation
        self.quantity = quantity
        self.price = price
        
    def to_string(self):
        return f"Order(symbol={self.symbol}, operation={self.operation}, quantity={self.quantity}, price={self.price})"
    
class Trade():
    def __init__(self,  order: Order = None, execution_date: datetime.datetime = None, *args, **kwargs):
        if order is not None:
            self.symbol = order.symbol
            self.operation = order.operation
            self.quantity = order.quantity
            self.price = order.price
        else:
            self.symbol = kwargs['symbol'] if 'symbol' in kwargs else None
            self.operation = kwargs['operation'] if 'operation' in kwargs else None
            self.quantity = kwargs['quantity'] if 'quantity' in kwargs else None
            self.price = kwargs['price'] if 'price' in kwargs else None

        if execution_date is not None:
            if not isinstance(execution_date, datetime.datetime):
                self.execution_date = str(execution_date)
            else:
                self.execution_date = execution_date.strftime('%Y-%m-%d %H:%M:%S')
        else:
            self.execution_date = None
        
    def to_dataframe(self):
        return pd.DataFrame([self.__dict__], index=[0])
    
class TransactionLog:
    def __init__(self):
        init = Trade()
        self.dataframe = pd.DataFrame(init.__dict__, index=init.__dict__.keys()).drop(init.__dict__.keys(), axis=0)
        # logger.info(f"TransactionLog: init: {init.__dict__} df: {self.dataframe}")
        # self.dataframe = self.dataframe

    def append(self, trade: Trade):
        self.dataframe = pd.concat([self.dataframe, trade.to_dataframe()], ignore_index=True)
        # self.df = self.df.append(trade.__dict__, ignore_index=True)

    def __iter__(self):
        for _, row in self.dataframe.iterrows():
            yield Trade(**row.to_dict())

    def __getitem__(self, index):
        dct = self.dataframe.loc[index].to_dict()
        order = Order(symbol=dct['symbol'], operation=dct['operation'], quantity=dct['quantity'], price=dct['price'])
        trade = Trade(order=order, execution_date=dct['execution_date'])
        return trade

    def __len__(self):
        return len(self.dataframe)
    
class Portfolio:
    def __init__(self, initial_cash):
        self._cash = initial_cash
        self.holdings = {}  # Format: {'symbol': (quantity, average_price)}
        self.transaction_log = TransactionLog()
        
    def get_cash(self):
        return self._cash
    
    def buy_stock(self, symbol=None, quantity=None, price=None, trade_date=None):
        assert symbol is not None
        assert quantity is not None
        assert price is not None
        assert trade_date is not None
        if price * quantity > self._cash:
            raise InsufficientCashException("Insufficient cash to buy")
        self._cash -= price * quantity
        if symbol in self.holdings:
            # TODO: unit-test
            prior_quantity = self.holdings[symbol][0]
            prior_price = self.holdings[symbol][1]
            prior_total_cost = prior_quantity * prior_price # quantity * price
            new_quantity = self.holdings[symbol][0] + quantity
            added_cost = price * quantity
            new_cost = prior_total_cost + added_cost
            self.holdings[symbol] = (new_quantity, price)
            self.average_price = new_cost / new_quantity
        else:
            self.holdings[symbol] = (quantity, price)
            self.average_price = price
        self.transaction_log.append(Trade(order=Order(symbol, 'buy', quantity, price), execution_date=trade_date))
        
    def sell_stock(self, symbol=None, quantity=None, price=None, trade_date=None):
        assert symbol is not None
        assert quantity is not None
        assert price is not None
        assert trade_date is not None
        symbol_holding = self.holdings.get(symbol)
        if symbol_holding is None:
            raise InsufficientHoldingsException(f"Insufficient shares to sell: {symbol} desires sell quantity: {quantity} holdings: {symbol_holding}")
        quantity = min(quantity, symbol_holding[0])
        self._cash += price * quantity
        quantify_diff = quantity - self.holdings[symbol][0]
        if quantify_diff >= 0:
            if quantify_diff > 0:
                logger.warning(f"Portfolio.sell_stock: selling more shares than owned of {symbol} by difference of {quantify_diff} quantity: {quantity} holdings: {self.holdings[symbol][0]}")
            logger.debug(f"Portfolio.sell_stock: selling all shares of {symbol}")
            del self.holdings[symbol]
        else:
            total_quantity = self.holdings[symbol][0] - quantity
            self.holdings[symbol] = (total_quantity, self.holdings[symbol][1])
        self.transaction_log.append(Trade(order=Order(symbol, 'sell', quantity, price), execution_date=trade_date))
        
    def execute_trade(self, order: Order, trade_date: datetime.datetime):
        if order.operation == 'buy':
            self.buy_stock(order.symbol, order.quantity, order.price, trade_date)
        elif order.operation == 'sell':
            self.sell_stock(order.symbol, order.quantity, order.price, trade_date)
        else:
            raise UnrecognizedTradeAction(f"Portfolio.execute_trade: order {order} not executed because operation {order.operation} is not buy or sell")
        
    def get_stock_average_cost(self):
        total_value = 0
        for symbol, (quantity, price) in self.holdings.items():
            total_value += quantity * price  # Assuming current price = average price
        return total_value
    
    def get_holdings(self):
        return self.holdings
    
    def get_transaction_log(self):
        return self.transaction_log

class TradeSimulator:
    
    def __init__(self, portfolio: Portfolio, data_handler: DataHandler):
        self.portfolio = portfolio
        self.data_handler = data_handler
        self.orders = []
        self.submitted = 0
        self.executed = 0
        self.errors = 0
        self.expired = 0
        self.num_buys = 0
        self.num_sells = 0
        self.insufficient_cash = 0
        self.insufficient_holdings = 0
        self.no_data_for_symbol = 0
        self.no_data_for_symbol_date = 0
        self.date_in_future = 0
        self.same_day_submission = 0

    def set_initial_cash(self, initial_cash):
        self.portfolio.cash = initial_cash
        
    def submit_order(self, order: Order, submission_date: datetime.datetime):
        assert(submission_date is not None)
        self.orders.append((order, submission_date))
        self.submitted += 1
        
    def simulate_trades(self, trade_date: datetime.datetime, expiration_duration: int, model_type=None,):
        
        if trade_date.weekday() in [4,5]: # don't submit on Friday and Saturday
            logger.info(f"TradeSimulator.simulate_trades: not simulating trades on {trade_date} because it is a weekend")
            return None
            
        df = self.data_handler.get_stock_data_normalized()
        df = df[df.index.get_level_values(self.data_handler.date_col) == trade_date]
        orders = self.orders.copy()
        to_remove = []
        for order, submission_date in orders:
            # logger.info(f"processing order: {order.to_string()} submission_date: {submission_date} trade_date: {trade_date}")
            # check if trade_date is a weekend or NYSE holiday and skip if so (no trading on weekends or holidays)
            if submission_date + datetime.timedelta(days=expiration_duration) < trade_date:
                logger.debug(f"TradeSimulator.simulate_trades: expiring order {order} at {trade_date} submitted on {submission_date} because it is more than {expiration_duration} days old")
                to_remove.append((order, submission_date))
                self.expired += 1
            elif submission_date == trade_date:
                logger.debug("Same-day submission and trade date. Not Executing trade, but keeping order.")
                self.same_day_submission += 1
            elif submission_date > trade_date:
                logger.error(f"TradeSimulator.simulate_trades: not executing order {order} at {trade_date} submitted on {submission_date} because it is in the future")
                self.date_in_future += 1
            else:
                try:
                    df = self.execute_trade(df, order, trade_date, model_type=model_type)
                    if df is not None and not df.empty:
                        to_remove.append((order, submission_date))
                        self.executed += 1
                    else:
                        logger.error(f"TradeSimulator.simulate_trades: error executing trade: {order} at {trade_date} submitted on {submission_date}")
                        self.errors += 1
                        to_remove.append((order, submission_date))
                    # else:
                    #     raise ValueError(f"TradeSimulator.simulate_trades: error executing trade: {order} at {trade_date} submitted on {submission_date}")  
                except InsufficientCashException as ice: 
                    logger.debug(f"insufficient Cash. order: {order.to_string()} submission_date: {submission_date}") 
                    self.insufficient_cash += 1
                    to_remove.append((order, submission_date))
                    pass
                except InsufficientHoldingsException as ihe:   
                    logger.debug(f"insufficient Holdings. order: {order.to_string()} submission_date: {submission_date}") 
                    self.insufficient_holdings += 1
                    to_remove.append((order, submission_date))
                    pass
                # except Exception as e:
                #     logger.error(f"TradeSimulator.simulate_trades: error executing trade: {e}")
                #     raise e  
                
        for remove in to_remove:
            self.orders.remove(remove)
                     
        return df

    def execute_trade(self, df: pd.DataFrame, order: Order, trade_date: datetime.datetime, model_type = None):

        if model_type is None:
            trade_data_lin = df
        else:
            trade_data_lin = df
        if trade_data_lin.empty:
            logger.error(f"TradeSimulator.execute_trade: no data index for symbol {order.symbol} on {trade_date}")
            self.no_data_for_symbol += 1
            return None
        
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None,'display.max_colwidth', None):
        # logger.info(f"TradeSimulator.execute_trade: trade_data_lin BEFORE: {trade_data_lin}")
        # trade_data_lin_filtered = trade_data_lin[trade_data_lin.index.get_level_values(self.data_handler.date_col) == trade_date]
        trade_data_lin_filtered = df
        # logger.info(f"TradeSimulator.execute_trade: trade_data_lin AFTER: {trade_data_lin}")
        # return trade_data_lin
        if trade_data_lin_filtered.empty:
            logger.error(f"TradeSimulator.execute_trade: no data at date for symbol {order.symbol} on {trade_date}")
            self.no_data_for_symbol_date += 1
            return None
        price_low = trade_data_lin_filtered['low'].values[0]
        # logger.info(f"TradeSimulator.execute_trade: price_low: {price_low}")
        price_high = trade_data_lin_filtered['high'].values[0]
        
        if order.operation == 'buy':
            if order.price >= price_low:
                try:
                    self.portfolio.execute_trade(order, trade_date)
                    self.num_buys += 1
                except ValueError as e:
                    logger.error(f"TradeSimulator.execute_trade: error executing trade: {e}, likely due to invalid pivot (e.g. multiple rows per symbol-date) on trade_data_lin: {trade_data_lin}")
                    raise e
            else:
                logger.debug(f"TradeSimulator.execute_trade: order {order} not executed because price {order.price} is less than low price {price_low} on {trade_date}")
        elif order.operation == 'sell':
            if order.symbol in self.portfolio.holdings:
                if order.price <= price_high:
                    self.portfolio.sell_stock(symbol=order.symbol, quantity=order.quantity, price=order.price, trade_date=trade_date)
                    self.num_sells += 1
                else:
                    logger.debug(f"TradeSimulator.execute_trade: order {order} not executed because price {order.price} is greater than high price {price_high} on {trade_date}")
            else:
                raise InsufficientHoldingsException(f"TradeSimulator.execute_trade: order {order} not executed because no holdings on {trade_date}")
                # logger.debug(f"TradeSimulator.execute_trade: order {order} not executed because no holdings on {trade_date}")
        elif order.operation == 'hold':
            logger.debug(f"TradeSimulator.execute_trade: order {order} not executed because operation is hold")
        else:
            raise UnrecognizedTradeAction(f"TradeSimulator.execute_trade: order {order} not executed because operation {order.operation} is not buy or sell")
        return trade_data_lin

    def get_transaction_log(self):
        return self.portfolio.get_transaction_log()
    
    def get_orders(self):
        return self.orders
    
    def get_current_holdings(self):
        return self.portfolio.get_holdings()
    
    @timing.timer_func
    def get_current_price_value(self, current_date, print_holdings=False, in_logger: logging.Logger=logger):
        
        value = self.portfolio.get_cash()
        last_non_nan_date = self.data_handler.pivoted_forecast_dimensions_df[self.data_handler.pivoted_forecast_dimensions_df['adjClose'].notna()]
        last_non_nan_date = last_non_nan_date[last_non_nan_date.index.get_level_values(self.data_handler.date_col) <= current_date]
        last_non_nan_date = last_non_nan_date[last_non_nan_date.index.get_level_values(self.data_handler.date_col) == 
                                              last_non_nan_date.index.get_level_values(self.data_handler.date_col).max()]
        for symbol, symbol_history in last_non_nan_date.groupby(by='symbol'):
            holdings = self.portfolio.get_holdings()
            if symbol in holdings:
                # symbol_history = symbol_history[symbol_history.index.get_level_values(self.data_handler.date_col) == current_date]
                if symbol_history.empty:
                    in_logger.error(f"TradeSimulator.get_current_price_value: no data for symbol {symbol} on {current_date}, though it's in holdings")
                    continue
                symbol_row = symbol_history.iloc[0]
                this_value = symbol_row['adjClose'] * holdings[symbol][0]
                this_cost = holdings[symbol][0] * holdings[symbol][1]
                # format to two decimal places for readability
                this_value = round(this_value, 2)
                if print_holdings:
                    in_logger.info(f"TradeSimulator.get_current_price_value: symbol: {symbol} delta: {round(this_value - this_cost, 2)} value: {round(this_value, 2)}" +
                                f" cost: {round(this_cost, 2)} holdings: {round(holdings[symbol][0], 2)} price: {round(symbol_row['adjClose'], 2)}")
                value += this_value
        return value
    
    def print_current_holdings(self):
        holdings = self.portfolio.get_holdings()
        for symbol, (quantity, price) in holdings.items():
            logger.info(f"symbol: {symbol} quantity: {quantity} price: {price}")
            
    def print_transaction_stats(self):
        logger.info(f"submitted: {self.submitted} executed: {self.executed} errors: {self.errors} expired: {self.expired} num_buys: {self.num_buys} num_sells: {self.num_sells} " +
              f"insufficient_cash: {self.insufficient_cash} insufficient_holdings: {self.insufficient_holdings} no_data_for_symbol: {self.no_data_for_symbol} " +
              f"no_data_for_symbol_date: {self.no_data_for_symbol_date} date_in_future: {self.date_in_future} same_day_submission: {self.same_day_submission}")
    
    
class StrategySimulator:

    def __init__(self, data_handler: DataHandler, trade_simulator: TradeSimulator, model_type: str, strategy: Callable[[str, pd.DataFrame], List[Order]]):
        self.data_handler = data_handler
        self.trade_simulator = trade_simulator
        self.model_type = model_type
        self.max_value = 0
        self.min_value = 99999999999999999
        self.strategy = strategy
    
    def six_way_spread_weighted_outer(symbol, symbol_data):
         
        # logger.info(f"clear_up_or_down: symbol: {symbol} symbol_data: {symbol_data}")
        orders = []
        dollars_to_spend = 1
        symbol_data = symbol_data.iloc[0]
        half_fan = 3
        # sell halfway between the two highs     
        for i in range (1, half_fan):
            price = (symbol_data['forecast_leading_high_1'] + symbol_data['high']) * (0.1 + (0.8 * i / half_fan))
            quantity = (dollars_to_spend + (i/3)) / (price + 0.0001)
            orders.append(Order(symbol=symbol, operation='sell', quantity=quantity, price=price))
        # buy halfway between the two lows
        for i in range (1, half_fan):
            price = (symbol_data['forecast_leading_low_1'] + symbol_data['low']) * (0.1 - (0.8 * i / half_fan))
            quantity = (dollars_to_spend + (i/3)) / (price + 0.0001)
            orders.append(Order(symbol=symbol, operation='buy', quantity=quantity, price=price))
        return orders
    
    def up_50_down_50(symbol, symbol_data):
         
        # logger.info(f"clear_up_or_down: symbol: {symbol} symbol_data: {symbol_data}")
        orders = []
        dollars_to_spend = 1
        symbol_data = symbol_data.iloc[0]
        # sell halfway between the two highs
        price = (symbol_data['forecast_leading_high_1'] + symbol_data['high']) / 2
        quantity = dollars_to_spend / (price + 0.0001)
        orders.append(Order(symbol=symbol, operation='sell', quantity=quantity, price=price))
        # buy halfway between the two lows
        price = (symbol_data['forecast_leading_low_1'] + symbol_data['low']) / 2
        quantity = dollars_to_spend / (price + 0.0001)
        orders.append(Order(symbol=symbol, operation='buy', quantity=quantity, price=price))
        return orders
    
    def clear_up_or_down_5_day(symbol, symbol_data):
        # logger.info(f"clear_up_or_down: symbol: {symbol} symbol_data: {symbol_data}")
        symbol_data = symbol_data.iloc[0]
        if symbol_data['forecast_leading_high_5'] > symbol_data['high'] and symbol_data['forecast_leading_low_5'] > symbol_data['low']:
            return [Order(symbol=symbol, operation='buy', quantity=1, price=symbol_data['high'])]
        elif symbol_data['forecast_leading_high_5'] < symbol_data['high'] and symbol_data['forecast_leading_low_5'] < symbol_data['low']:
            return [Order(symbol=symbol, operation='sell', quantity=1, price=symbol_data['high'])]
        return []
    
    
    def clear_up_or_down_3_day(symbol, symbol_data):
        # logger.info(f"clear_up_or_down: symbol: {symbol} symbol_data: {symbol_data}")
        symbol_data = symbol_data.iloc[0]
        if symbol_data['forecast_leading_high_3'] > symbol_data['high'] and symbol_data['forecast_leading_Low_3'] > symbol_data['low']:
            return 'buy'
        elif symbol_data['forecast_leading_high_3'] < symbol_data['high'] and symbol_data['forecast_leading_Low_3'] < symbol_data['low']:
            return 'sell'
        return 'hold'
    
    
    def clear_up_or_down_3_day(symbol, symbol_data):
        # logger.info(f"clear_up_or_down: symbol: {symbol} symbol_data: {symbol_data}")
        symbol_data = symbol_data.iloc[0]
        if symbol_data['forecast_leading_high_3'] > symbol_data['high'] and symbol_data['forecast_leading_low_3'] > symbol_data['low']:
            return 'buy'
        elif symbol_data['forecast_leading_high_3'] < symbol_data['high'] and symbol_data['forecast_leading_low_3'] < symbol_data['low']:
            return 'sell'
        return 'hold'
    
    
    def clear_up_or_down_1_day(symbol, symbol_data):
        # logger.info(f"clear_up_or_down: symbol: {symbol} symbol_data: {symbol_data}")
        symbol_data = symbol_data.iloc[0]
        if symbol_data['forecast_leading_high_1'] > symbol_data['high'] and symbol_data['forecast_leading_low_1'] > symbol_data['low']:
            return 'buy'
        elif symbol_data['forecast_leading_high_1'] < symbol_data['high'] and symbol_data['forecast_leading_low_1'] < symbol_data['low']:
            return 'sell'
        return 'hold'
    
    @timing.timer_func    
    def simulate_strategy_for_date(self,data_by_date, date):
        
        orders = self.strategy(data_by_date) 
        for order in orders:
            self.trade_simulator.submit_order(order, date)
     
    @timing.timer_func       
    def simulate_trades(self, date: datetime.datetime, expiration_duration=1):
        return self.trade_simulator.simulate_trades(date, expiration_duration, model_type=self.model_type)

    
    @timing.timer_func
    def simulate_strategy(self, start_date: datetime.datetime, end_date: datetime.datetime, expiration_duration=1):
        
        trade_data = self.data_handler.get_stock_data_normalized()
        start_date = pd.to_datetime(max(start_date, trade_data.index.get_level_values(self.data_handler.date_col).min())).normalize()
        end_date = pd.to_datetime(min(end_date, trade_data.index.get_level_values(self.data_handler.date_col).max())).normalize()
        starting_value = self.trade_simulator.get_current_price_value(start_date)
        # unique_symbols = trade_data.index.get_level_values('symbol').unique()
        logger.info(f"start_date: {start_date} end_date: {end_date}")
        for date in pd.date_range(start_date, end_date):    
            trade_data[trade_data.index.get_level_values(self.data_handler.date_col) == datetime.datetime(2024, 2, 12)]
            data_by_date = trade_data[trade_data.index.get_level_values(self.data_handler.date_col) == date]
            try:
                logger.info(f"date :{date} start_date: {start_date} end_date: {end_date}")
                logger.info(f"simulate_strategy: date: {date} len(data_by_date): {len(data_by_date)} strategy: {self.strategy}")
                self.simulate_strategy_for_date(data_by_date, date)
                self.simulate_trades(date, expiration_duration)
                self.print_balance(starting_value, start_date, date)
                
                value = self.trade_simulator.get_current_price_value(date)
                if value > self.max_value:
                    self.max_value = value
                if value < self.min_value:
                    self.min_value = value
            except Exception as e:
                logger.error(f"simulate_strategy: error: {e}")
                raise e

    @timing.timer_func
    def print_balance(self, starting_value, starting_date, date, print_holdings=False, in_logger: logging.Logger=logger):
        import math
        price_value = self.trade_simulator.get_current_price_value(date, print_holdings=print_holdings, in_logger=in_logger)
        if math.isnan(price_value):
            in_logger.error(f"TradeSimulator.print_balance: price_value is NaN")
            return
        cash = self.trade_simulator.portfolio.get_cash()
        stock_cost = self.trade_simulator.portfolio.get_stock_average_cost()
        stock_value = price_value - cash
        appreciation = stock_value - stock_cost
        portfolio_appreciation = price_value - starting_value
        in_logger.info(f"***************************************************")
        # in_logger.info(f"  day: {date} model_type: {self.model_type} strategy: {self.strategy.__name__} ")
        in_logger.info(f"  day: {date} total: {round(price_value, 2)}")
        in_logger.info(f"  day: {date} cash: {round(cash, 2)}")
        in_logger.info(f"  day: {date} stock value: {round(stock_value, 2)}")
        in_logger.info(f"  day: {date} stock cost: {round(stock_cost, 2)}")
        in_logger.info(f"  day: {date} stock appreciation: {round(appreciation, 2)}")
        in_logger.info(f"  day: {date} portfolio appreciation: {round(portfolio_appreciation, 2)}")
        in_logger.info(f"  day: {date} stock performance: {round(100.0 * appreciation / max(stock_cost, 1), 2)} %")
        in_logger.info(f"  day: {date} portfolio performance: {round(100.0 * portfolio_appreciation / starting_value, 2)} %")
        if print_holdings:           
            in_logger.info(f" starting_date:{starting_date} starting_value: {starting_value} day: {date} max value: {round(self.max_value, 2)} min value: {round(self.min_value, 2)}")
        self.trade_simulator.print_transaction_stats()
        in_logger.info(f"***************************************************")
        
        

