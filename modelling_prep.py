import concurrent
import datetime
from dotenv import load_dotenv
import json
import logging
import math
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import pickle
import random
from ratelimit import limits, sleep_and_retry
import requests
import tenacity as tnc
import timing
import yfinance as yfin

load_dotenv()
FMP_API_KEY = os.environ['FMP_API_KEY']
date_col = 'date'

# if run from notebook
from IPython.display import display

# Initialize logger
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

def fetch_and_save_yfin_data(tickers, start, end):
    
    data_path = os.path.join(os.getcwd(), "data")  # assuming data_path is the current working directory

    @tnc.retry(stop=tnc.stop_after_attempt(5), wait=tnc.wait_exponential(multiplier=1, min=2, max=10))
    def fetch_data(ticker):
        
        pickle_file_name = os.path.join(data_path, f"yfin_{ticker}.pkl".replace("/", "-"))
        csv_file_name = os.path.join(data_path, f"yfin_{ticker}.csv".replace("/", "-"))
        
        if os.path.exists(pickle_file_name):
            with open(pickle_file_name, 'rb') as file:
                df = pickle.load(file)
        else:
            if os.path.exists(csv_file_name):
                df = pd.read_csv(csv_file_name)
            else:
                df = yfin.download(ticker, start=start, end=end, progress=False, threads=False)
                df['symbol'] = ticker
                # df.to_csv(csv_file_name)
            logger.debug(f"before fetched_df.index {df.index}")
            df = df.reset_index().set_index([date_col, 'symbol'])
            logger.debug(f"after fetched_df.index {df.index}")
            with open(pickle_file_name, 'wb') as file:
                pickle.dump(df, file)

        return df

    yfin_df_pickle_file_name = os.path.join(data_path, "yfin_df.pkl")
    yfin_df_csv_file_name = os.path.join(data_path, "yfin_df.csv")
    logger.debug(f"yfin_df_pickle_file_name: {yfin_df_pickle_file_name}")
    logger.debug(f"yfin_df_csv_file_name: {yfin_df_csv_file_name}")
    if os.path.exists(yfin_df_pickle_file_name):
        logger.debug(f"Loading {yfin_df_pickle_file_name}")
        with open(yfin_df_pickle_file_name, 'rb') as file:
            yfin_df = pickle.load(file)
    elif os.path.exists(yfin_df_csv_file_name):
        logger.debug(f"Loading {yfin_df_csv_file_name}")
        yfin_df = pd.read_csv(os.path.join(data_path, "yfin_df.csv"))
    else:
        yfin_dfs = [] 
        logger.debug(f"Reconstructing yfin_df from tickers{tickers}")   
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map((lambda ticker: fetch_data(ticker)), tickers)
            yfin_dfs = [ticker_df for ticker_df in results if ticker_df is not None]
        logger.debug(f'before Concattenating yfin_dfs len: {len(yfin_dfs)} yfin_dfs.index: {yfin_dfs.index}')
        yfin_df = pd.concat(yfin_dfs, axis=0)
        yfin_df = yfin_df.reset_index().set_index([date_col, 'symbol']) # drop=True
        logger.debug(f'after Concattenating yfin_dfs len: {len(yfin_dfs)} yfin_dfs.index: {yfin_dfs.index}')
        # logger.debug(f"writing yfin_df to {yfin_df_csv_file_name}")
        # yfin_df.to_csv(os.path.join(data_path, "yfin_df.csv"))
        logger.debug(f"writing yfin_df to {yfin_df_pickle_file_name}")
        with open(yfin_df_pickle_file_name, 'wb') as file:
            pickle.dump(yfin_df, file)

    return yfin_df


def fetch_and_process_data(stock, endpoint, data_path):
    
    logger.debug(f"fetch_and_process_data stock: {stock} endpoint:  {endpoint}")
    endpoint_url_substring = endpoint if type(endpoint) == str else endpoint[0]
    # json_file_name = os.path.join(data_path, f"{endpoint_url_substring}_{stock}.json".replace("/", "-"))s
    pickle_file_name = os.path.join(data_path, f"{endpoint_url_substring}_{stock}.pkl".replace("/", "-"))
    loaded = False
    df_combined = None
    if os.path.exists(pickle_file_name):
        with open(pickle_file_name, 'rb') as file:
            logger.debug(f"Loading data from pickle file: {pickle_file_name}")
            try:
                df = pickle.load(file)
                loaded = True
            except Exception as e:
                logger.warning(f"Error loading data from pickle file: {pickle_file_name} {e}")
                pass
            
    if not loaded:
        # if os.path.exists(json_file_name):
        #     with open(json_file_name, 'r') as file:
        #         logger.debug(f"Loading data from json file: {json_file_name}")
        #         data = json.load(file)
        # else:
        response = api_call(stock, endpoint_url_substring)
        if response.status_code != 200:
            logger.error(f"Error fetching data from endpoint: {endpoint_url_substring} for stock: {stock} response.status_code: {response.status_code}")
            return None
        data = response.json()
        if not data:
            logger.error(f"Error fetching data from endpoint: {endpoint_url_substring} for stock: {stock} response.status_code: {response.status_code}")
            return None
            # with open(json_file_name, 'w') as file:
            #     json.dump(data, file, indent=4)
            #     logger.debug(f"JSON data saved to file: {json_file_name}")
        if type(endpoint) == tuple and len(endpoint) > 2:
            df = pd.DataFrame(data)
            dataframe_transformer = endpoint[2]
            df_combined = dataframe_transformer(df)
        else:
            data_getter = (lambda x: x) if type(endpoint) == str else endpoint[1]
            data_it = data_getter(data)
            # if not data_it or len(data_it) == 0:
            #     return None
            # logger.debug(f"len(data_it): {len(data_it)}")
            entries = [datapoint for datapoint in data_it]
            # rows = []
            dfs = []
            for entry in entries:
                if type(entry) == dict:
                    entry = {k:[v] for k,v in entry.items()}  # WORKAROUND
                    if 'symbol' not in entry:
                        entry['symbol'] = stock # handle non-batch endpoints that don't return symbol
                    df_temp = pd.DataFrame(entry, index=(['symbol', 'date']))
                    df_temp = df_temp.reset_index(drop=True).set_index(['symbol', 'date'])
                    dfs.append(df_temp)
                elif type(entry) == tuple:
                    df_temp = pd.DataFrame(**dict(entry[1]._asdict()), index=pd.MultiIndex(['symbol', 'date']))
                    # df_temp = df_temp.rename_axis(['symbol', 'date'])
                    # df_temp.index = pd.MultiIndex.from_tuples(df_temp.index)
                    dfs.append(df_temp)
                else:
                    df_temp = pd.DataFrame(dict(entry._asdict()), index=pd.MultiIndex(['symbol', 'date']))
                    df_temp = df_temp.rename_axis(['symbol', 'date'])
                    dfs.append(df_temp)        
                
            if len(dfs) > 0:
                df_combined = pd.concat(dfs, axis=0)
                
        if df_combined is None or len(df_combined) == 0:
            return None
            # df_combined = pd.DataFrame({'symbol': stock, 'date': '1970-01-01'}, index=['symbol', 'date'])
        
        logger.debug(f"symbol: {stock}  df_temp.shape: {df_combined.shape} len(df_temp): {len(df_combined)}")  

        # df_combined['symbol'] = stock # some non-batch endpoints don't always return symbol in every record
        df_combined.sort_index(inplace=True, ascending=True)
        df_combined = df_combined.apply(pd.to_numeric, errors='coerce').fillna(df_combined).drop_duplicates()
        logger.debug(f"saving to pickle file: {pickle_file_name}")
        with open(pickle_file_name, 'wb') as file:
            pickle.dump(df_combined, file)

    return df_combined

@sleep_and_retry
@limits(calls=300, period=datetime.timedelta(seconds=60).total_seconds())
@tnc.retry(stop=tnc.stop_after_attempt(5), wait=tnc.wait_exponential(multiplier=1, min=2, max=5))
def api_call(stock, endpoint_url_substring): # FMP_API_KEY
    endpoint_url = f"https://financialmodelingprep.com/api/v3/{endpoint_url_substring}/{stock}?period=quarter&apikey={FMP_API_KEY}"
    logger.debug(f"Fetching data from endpoint: {endpoint_url}")
    response = requests.get(endpoint_url)
    return response

ranks = {
    'Strong Buy': 1,
    'Buy': 1,
    'Accumlate': 1,
    'Outperform': 2,
    'Sector Outperform': 2,
    'Overweight': 2,
    'Hold': 3,
    'Neutral': 3,
    'Sector Perform': 3,
    'Equal-Weight': 3,
    'Underperform': 4,
    'Sector Underperform': 4,
    'Underweight': 4,
    'Sell': 5,
    'Strong Sell': 5,
}
rank_df = pd.DataFrame(ranks.items(), columns=['newGrade', 'rank'])
    
def process_combined(stock, data_path, start_date: datetime.datetime, end_date: datetime.datetime):
        
        df_combined = pd.DataFrame()
        
        logger.debug(f"process_combined stock: {stock} data_path: {data_path}")
        pickle_file_name = os.path.join(data_path, f"fmp_features_combined_{stock}.pkl")

        # takes json structure like sample_data to have a single structure per symbol-date with a key for every unique value of gradingCompany 
        # and a value for the newGrade
        def transform_grade_json(df): 
             
            # df = pd.DataFrame(data)
            joined_df = pd.merge(df, rank_df.set_index('newGrade'), on='newGrade', how='left')
            joined_df = joined_df.drop(['previousGrade', 'newGrade'], axis=1)
            joined_df = joined_df.set_index(['symbol', 'date'])
            joined_df['combined_name'] = "gradingCompany_newGrade_" + joined_df['gradingCompany'].replace(' ', '_', regex=True)
            df_pivoted = joined_df.pivot_table(columns='combined_name', values='rank', index=['symbol', 'date'], aggfunc='sum')
            # drop combined_name from index
            # df_pivoted.rename_axis(None, axis=1)
            # df_pivoted = df_pivoted.fillna(99999)

            return df_pivoted.drop_duplicates()
            
        loaded = False
        if os.path.exists(pickle_file_name):
            logger.debug(f"Loading combined data from pickle file: {pickle_file_name}")
            with open(pickle_file_name, 'rb') as file:
                try:
                    df_combined = pickle.load(file)
                    loaded = True
                except Exception as e:
                    logger.warning(f"Error loading data from pickle file: {pickle_file_name} {e}")
                    pass
        if not loaded:
            endpoints = [
                # 'grade',
                ('historical-price-full', (lambda x: x.get('historical'))),
                ('grade', None, transform_grade_json),
                'income-statement', 
                'balance-sheet-statement', 
                'cash-flow-statement', 
                ('historical-price-full/stock_dividend', (lambda x: x.get('historical'))),
                
                
            ]
            
            for endpoint in endpoints:
                logger.debug(f"endpoint: {endpoint}")
                logger.debug(f"len(endpoint): {len(endpoint)}")
                df_fetched = fetch_and_process_data(stock, endpoint, data_path)
                if df_fetched is None or df_fetched.empty or len(df_fetched) == 0:
                    continue
            
                logger.debug(f"{endpoint} df_fetched.index: {df_fetched.index}")
                logger.debug(f"{endpoint} df_fetched.columns: {df_fetched.columns}")
                logger.debug(f"{endpoint} df_combined.index: {df_combined.index}")
                logger.debug(f"{endpoint} df_combined.columns: {df_combined.columns}")
                cols_to_use = df_fetched.columns.difference(df_combined.columns)
                if (df_combined is not None and not df_combined.empty and len(df_combined) > 0):
                    logger.debug(f"df_fetched {df_fetched.shape} df_combined {df_combined.shape}")
                    df_combined = pd.merge(df_combined, df_fetched[cols_to_use], on=['symbol', 'date'], how='outer').sort_index()
                else:
                    df_combined = df_fetched
                logger.debug(f"{endpoint} df_combined.shape: {df_combined.shape}")
                logger.debug(f"{endpoint} df_combined.columns: {df_combined.columns}")

            # shareholders equity is assets - liabilities
            df_combined['AVG Shareholders Equity'] = (df_combined['totalStockholdersEquity']+df_combined['totalStockholdersEquity'].shift(4))/2
            df_combined['AVG Shareholders Equity'] = np.minimum(1, df_combined['AVG Shareholders Equity'])
            # ROE = return on equity = net income / avg shareholders equity
            df_combined['ROE'] = df_combined['netIncome']/df_combined['AVG Shareholders Equity'] 

            df_combined['AVG Assets'] = (df_combined['totalAssets']+df_combined['totalAssets'].shift(4))/2
            df_combined['AVG Assets'] = np.minimum(1, df_combined['AVG Assets'])
            df_combined['ROA'] = df_combined['netIncome']/df_combined['AVG Assets'] 
            
            # df_combined = df_combined.reset_index().set_index(['symbol', 'date'])
            # df_combined = df_combined.ffill()

            with open(pickle_file_name, 'wb') as file:
                pickle.dump(df_combined, file)

        return df_combined


def create_financials_df(stock, data_path, start_date: datetime.datetime, end_date: datetime.datetime):
    
    logger.debug(f"Creating financials dataframe for {stock}...")
    try:
        return process_combined(stock, data_path, start_date, end_date)
    except Exception as e:
        logger.error(f"Error creating financials dataframe for {stock}: {e} data_path: {data_path} start_date: {start_date} end_date: {end_date}")
        return None
        # raise e

def plot_ticker(stock, df, fig=None):
    
    if not fig:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
    logger.debug(f"Plotting ticker {stock}...")
    logger.debug(f"df.columns {df.columns}")
    
    for col in df.columns:
        if col in [
            'freeCashFlow',
            'Volume',
            
            ]:  
            column_plot = go.Scatter(
                x=df.index, 
                y=df[col], 
                name=str(col) + ' (Billion Dollar)',
                # fill='tonexty',
                connectgaps=True,
                # legendgroup="ROE Metrics",
                # legendgrouptitle_text="ROE Metrics (Billion Dollar)",
            )
            fig.add_trace(column_plot)
        elif col in [
            'Market Cap',
            'netIncome', 
            'shortTermDebt',
            'longTermDebt',
            'dividendsPaid,'
            'totalDebt',
            'totalAssets',
            'totalStockholdersEquity',
            'totalLiabilities',
            'weightedAverageShsOut',
            'cashAndCashEquivalents', 
            'shortTermInvestments',
            'commonStockSharesOutstanding',
            'commonStockEquity',
            'commonStock',
            'dividendsPaid'
            ]:  
            column_plot = go.Scatter(
                x=df.index, 
                y=df[col], 
                name=str(col) + ' (Billion Dollar)',
                # fill='tonexty',
                connectgaps=True,
                visible='legendonly',
                # legendgroup="Liabilities Metrics",
                # legendgrouptitle_text="Liabilities (Billion Dollar)",
            )
            fig.add_trace(column_plot)    
        # elif col in [
        #     'Cash and cash equivalents',
        #     'Long-term investments',
        #     'Receivables',
        #     'Short-term investments',
        #     ]:  
        #     column_plot = go.Scatter(
        #         x=df.index, 
        #         y=df[col], 
        #         name=str(col),
        #         # fill='tonexty',
        #         connectgaps=True,
        #         visible='legendonly',
        #         legendgroup="Other Metrics",
        #         legendgrouptitle_text="Other Metrics (Billion Dollar)",)
        #     fig.add_trace(column_plot)  
        elif col in [
            'P/E TTM',
            'EPS TTM',
            'log(P/E TTM)',
            'label',
            
            ]:  
            column_plot = go.Scatter(
                x=df.index, 
                y=df[col], 
                name=str(col) + ' (Dollars)',
                # fill='tonexty',
                yaxis='y2',
                connectgaps=True,
                # legendgroup="Dividend Metrics (Dollars)",
                # legendgrouptitle_text="Dividend Metrics (Dollars)",
            )
            fig.add_trace(column_plot) 
        elif col in [
            'Dividend per Share',
            'adjDividend'
            'dividend',
            'eps',
            ]:  
            column_plot = go.Scatter(
                x=df.index, 
                y=df[col], 
                name=str(col) + ' (Dollars)',
                # fill='tonexty',
                yaxis='y2',
                connectgaps=True,
                visible='legendonly',
                # legendgroup="Dividend Metrics (Dollars)",
                # legendgrouptitle_text="Dividend Metrics (Dollars)",
            )
            fig.add_trace(column_plot) 
        elif col in [
            'ROE', 
            'ROA',
            'HL_PCT',
            'PCT_change',
            ]:
            column_plot = go.Scatter(
                x=df.index, 
                y=df[col], 
                name=str(col) + ' (Ratio)',
                # fill='tonexty',
                connectgaps=True,
                yaxis='y3',
                # visible='legendonly',
                # legendgroup="Ratio Metrics",
                # legendgrouptitle_text="Ratio Metrics",
                )
            fig.add_trace(column_plot)
        elif col in [
            'netProfitMargin',
            'netIncomeRatio',
            'Price to Book',
            ]:
            column_plot = go.Scatter(
                x=df.index, 
                y=df[col], 
                name=str(col) + ' (Ratio)',
                # fill='tonexty',
                connectgaps=True,
                yaxis='y3',
                visible='legendonly',
                # legendgroup="Ratio Metrics",
                # legendgrouptitle_text="Ratio Metrics",
                )
            fig.add_trace(column_plot)
            
            
import modelling_prep as mp
import importlib
importlib.reload(mp)

def plot_ichimoku(fig, df):
    
    def get_fill_color(label):
        if label >= 1:
            return 'rgba(0, 250, 0, 0.4)'
        else:
            return 'rgba(250, 0, 0, 0.4)'
    
    # base and coversion lines
    fig.add_traces(go.Scatter(
        legendgroup="Ichimoku Clouds (Dollar)", legendgrouptitle_text="Ichimoku Clouds (Dollar)", yaxis='y2', 
        visible='legendonly',
        x=df.index, y=df['baseline'], name="Baseline (trailing 52 MA)", line=dict(color='rgba(255,64,255,0.6)')))
    fig.add_traces(go.Scatter(
        legendgroup="Ichimoku Clouds (Dollar)", legendgrouptitle_text="Ichimoku Clouds (Dollar)", yaxis='y2',
        visible='legendonly',
        x=df.index, y=df['conversion'], name="Conversion (trailing 9 MA)", line=dict(color='rgba(64,64,255,0.6)')))
    
    # ichimoku clouds
    df_copy = df.copy(deep=True)  # Make a copy of the dataframe
    df_copy['label'] = np.where(df_copy['spanA'] > df_copy['spanB'], 1, 0)
    df_copy['group'] = df_copy['label'].ne(df_copy['label'].shift()).cumsum()
    df_copy_groups = df_copy.groupby('group')
    
    dfs = []
    for group, df_arg in df_copy_groups:
        fig.add_traces(go.Scatter(
            legendgroup="Ichimoku Clouds (Dollar)", legendgrouptitle_text="Ichimoku Clouds (Dollar)", yaxis='y2',
            visible='legendonly',
            x=df_arg.index, line=dict(color='rgba(0,0,0,0)'), y=df_arg['spanA'], name="Span A", showlegend=False))
        fig.add_traces(go.Scatter(
            legendgroup="Ichimoku Clouds (Dollar)", legendgrouptitle_text="Ichimoku Clouds (Dollar)", yaxis='y2',
            visible='legendonly',
            x=df_arg.index, line=dict(color='rgba(0,0,0,0)'), y=df_arg['spanB'], fill='tonexty', showlegend=False, fillcolor=get_fill_color(df_arg['label'].iloc[0]), name="Span B"))       

default_whitelist_symbols = ['AMZN', 'GOOG']

def plot_features(df, whiltelist_symbols=default_whitelist_symbols, in_fig=None):
    
    fig = in_fig or go.Figure()
    
    for stock, df_input in df.groupby(by='symbol',group_keys=False):
        
        if not whiltelist_symbols or stock not in whiltelist_symbols:
            continue
         
        df = df_input.reset_index().set_index(date_col).sort_index(ascending=True)  # Make a copy of the dataframe
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # plot the modelling_prep metrics
        mp.plot_ticker(stock, df, fig=fig)
        
        # plot the candlestick price data
        fig.add_trace(go.Candlestick(
                    name="Stock Price (Candlestick)",
                    x=df.index,
                    yaxis='y2',
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['adjClose'],
                    legendgroup="Dollar Price",
                    legendgrouptitle_text="Dollar Price",
                    # visible='legendonly',
                    ))
        
        # plot Ichimoku-related plots
        plot_ichimoku(fig, df)

        fig.add_trace(
            go.Scatter(
                    x=list(df.index),
                    y=list(df.high),
                    legendgroup="Stock Price (High/Low)", legendgrouptitle_text="Stock Price (High/Low Dollar)", yaxis='y2',
                    # visible='legendonly',
                    name="High",
                    line=dict(color="#33CFA5")))

        fig.add_trace(
            go.Scatter(x=list(df.index),
                    y=[df.high.mean()] * len(df.index),
                    legendgroup="Stock Price (High/Low)", legendgrouptitle_text="Stock Price (High/Low Dollar)", yaxis='y2',
                    name="High Average",
                    visible=False,
                    line=dict(color="#33CFA5", dash="dash")))

        fig.add_trace(
            go.Scatter(x=list(df.index),
                    y=list(df.low),
                    legendgroup="Stock Price (High/Low)", legendgrouptitle_text="Stock Price (High/Low Dollar)", yaxis='y2',
                    # visible='legendonly',
                    name="low",
                    line=dict(color="#F06A6A")))

        fig.add_trace(
            go.Scatter(x=list(df.index),
                    y=[df.low.mean()] * len(df.index),
                    legendgroup="Stock Price (High/Low)", legendgrouptitle_text="Stock Price (High/Low Dollar)", yaxis='y2',
                    name="low Average",
                    visible=False,
                    line=dict(color="#F06A6A", dash="dash")))

        
        fig.update_layout(
            title=stock, 
            height=500, 
            width=700, 
            showlegend=True, 
            autotypenumbers='convert types',
            xaxis=dict(
                domain=[0.05, 0.95]
            ),
            # pass the y-axis title, titlefont, color
            # and tickfont as a dictionary and store
            # it an variable yaxis
            yaxis=dict(
                title="Metrics (Dollars Billion)",
                titlefont=dict(
                    color="#0000ff"
                ),
                tickfont=dict(
                    color="#0000ff"
                ),
                anchor="free",
                position=0.0  # specifying the position of the axis 
            ),
            
            # pass the y-axis 2 title, titlefont, color and
            # tickfont as a dictionary and store it an
            # variable yaxis 2
            yaxis2=dict(
                title="Price (Dollars)",
                titlefont=dict(
                    color="#FF0000"
                ),
                tickfont=dict(
                    color="#FF0000"
                ),
                anchor="free",  # specifying x - axis has to be the fixed
                overlaying="y",  # specifyinfg y - axis has to be separated
                side="left",  # specifying the side the axis should be present
                position=0.05  # specifying the position of the axis
            ),
        
            # pass the y-axis 3 title, titlefont, color and 
            # tickfont as a dictionary and store it an
            # variable yaxis 3
            yaxis3=dict(
                title="Ratio",
                titlefont=dict(
                    color="#006400"
                ),
                tickfont=dict(
                    color="#006400"
                ),
                anchor="free",     # specifying x - axis has to be the fixed
                overlaying="y",  # specifyinfg y - axis has to be separated
                side="right",  # specifying the side the axis should be present
                position=0.95  # specifying the position of the axis
            ),
        )
        
        if not in_fig:
            fig.show()
            
def _join_forecast_to_actuals(
    actuals_df, 
    forecast_df, 
    actual_col, 
    forecast_leading_days, 
    classify_change_threshold_ratio_up,
    classify_change_threshold_ratio_down,
    classification_comparison_column,
    ):

    actual_cols_to_keep = set(['symbol', date_col, 'label_date', 'open', 'close', 'high', 'low', 'adjClose'])
    if classification_comparison_column is not None:
        actual_cols_to_keep.add(classification_comparison_column)
    if actual_col not in actual_cols_to_keep:
        actual_cols_to_keep.add(actual_col)
    actual_cols_to_keep = list(actual_cols_to_keep)
    actuals_df_copy = actuals_df.reset_index()[actual_cols_to_keep].copy(deep=True)
    # simple way to join with rows whose label_date is our date
    actuals_df_copy_2 = actuals_df_copy[['symbol', date_col, 'label_date', actual_col]].copy() # Rename 'label_date' in the copy to date_col for the merge
    actuals_df_copy_2 = actuals_df_copy_2.rename(
            columns={
                    'label_date': date_col,
                    date_col: 'label_date'
                    })
    # join with the actuals for whose date (in reverse) we are the future target 
    actuals_df_copy = pd.merge(actuals_df_copy, actuals_df_copy_2, on=['symbol', date_col], how='outer', suffixes=('', '_reverse'))
    if classification_comparison_column is not None:
        actuals_df_copy = mp.classify_func_transform(
            actuals_df_copy,
            classification_comparison_column, 
            actual_col + '_reverse', 
            classify_change_threshold_ratio_up,
            classify_change_threshold_ratio_down,
            'class_actual',         
            )
    
    sameday_forecast_df_copy = forecast_df.reset_index()[['symbol', date_col, 'forecast']].copy(deep=True).reset_index()
    sameday_forecast_df_copy[date_col] = pd.to_datetime(sameday_forecast_df_copy[date_col])
    sameday_forecast_df_copy.set_index(['symbol', date_col])
    
    target_forecast_df_copy = forecast_df.reset_index()[['symbol', date_col, 'forecast']].copy(deep=True).reset_index()
    target_forecast_df_copy['label_date_reverse'] = pd.to_datetime(target_forecast_df_copy[date_col])
    target_forecast_df_copy.set_index(['symbol', 'label_date_reverse'])
    
    logger.debug(f"target_forecast_df_copy.dtypes: {target_forecast_df_copy.dtypes}")
    # target_forecast_df_copy.dtypes.to_csv('target_forecast_df_copy.csv')
    
    forecast_actuals_df = pd.merge(
            actuals_df_copy, 
            sameday_forecast_df_copy[['symbol', date_col, 'forecast']], 
            on=['symbol', date_col], 
            how='outer')
    
    forecast_actuals_df = pd.merge(
            forecast_actuals_df, 
            target_forecast_df_copy[['symbol', 'label_date_reverse', 'forecast']], 
            on=['symbol', 'label_date_reverse'], 
            how='outer', 
            suffixes=('_leading', '_target'))
    
    forecast_actuals_df['actual_column_name'] = actual_col
    forecast_actuals_df['forecast_leading_days'] = forecast_leading_days

    return forecast_actuals_df


def join_forecast_to_actuals(
    actuals_df, forecast_df, actual_col, forecast_leading_days, 
    classify_change_threshold_ratio_up,
    classify_change_threshold_ratio_down,
    classification_comparison_labels=None):
    df = _join_forecast_to_actuals (
        actuals_df, forecast_df, actual_col, forecast_leading_days, 
        classify_change_threshold_ratio_up,
        classify_change_threshold_ratio_down,
        classification_comparison_labels)
    return df

# TODO - change this to alter the row via vector sets rather than a python function
# def classify_func(forecast, actual, clasiify_change_threshold_ratio_up, clasiify_change_threshold_ratio_down):
#     def return_func(row):
#         if ((row[forecast] - row[actual]) / abs(row[actual])) >= clasiify_change_threshold_ratio_up:
#             return 1
#         elif ((row[forecast] - row[actual]) / abs(row[actual])) <= clasiify_change_threshold_ratio_down:
#             return -1
#         else:
#             return 0
#     return return_func

@timing.timer_func
def classify_func_transform(
    df,
    forecast_col,
    actual_col, 
    classify_change_threshold_ratio_up, 
    classify_change_threshold_ratio_down, 
    output_column_name):

        mask_price_lower = ((df[forecast_col] - df[actual_col]) / np.abs(df[actual_col]) <= classify_change_threshold_ratio_down)
        mask_price_higher = ((df[forecast_col] - df[actual_col]) / np.abs(df[actual_col]) >= classify_change_threshold_ratio_up)
        
        higher_df = df[mask_price_higher].reset_index()
        lower_df = df[mask_price_lower].reset_index()
        equal_df = df[(~(mask_price_higher | mask_price_lower))].reset_index()

        higher_df[output_column_name] = 1
        lower_df[output_column_name] = -1
        equal_df[output_column_name] = 0
        
        df = None
        lower_df = lower_df.set_index(['symbol', date_col])
        equal_df = equal_df.set_index(['symbol', date_col])
        higher_df = higher_df.set_index(['symbol', date_col])
            
        df = pd.concat([lower_df, equal_df, higher_df], axis=0)
        # df = df.set_index(['symbol',date_col])
        
        return df
       


def classify_and_store(

        forecast,
        classify_confidence_threshold_up,
        classify_confidence_threshold_down,
        column_name,
        negative_column=-1,
        equal_column=0,
        positive_column=1):

            lower_prob = forecast[negative_column]
            equal_prob = forecast[equal_column]
            higher_prob = forecast[positive_column] 

            mask_price_lower = ((lower_prob > equal_prob) & (lower_prob > higher_prob) & (lower_prob > classify_confidence_threshold_down))
            mask_price_higher = ((higher_prob > equal_prob) & (higher_prob > lower_prob) & (higher_prob > classify_confidence_threshold_up))

            higher_df = forecast[mask_price_higher].reset_index()
            lower_df = forecast[mask_price_lower].reset_index()
            equal_df = forecast[(~(mask_price_higher | mask_price_lower))].reset_index()

            # 'forecast'
            higher_df[column_name] = 1
            lower_df[column_name] = -1
            equal_df[column_name] = 0
            
            forecast = pd.concat([lower_df, equal_df, higher_df], axis=0)
            forecast = forecast.set_index(['symbol',date_col])
            return forecast
        
@timing.timer_func
def draw_forecast_and_actuals(df_input, stock, actual_col, draw_class_actuals=False, model_type=None, training_config=None):
    
    import sklearn_util as sklu
    import importlib
    importlib.reload(sklu)

    if draw_class_actuals:
        metrics_df, metrics_string = sklu.calculate_metrics(df_input, 'forecast_target', 'class_actual')
    else:
        metrics_df, metrics_string = sklu.calculate_metrics(df_input, 'forecast_target', actual_col )
    
    df_close_date = df_input.copy(deep=True).reset_index().set_index(date_col).sort_index(ascending=True) 
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        name=stock+": leading forecast",
        x=df_close_date.index,
        y=df_close_date['forecast_leading'],
        mode='lines+markers',  # Add markers
        marker=dict(opacity=0.5),
        # visible='legendonly',
        ))

    fig.add_trace(go.Scatter(
            name=stock+": target forecast",
            x=df_close_date.index,
            y=df_close_date['forecast_target'],
            mode='lines+markers',  # Add markers
            visible='legendonly',
            ))

    fig.add_trace(go.Scatter(
            name=stock+f": LogadjClose",
            x=df_close_date.index,
            y=df_close_date[f"LogadjClose"],
            mode='lines+markers',  # Add markers
            # visible='legendonly',
            ))     

    fig.add_trace(go.Scatter(
            name=stock+f": {actual_col}",
            x=df_close_date.index,
            y=df_close_date[f"{actual_col}"],
            mode='lines+markers',  # Add markers
            visible='legendonly',
            ))

    if draw_class_actuals:
        fig.add_trace(go.Scatter(
                name=stock+f": class_actual",
                x=df_close_date.index,
                y=df_close_date[f"class_actual"],
                mode='lines+markers',  # Add markers
                visible='legendonly',
                ))    
    
    fig.update_layout(
        title=stock, 
        height=300, 
        width=900, 
        showlegend=True, 
        autotypenumbers='convert types',
        title_text=f"{stock} {model_type[0:5]} hrzn: {training_config.forecast_out} {metrics_string}"
        )
    
    fig.show()      
            
def remove_high_nan_columns(df, threshold=0.01, ignore=[]):
    # Calculate the percentage of NaN values in each column
    # nan_percentages = df.dropna(how='all').isna().mean()
   
    # columns_to_remove = get_high_nan_columns(df, threshold, ignore)
    # columns_to_keep = [col for col in df.columns if (col in columns_to_remove and col not in ignore) ]
    # columns_to_keep = pd.Index(columns_to_keep)
    # Return a new DataFrame with only the columns to keep
    # return df[columns_to_keep]
    
    return remove_columns(df, get_high_nan_columns(df, threshold, ignore))

def remove_columns(df, columns_to_remove):
    # Return a new DataFrame with only the columns to keep
    return df.drop(columns=columns_to_remove)

def get_high_nan_columns(df, threshold=0.01, ignore=[]):
    
    nan_percentages = df.isna().mean()
    columns_to_remove = nan_percentages[nan_percentages >= threshold].index.tolist()
    return columns_to_remove
                      


            


            
            





