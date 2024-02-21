
import datetime
import logging
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import timing

logging.basicConfig(level=logging.INFO)
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


@timing.timer_func
def scale_and_get_scalar(df):

    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler

    # create an abs_scaler object
    scaler = MaxAbsScaler()
    # calculate the maximum absolute value for scaling the data using the fit method
    scaler.fit(df)
    # transform the data using the parameters calculated by the fit method (the maximum absolute values)
    scaled_data = scaler.transform(df)
    # store the results in a data frame
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    # visualize the data frame
    return df_scaled, scaler

def calculate_metrics(df, forecast_col, actual_col):
    
    # Assuming 'forecast' and 'actuals' are the column names for the forecasted and actual values
    dropna_df = df.dropna(subset=[forecast_col, actual_col])
    dropna_df = dropna_df[((dropna_df[forecast_col] != 0) | (dropna_df[actual_col] != 0))]
    if dropna_df.shape[0] == 0:
        return dropna_df, " No forecasts"

    forecast = dropna_df[forecast_col]
    actuals = dropna_df[actual_col]
    logger.debug(f"forecast: {forecast.shape}")
    logger.debug(f"actuals: {actuals.shape}")

    mae = mean_absolute_error(actuals, forecast)
    mse = mean_squared_error(actuals, forecast)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(actuals - forecast)) * 100
    mte = np.sum(np.abs(actuals - forecast)) / (actuals.size)
    
    # format for 2 degrees of precision on floats and 0 on integers and percentages
    metrics_string_format = "{:.2f}" if mae < 10 else "{:.1f}"
    mae = metrics_string_format.format(mae)
    mse = metrics_string_format.format(mse)
    rmse = metrics_string_format.format(rmse)
    mape = metrics_string_format.format(mape)
    mte = metrics_string_format.format(mte)
    negative_actuals = actuals[actuals < 0].shape[0]
    positive_actuals = actuals[actuals > 0].shape[0]
    neutral_actuals = actuals[actuals == 0].shape[0]
    negative_forecast = forecast[forecast < 0].shape[0]
    positive_forecast = forecast[forecast > 0].shape[0]
    neutral_forecast = forecast[forecast == 0].shape[0]
    total_actuals = actuals.dropna().shape[0]
    logger.info(f" negative_actuals: {negative_actuals} neutral_actuals: {neutral_actuals} positive_actuals: {positive_actuals} total_actuals: {total_actuals}" +
        f"\n negative_forecast: {negative_forecast} neutral_forecast: {neutral_forecast} positive_forecast: {positive_forecast}")
    # confusion matrix for actuals vs forecast
    confusion_matrix = pd.DataFrame()
    confusion_matrix['actuals'] = [negative_actuals, neutral_actuals, positive_actuals]
    confusion_matrix['forecast'] = [negative_forecast, neutral_forecast, positive_forecast]
    confusion_matrix.index = ['negative', 'neutral', 'positive']
    confusion_matrix['total'] = confusion_matrix['actuals'] + confusion_matrix['forecast']
    confusion_matrix.loc['total'] = confusion_matrix.sum()
    confusion_matrix['total'] = confusion_matrix.sum(axis=1)
    
    
    metric_string = f"\n MAE: {mae} MSE: {mse} RMSE: {rmse} MAPE: {mape} % MTE: {mte}"
    # logger.debug("Metrics: " + metric_string)
    
    return dropna_df, metric_string

def get_best_features(input_X, output_col, k=10):
    
    # Use f_regression to select the 10 most important features
    selector = SelectKBest(score_func=f_regression, k=k)
    # get the top k transforms
    top_k_features = selector.fit_transform(input_X.drop(output_col, axis=1), input_X[[output_col]])
    # which were selected? add those to list
    mask = selector.get_support() #list of booleans
    new_features = [] # The list of your K best features
    for bool_val, feature in zip(mask, input_X.columns):
        if bool_val:
            new_features.append(feature)
     # Create new consisting of the original data using its names, with only the top k columns
    dataframe = pd.DataFrame(top_k_features, columns=new_features, index=input_X.index)
    return dataframe, new_features


# get importance
def get_importance(importance, training_df):

    # # summarize feature importanceß
    # for i,v in enumerate(importance):
    #     logger.info(f'Feature: {training_df.columns[i]}, Score: {v}')
        
    importances_df = pd.DataFrame()
    importances_df['coef'] = importance
    importances_df['features'] = training_df.columns
    importances_df = importances_df.sort_values(by='coef', ascending=False)

    # logger.info(f"importance: {importance} len(importance): {len(importance)}")
    # logger.info(f"training_X.columns: {training_X.columns} len(training_X.columns): {len(training_X.columns)}")

    # importance, index=training_X.columns, columns=training_X.columns)
    
    # importances_df[np.abs(importances_df['coef']) >= 10] # .sort_values(by='coef').to_csv(f'model_importances_{name}.csv')
    return importances_df

def draw_forecast_and_actuals(df_input):
    
    # metrics_df, metrics_string = calculate_metrics(df_input, 'forecast_target', actual_col )
    
    columns_to_use = list(df_input.columns) # assume don't want indexed
    columns_not_to_use = ['symbol', 'model_type', 'Date', 'index']
    # get the non-index columns to use for plotting (assume index is 'Date') 
    for not_to_use in columns_not_to_use:
        if not_to_use in columns_to_use:
            columns_to_use.remove(not_to_use)   
            
    df_ordered = df_input.copy(deep=True).reset_index().set_index('Date').sort_index(ascending=True) 
    
    for symbol, symbol_df in df_ordered.groupby('symbol'):

        fig = go.Figure()
        
        for model_type, model_type_df in symbol_df.groupby('model_type'):
            # df_ordered.loc[model_type_df.index, 'forecast_'+model_type] = model_type_df['forecast']
            # df_ordered.loc[model_type_df.index, 'forecast_'+model_type+'_upper'] = model_type_df['forecast_upper']
            # df_ordered.loc[model_type_df.index, 'forecast_'+model_type+'_lower'] = model_type_df['forecast_lower']
        
            for column in columns_to_use:
                fig.add_trace(go.Scatter(
                        name=f"{symbol} {model_type} {column}",
                        x=model_type_df.index,
                        y=model_type_df[column],
                        visible='legendonly',
                        ))

        fig.update_layout(
            title=symbol, 
            height=700, 
            width=1200, 
            showlegend=True, 
            autotypenumbers='convert types',
            title_text=symbol # + metrics_string
            )
            
        fig.show()
    