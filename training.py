
import config as cfg
import datetime
import pandas as pd
import importlib
from itertools import product
import logging
import modelling_prep as mp
import numpy as np
import os
import pickle
import plotly.graph_objects as go
import sklearn_util as sklu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import timing

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not os.path.exists('logs'):
    os.makedirs('logs')
filename = f"logs/{__name__}_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log"
if not os.path.exists(filename):
    with open(filename, 'wt') as f:
        f.write(f'*** {filename} ***\n')   


importlib.reload(cfg)
importlib.reload(mp)
importlib.reload(sklu)

# if run from notebook
from IPython.display import display


def fill_missing_dates(df, conf, end_date=None):
    
    # # df = inpupt_df.sort_values(by=[conf.date_col])
    # # Create a new DataFrame with all possible dates for each symbol
    # symbols = df.index.get_level_values('symbol').unique()
    # dates = pd.date_range(start=df.index.get_level_values(conf.date_col).min(), end=end_date, freq='D')
    # # Create all combinations of symbols and dates
    # combinations = list(product(symbols, dates))
    # # Convert these combinations into a DataFrame
    # df_new = pd.DataFrame(combinations, columns=['symbol', conf.date_col]).set_index(['symbol',conf.date_col])
    # df = pd.merge(df_new, df, how='left', left_index=True, right_index=True)
    # # df = df.set_index(['symbol',conf.date_col])
    
    df.reset_index(inplace=True)
    
    # Create a new date range that covers all the dates
    all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')

    # Reindex the DataFrame to include all dates
    df.reindex(all_dates)
    
    df.reset_index(inplace=True, drop=True)
    
    df.set_index(['symbol',conf.date_col], inplace=True)

    # Forward fill the missing values
    # df.ffill(inplace=True)
    
    return df

def train_models(training_config: cfg.TrainingConfig, conf: cfg.Config, df: pd.DataFrame):
    with pd.option_context(
            # 'display.max_rows', None, 
            'display.max_columns', None,
            'display.max_colwidth', None):
            # verify that kwargs keys are equal to the contents of grouping_dimensions

        @timing.timer_func
        def fill_dates(training_config, conf, df):
            logger.info(f"training_config: {training_config.__dict__}")
            logger.info(f"df (input): {df.shape}")
                # display(df)
                #df.groupby(by='symbol').apply(fill_missing_dates, conf=conf, end_date=max_date).reset_index(level=0, drop=True)
            logger.info("filling missing dates {df.shape}")
            df = fill_missing_dates(df, conf=conf) # .reset_index(level=0, drop=True)
            logger.debug(f"filled missing: {df.shape}")
            # display(df[[training_config.forecast_col, 'LogadjClose']].head(20))
            return df
        df = fill_dates(training_config, conf, df)
        
        # @timing.timer_func
        def add_labels(df):
            # fill from the reverse direction to get the most recent non-NaN value
            # dfreg[forecast_col] = dfreg[forecast_col].bfill()
            # df['label'] = df[training_config.forecast_col].shift(-training_config.forecast_out)
            # df['label'].bfill(inplace=True)
            # df['label_date'] = df.index.get_level_values(conf.date_col).shift(training_config.forecast_out,freq='D', ) # assume daily close values
            label_assignment = {
                'label': df[training_config.forecast_col].bfill().shift(-training_config.forecast_out),
                'label_date': df.index.get_level_values(conf.date_col).shift(training_config.forecast_out,freq='D', ) # assume daily close values
            }
            df = df.assign(**label_assignment)
            return df
  
        logger.info(f"adding labels: {df.shape}")
        target_training_df = df.groupby(by='symbol', group_keys=False).apply(add_labels)
        df = None
        logger.debug(f"added labels: {target_training_df.shape}")
        # display(target_training_df[['label', 'label_date', training_config.forecast_col, 'LogadjClose']].head(20))
        
        @timing.timer_func
        def fill_numerics(df):
            logger.info(f"forward filling numerics: {df.shape}")
            # target_training_filled_df = target_training_df.ffill()
            # df = df.sort_values('symbol')
            df.sort_index(inplace=True)
            df = df.groupby('symbol').ffill()
            # df = df.ffill() * (1 - df.isnull().astype(int)).groupby('symbol').cumsum().map(lambda x: None if x == 0 else 1)
            logger.debug(f"forward filled numerics: {df.shape}")
            return df
        target_training_filled_df = fill_numerics(target_training_df)
        target_training_df = None
        # display(target_training_filled_df[['label', 'label_date', training_config.forecast_col, 'LogadjClose']].head(20))

        logger.info(f"classifying: {target_training_filled_df.shape}")
        target_training_classified_df = mp.classify_func_transform(
            target_training_filled_df,
            forecast_col='label', 
            actual_col=training_config.forecast_col,
            classify_change_threshold_ratio_up=training_config.classify_change_threshold_ratio_up,
            classify_change_threshold_ratio_down=training_config.classify_change_threshold_ratio_down,
            output_column_name='class'
        )
        # display(target_training_classified_df[['label', 'label_date', training_config.forecast_col, 'LogadjClose']].head(20))
        logger.debug(f"classified: {target_training_classified_df.shape}")
        target_training_filled_df = None

        @timing.timer_func
        def get_training_and_holdout(df: pd.DataFrame):
            df = df.reset_index()
            df[conf.date_col] = pd.to_datetime(df[conf.date_col])
            max_date = min(np.max(df[conf.date_col]), conf.end_date)
            min_date = max(np.min(df[conf.date_col]), conf.start_date)
            mask_training = ((df[conf.date_col] >= min_date + datetime.timedelta(days=conf.largest_forecast_out)) & (df[conf.date_col] < max_date + datetime.timedelta(days=-(conf.total_holdout_base))))
            # training and holdout will overlap by total_holdout_base because holdout needs it for timseries offset features. We need to move back holdout enough
            # for features (largest_day_offset) in addition to total_holdout_base because when we simualte it will want actuals for those days
            mask_holdout = (df[conf.date_col] >= max_date + datetime.timedelta(days = -(conf.total_holdout_base+conf.largest_forecast_out)))
            training = df[mask_training]
            holdout = df[mask_holdout]  
            training = training.set_index(['symbol',conf.date_col])
            holdout = holdout.set_index(['symbol',conf.date_col])
            return training, holdout

        training, holdout = get_training_and_holdout(target_training_classified_df)
        min_training_date = np.min(training.index.get_level_values(conf.date_col))
        max_training_date = np.max(training.index.get_level_values(conf.date_col))
        min_holdout_date = np.min(holdout.index.get_level_values(conf.date_col))
        max_holdout_date = np.max(holdout.index.get_level_values(conf.date_col))
        
        logger.info(f"split train: {training.shape} from: {min_training_date} to: {max_training_date} and holdout: {holdout.shape} from: {min_holdout_date} to: {max_holdout_date}")
        # display(training, holdout)
    
        # holdout_filled = fill_missing_dates(holdout, conf=conf, end_date=max_holdout_date) # .reset_index(level=0, drop=True)

        # max_date = np.max(training.index.get_level_values('Date'))
        training_sample = training.replace([np.inf, -np.inf], np.nan).dropna().sample(n=training_config.num_training_samples, random_state=1, axis=0)
        holdout_filled = holdout.replace([np.inf, -np.inf], np.nan).dropna()
        training_X = training_sample.drop(['label', 'class'], axis=1).select_dtypes(include='number')
        holdout_X = holdout_filled.drop(['label', 'class'], axis=1).select_dtypes(include='number')
        training_y = np.array(training_sample['label'])
        training_class = np.array(training_sample['class'])

        # display(best_features)
        holdout_X = holdout_X[training_X.columns]
        features = training_X.columns # features + list(programmatic_features)
        logger.info(f"features {features}")
        training_x_scaled, training_x_scaler = sklu.scale_and_get_scalar(training_X[features])
        scaled_features_holdout = holdout_X[features]
        holdout_x_scaled = pd.DataFrame(training_x_scaler.transform(scaled_features_holdout), columns=scaled_features_holdout.columns, index=scaled_features_holdout.index)

        # # # Split the data into a training set and an evaluation set
        X_train, X_eval, y_train, y_eval, class_train, class_eval = train_test_split(training_x_scaled, training_y, training_class, test_size=0.02) #, random_state=42)

        logger.info("Training and evaluation data shapes:")
        logger.info(f"X_train: {X_train.shape} X_eval: {X_eval.shape} y_train: {y_train.shape} y_eval: {y_eval.shape} class_train: {class_train.shape} class_eval: {class_eval.shape}")
        
        # for training_config in conf.training_configs:

        @timing.timer_func
        def attempt_cache(model_path, filename, lambda_func):

            # if os.path.exists(os.path.join(model_path, filename)):
            #     return pickle.load(open(os.path.join(model_path, filename), 'rb'))
            
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            
            model = lambda_func()
            
            try:
                score = model.score(X_eval, y_eval)
                logger.info(f"model: {model} score: {score}")
            except Exception as e:
                logger.warning(f"Error getting score for model: {model} {e}")
            
            pickle.dump(model, open(os.path.join(model_path, filename), 'wb'))
            
            
            return model

        model_type = 'log_regression'
        
        @timing.timer_func
        def log_regression(X_train, class_train):
            model = LogisticRegression(max_iter=2000, random_state=1)
            model.fit(X_train, class_train)
            return model
        filename = training_config.create_filename(model_type) + '.pkl'
        logreg = attempt_cache(conf.model_path, filename, lambda : log_regression(X_train, class_train))
        display(logreg)

        @timing.timer_func
        def predict_classifier(class_reg, holdout_x_scaled):
            forecast_output = class_reg.predict_proba(holdout_x_scaled)
            forecast = holdout_x_scaled.copy(deep=True)
            forecast = pd.concat([forecast, pd.DataFrame(forecast_output, columns=class_reg.classes_, index=forecast.index)], axis=1)
            forecast = mp.classify_and_store(
                    forecast,
                    training_config.classify_confidence_threshold_up,
                    training_config.classify_confidence_threshold_down,
                    'forecast'
                )
            return forecast

        forecast = predict_classifier(logreg, holdout_x_scaled)
        actuals_df = holdout_filled 
        forecast_df = forecast 
        forecast_actuals_logregression_df = mp.join_forecast_to_actuals(
            actuals_df, forecast_df, training_config.forecast_col, training_config.forecast_out, 
            training_config.classify_change_threshold_ratio_up,
            training_config.classify_change_threshold_ratio_down,
            training_config.forecast_root_col)

        # forecast_actuals_logregression_df
        forecast_actuals_logregression_saved = cfg.save_forecast_actuals_regression(
            conf,
            training_config,
            forecast_actuals_logregression_df, 
            model_type, 
            )

        for stock, df in forecast_actuals_logregression_saved.groupby(by='symbol', group_keys=False):
            if stock in conf.tickers_to_highlight:
                mp.draw_forecast_and_actuals(df, stock, training_config.forecast_col, True, model_type=model_type, training_config=training_config)    
            # else:
            #     metrics_df, metrics_string = sklu.calculate_metrics(df, 'forecast_target', 'class_actual' )
            #     logger.info(f"{stock} model_type: {model_type} {metrics_string}")          
        
        def print_training_metrics(df):
            metrics_df, metrics_string = sklu.calculate_metrics(
            df, 'forecast_target', 'class_actual' )
            logger.info(f"global: training start_date: {conf.start_date} holdout start date: {np.min(df.index.get_level_values(conf.date_col))} end_date: {conf.end_date} \n" +
                    f" model_type: {model_type} forecast target: {training_config.forecast_col} forecast_out: {training_config.forecast_out}" +
                    f" forecast_mean_trailing_days: {training_config.forecast_mean_trailing_days} day_offsets: {conf.day_offsets} total_holdout_base:{conf.total_holdout_base} \n" +
                    f" model_configs: {conf.model_configs} num samples: {training_config.num_training_samples} change threshold:{training_config.classify_change_threshold_ratio_up}/{training_config.classify_change_threshold_ratio_down}" +
                    f" confidence threshold: {training_config.classify_confidence_threshold_up}/{training_config.classify_confidence_threshold_down} \n" +
                    f" metrics: {metrics_string}")
            return metrics_df, metrics_string

        metrics_df, metrics_string = print_training_metrics(forecast_actuals_logregression_saved)
        logger.info(f"metrics_string: {metrics_string} metrics_df: {metrics_df.shape}")
        
        return
    
        # display(metrics_df)
        # forecast_actuals_logregression_saved 

        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GridSearchCV

        def tune_hyperparameters(X_train, y_train):
            # Define the parameter grid
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            }

            # Create a Ridge object
            ridge = Ridge()
            # Create the GridSearchCV object
            grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=3)
            # Fit the GridSearchCV object to the data
            grid_search.fit(X_train, y_train)
            # Get the best estimator
            best_model = grid_search.best_estimator_
            return best_model

        # Save the model
        filename = f'LinearRegression_{training_config.forecast_col}_{training_config.num_training_samples}_smpl_{training_config.forecast_out}_day.pkl'
            
        filename = training_config.create_filename(model_type) + '.pkl'
        clfreg = attempt_cache(conf.model_path, filename, lambda : tune_hyperparameters(X_train, y_train))
        display(clfreg)

        holdout_x_scaled = holdout_x_scaled.dropna()
        forecast_output = clfreg.predict(holdout_x_scaled.dropna())
        forecast = holdout_x_scaled.copy(deep=True)
        forecast['forecast'] = forecast_output
        # display(forecast)

        actuals_df = holdout_filled
        forecast_df = forecast 
        forecast_actuals_regression_df = mp.join_forecast_to_actuals(
            actuals_df, forecast_df, training_config.forecast_col, training_config.forecast_out, 
            training_config.classify_change_threshold_ratio_up,
            training_config.classify_change_threshold_ratio_down,
            training_config.forecast_root_col)
        forecast_actuals_regression_saved = cfg.save_forecast_actuals_regression(conf, training_config, forecast_actuals_regression_df, 'LinearRegression')
        logger.info(f"forecast_actuals_regression_saved: {forecast_actuals_regression_saved.shape}")
        metrics_df, metrics_string = print_training_metrics(forecast_actuals_regression_saved)
        logger.info(f"metrics_string: {metrics_string} metrics_df: {metrics_df.shape}")
        # perf_metrics_regression_df = sklu.calculate_metrics(forecast_actuals_regression_df, forecast_col='forecast_target', actual_col=training_config.forecast_col)
        
        # display(forecast_actuals_regression_df)

        importances_regression_df = sklu.get_importance(clfreg.coef_, training_x_scaled)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
                x=importances_regression_df['features'], 
                y=importances_regression_df['coef'], 
                name="regression feature importance",
                connectgaps=True,
                ))
        fig.show()


        model_type = None
        if conf.do_mlp:
            
            model_type = 'MLPClassifier'
            # from sklearn.neural_network import MLPRegressor
            # from sklearn.datasets import make_regression
            # from sklearn.model_selection import train_test_split
            # # >>> X, y = make_regression(n_samples=200, random_state=1)
            # # >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
            # # ...                                                     random_state=1)
            # mlpregr = MLPRegressor(random_state=1,max_iter=500).fit(X_train, y_train)
            # mlpregr

            # # Save the model
            # model_path = 'models'
            # filename = f'MLPRegression_{forecast_col.lower()}_{num_training_samples}_smpl_{forecast_out}_day.pkl'
                
            # mlpregr = attempt_cache(model_path, filename, lambda : MLPRegressor(random_state=1,max_iter=1000).fit(X_train, y_train))
            # display(mlpregr)
            
            from sklearn.neural_network import MLPClassifier

            # mlpregr = MLPClassifier(random_state=1,max_iter=1000).fit(X_train, class_train)
            # mlpregr

            filename = training_config.create_filename(model_type) + '.pkl'
            mlpregr = attempt_cache(conf.model_path, filename, lambda : MLPClassifier(random_state=1,max_iter=2000,early_stopping=True).fit(X_train, class_train))
            display(mlpregr)
            
            forecast = predict_classifier(mlpregr, holdout_x_scaled)
            
            actuals_df = holdout_filled
            forecast_df = forecast
            MLPC_forecast_with_actuals_df = mp.join_forecast_to_actuals(
                actuals_df, forecast_df, training_config.forecast_col, training_config.forecast_out, 
                training_config.classify_change_threshold_ratio_up,
                training_config.classify_change_threshold_ratio_down,
                training_config.forecast_root_col)
            MLPC_forecast_with_actuals__saved_df = cfg.save_forecast_actuals_regression(conf, training_config, MLPC_forecast_with_actuals_df, model_type)

            for stock, df in MLPC_forecast_with_actuals__saved_df.groupby(by='symbol', group_keys=False):
                if stock in conf.tickers_to_highlight:
                    mp.draw_forecast_and_actuals(df, stock, training_config.forecast_col, True, model_type=model_type, training_config=training_config) 
                # else:
                #     metrics_df, metrics_string = sklu.calculate_metrics(df[(df['forecast_target'] != 0) & (df['class_actual'] != 0)], 'forecast_target', 'class_actual' )
                #     logger.info(f"{stock} model_type: {model_type} {metrics_string}")   
            
            metrics_df, metrics_string = print_training_metrics(MLPC_forecast_with_actuals__saved_df)
            logger.info(f"metrics_string: {metrics_string} metrics_df: {metrics_df.shape}")
            # display(metrics_df)
        
            # display(MLPC_forecast_with_actuals__saved_df)

        # Save the model

        if conf.do_poly:
            model_type = 'PolyRegression'
            poly_feature_count = 2
            filename = training_config.create_filename(model_type) + '.pkl'
            clfpolyrgr = attempt_cache(conf.model_path, filename, lambda : make_pipeline(PolynomialFeatures(poly_feature_count), Ridge()).fit(X_train, y_train))
            display(clfpolyrgr)

            confidencereg = clfpolyrgr.score(X_eval, y_eval)
            display(confidencereg)

            forecast_output = clfpolyrgr.predict(holdout_x_scaled.dropna())
            forecast = holdout_x_scaled.copy(deep=True)
            forecast['forecast'] = forecast_output
            # display(forecast)
                    
            actuals_df = holdout_filled
            forecast_df = forecast
            poly_2_forecast_with_actuals_df = mp.join_forecast_to_actuals(
                actuals_df, forecast_df, training_config.forecast_col, training_config.forecast_out, 
                training_config.classify_change_threshold_ratio_up,
                training_config.classify_change_threshold_ratio_down,
                training_config.forecast_root_col)

            for stock, df in poly_2_forecast_with_actuals_df.groupby(by='symbol', group_keys=False):
                if stock in conf.tickers_to_highlight:
                    mp.draw_forecast_and_actuals(df, stock, training_config.forecast_col, model_type=model_type, training_config=training_config)    
            # display(poly_2_forecast_with_actuals_df)
            
            metrics_df, metrics_string = print_training_metrics(poly_2_forecast_with_actuals_df)
            logger.info(f"metrics_string: {metrics_string} metrics_df: {metrics_df.shape}")
            
            # perf_metrics_poly2_df = sklu.calculate_metrics(poly_2_forecast_with_actuals_df, forecast_col='forecast_target', actual_col=training_config.forecast_col)
            # display(perf_metrics_poly2_df)
            
        model_type = None
        rfreg = None
        # do_forest = True

        if conf.do_forest:

            model_type = 'RFClassifier'
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.model_selection import GridSearchCV

            def tune_hyperparameters(X_train, y_train):
                # Define the parameter grid
                param_grid = {
                    'n_estimators':  [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                }

                # Create a RandomForestRegressor object
                rf = RandomForestClassifier(random_state=1)

                # Create the GridSearchCV object
                # grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)

                # Fit the GridSearchCV object to the data
                # grid_search.fit(X_train, y_train)

                # Get the best estimator
                # best_model = grid_search.best_estimator_
                rf.fit(X_train, y_train)

                return rf

            # # Use the function to get the best model
            # rfreg = tune_hyperparameters(X_train, y_train)
            # rfreg

            filename = training_config.create_filename(model_type) + '.pkl'
            rfreg = attempt_cache(conf.model_path, filename, lambda : tune_hyperparameters(X_train, class_train))
            display(rfreg)

        #     confidencereg = rfreg.score(X_eval, y_eval)
        #     confidencereg
            forecast = predict_classifier(rfreg, holdout_x_scaled)
            # display(forecast[forecast['forecast'] != 0])
            

            actuals_df = holdout_filled
            forecast_df = forecast
            RFR_forecast_with_actuals_df = mp.join_forecast_to_actuals(
                actuals_df, forecast_df, training_config.forecast_col, training_config.forecast_out, 
                training_config.classify_change_threshold_ratio_up,
                training_config.classify_change_threshold_ratio_down,
                training_config.forecast_root_col)
            RFR_forecast_with_actuals__saved_df = cfg.save_forecast_actuals_regression(conf, training_config, RFR_forecast_with_actuals_df, model_type)

            for stock, df in RFR_forecast_with_actuals__saved_df.groupby(by='symbol', group_keys=False):
                if stock in conf.tickers_to_highlight:
                    mp.draw_forecast_and_actuals(df, stock, training_config.forecast_col, True, model_type=model_type, training_config=training_config)    
                    
            metrics_df, metrics_string = sklu.calculate_metrics(
                RFR_forecast_with_actuals__saved_df[
                (RFR_forecast_with_actuals__saved_df['forecast_target'] != 0) & (RFR_forecast_with_actuals__saved_df['class_actual'] != 0)], 'forecast_target', 'class_actual' )
            logger.info(f"global model_type: {model_type} forecast target: {training_config.forecast_col} forecast_out: {training_config.forecast_out} forecast_mean_trailing_days: {training_config.forecast_mean_trailing_days}" +
                        f"num samples: {training_config.num_training_samples} change threshold:{training_config.classify_change_threshold_ratio_up}/{training_config.classify_change_threshold_ratio_down}" +
                        f"confidence threshold: {training_config.classify_confidence_threshold_up}/{training_config.classify_confidence_threshold_down}" +
                        f" metrics: {metrics_string}")
            # display(RFR_forecast_with_actuals__saved_df)
            metrics_df, metrics_string = print_training_metrics(RFR_forecast_with_actuals__saved_df)
            logger.info(f"metrics_string: {metrics_string} metrics_df: {metrics_df.shape}")

            # perf_metrics_rgr_df = sklu.calculate_metrics(RFR_forecast_with_actuals_df, forecast_col='forecast_target', actual_col=training_config.forecast_col)
            # display(perf_metrics_rgr_df)
            
            # mp.
            importances_forest_df = sklu.get_importance(rfreg.feature_importances_, training_x_scaled)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                    x=importances_forest_df['features'], 
                    y=importances_forest_df['coef'], 
                    name="forest feature importance",
                    connectgaps=True,
                    ))
            fig.show()



