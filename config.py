import dateparser
import pandas as pd
import json
import datetime
import logging
import os
import pickle

# if run from notebook
from IPython.display import display

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not os.path.exists('logs'):
    os.makedirs('logs')
filename = f"logs/{__name__}_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log"
if not os.path.exists(filename):
    with open(filename, 'wt') as f:
        f.write(f'*** {filename} ***\n')
    
class TrainingConfig():
    
    def __init__(self, parent_config, **entries):
        
        self.__dict__.update(entries)
        self.parent_config = parent_config

        # Access the parameters
        self.forecast_col = self['forecast_col'].format(
            forecast_root_col=self['forecast_root_col'], 
            forecast_mean_trailing_days=self['forecast_mean_trailing_days']
            ) 
    
    def __getitem__(self, name):
        return self.__dict__.get(name)
    
    def __setitem__(self, name, value):
        self.__dict__[name] = value
    
    def create_filename(self, model_type: str):
        
        filename = (
            f'{model_type}'
            f'_{self.forecast_col}'
            f'_{self.num_training_samples}_smpl'
            f'_{self.forecast_out}_day'
            f'_{self.classify_confidence_threshold_up}_confUp'
            f'_{self.classify_confidence_threshold_down}_confDown'
            f'_{self.classify_change_threshold_ratio_up}_changeUp'
            f'_{self.classify_change_threshold_ratio_down}_changeDown'
        )
        
        translation_table = str.maketrans(' /.', '___')
        return filename.translate(translation_table)
    
    def add_config_columns(self, df):
        
        new_df = df.copy()
        new_df['actual'] = self[self.forecast_col]
        new_df['conf_thrs'] = f'{self.classify_confidence_threshold_up}_{self.classify_confidence_threshold_down}'.replace('.', '_')
        new_df['chng_thrs'] = f'{self.classify_change_threshold_ratio_up}_{self.classify_change_threshold_ratio_down}'.replace('.', '_')
        
        return new_df
    
class Config:
    
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
        assert self['model_configs'], "model_configs is not defined in the config file"
        assert type(self['model_configs'] == list), "model_configs should be a list of model configurations"
        
        new_configs = []
        largest_forecast_out = 0
        for model_config in self['model_configs']:
            new_config = self.load_training_config(model_config)
            new_configs.append(new_config)
            if largest_forecast_out < new_config['forecast_out']:
                largest_forecast_out = new_config['forecast_out']    
        self.training_configs = new_configs
        self.largest_forecast_out = largest_forecast_out
        
        if self.data_path_suffix is not None:
            self.data_path = os.path.join(self.data_path, dateparser.parse(self.data_path_suffix).strftime('%Y-%m-%d'))
            self.feature_path = os.path.join(self.feature_path, dateparser.parse(self.data_path_suffix).strftime('%Y-%m-%d'))
            self.model_path = os.path.join(self.model_path, dateparser.parse(self.data_path_suffix).strftime('%Y-%m-%d'))
            self.results_path = os.path.join(self.results_path, dateparser.parse(self.data_path_suffix).strftime('%Y-%m-%d'))
            
        
        self.start_date = dateparser.parse(self.start_date)
        assert self.start_date, f"start_date is not defined correctly: {self.start_date} in the config file: {self.config_path}"
        self.end_date = dateparser.parse(self.end_date)
        assert self.end_date, f"end_date is not defined correctly: {self.end_date} in the config file: {self.config_path}"
        self.largest_day_offset = self.day_offsets[-1]
        self.start_date_extended = self.start_date + datetime.timedelta(-self.largest_day_offset)
        self.end_date_extended = self.end_date #+ datetime.timedelta(days=self.largest_forecast_out)

    def __getitem__(self, name):
        return self.__dict__.get(name)
    
    def __setitem__(self, name, value):
        self.__dict__[name] = value
    
    def load_training_config(self, file_path: str):
        full_path = os.path.join(self.config_path, file_path)
        with open(full_path) as f:
            return TrainingConfig(self, **json.load(f))
    

def load_config(config_path, json_file):
    
    full_file_path = os.path.join(config_path, json_file)
    # Load the config file
    with open(full_file_path) as f:
             
        logger.info(f"loading config at path {full_file_path}")
        config_data = json.load(f)
        config_data['config_path'] = config_path
        # Return an object of the Config class
        return Config(**config_data)
    
def load_results_pickle_file_class(config: Config, training_config: TrainingConfig, model_type: str):
    
    filename = training_config.create_filename(model_type) + ".pkl"
    pickle_file_name = os.path.join(config.results_path, filename)
    logger.info(f"Loading pickle file: {pickle_file_name}")
    with open(pickle_file_name, 'rb') as file:
        df = pickle.load(file)
        return df
    

def save_forecast_actuals_regression(config: Config, training_config: TrainingConfig, df: pd.DataFrame, model_type: str):
   
    logger.info(f"Saving forecast actuals for regression model")
    # display(df[df['forecast_leading'] > 0])
    # logger.info(f"^^^ THAT WAS THE DATAFRAME ^^^")
    if not os.path.exists(config.results_path):
        os.makedirs(config.results_path)
    df = training_config.add_config_columns(df).set_index(['symbol', 'date'])
    prior_columns = df.columns
    filename = training_config.create_filename(model_type) + ".pkl"
    pickle_file_name = os.path.join(config.results_path, filename)
    logger.info(f"Saving data to pickle file: {pickle_file_name}")
    with open(pickle_file_name, 'wb') as file:
        pickle.dump(df, file)
        
    # with open(pickle_file_name, 'rb') as file:
    #     df = pickle.load(file)
    #     post_columns = df.columns
    #     logger.info(f"Post-save load forecast actuals for regression model {pickle_file_name}")
    #     display(df[df['forecast_leading'] > 0])
    #     logger.info(f"^^^ THAT WAS THE OTHER DATAFRAME ^^^")
        
    # logger.info(f"Saving data to csv file: {os.path.join(config.results_path, training_config.create_filename(model_type) + '.csv')}")
    # df.to_csv(os.path.join(config.results_path, training_config.create_filename(model_type) + ".csv"))
    # df = pd.read_csv(os.path.join(config.results_path, training_config.create_filename(model_type) + ".csv"))
    # display(df[df['forecast_leading'] > 0])
    # logger.info(f"^^^ THAT WAS THE LAST DATAFRAME ^^^")
    # logger.info(f"Columns before and after save: {prior_columns} {post_columns} difference: {set(prior_columns) - set(post_columns)}")
    return df
    
