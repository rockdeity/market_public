import datetime
import logging
import os
from time import time 

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

def timer_func(func): 
    # This function shows the execution time of  
    # the function object passed 
    def wrap_func(*args, **kwargs): 
        t1 = time() 
        result = func(*args, **kwargs) 
        t2 = time() 
        logger.debug(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s') 
        return result 
    return wrap_func 