import logging
#import luigi
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# this import should go after matplotlib
from skopt import gp_minimize, dump

logger = logging.getLogger(__name__)

HYPERPARAMETERS = [
    {
        'name': 'max_depth',
        'range': (4, 30, 'uniform'),
        'value': 10,
        'integer': True
    },
    {
        'name': 'num_leaves',
        'range': (30, 60, 'uniform'),
        'value': 40,
        'integer': True
    },
    {
        'name': 'learning_rate',
        'range': (.01, .8, 'log-uniform'),
        'value': 0.45
    },
    {
        'name': 'n_estimators',
        'range': (100, 300, 'uniform'),
        'value': 150,
        'integer': True
    },
    {
        'name': 'min_split_gain',
        'range': (0, .5, 'uniform'),
        'value': 0.4
    },
    {
        'name': 'subsample',
        'range': (.5, 1, 'uniform'),
        'value': 0.6
    },
    {
        'name': 'colsample_bytree',
        'range': (.5, 1, 'uniform'),
        'value': 0.8
    },
]


class OptimizeHyperparams():

    def run(self):
        
        dataset_encoded = pd.read_csv('ops_drg.csv')
        Y = dataset_encoded['entgeltbetrag'].values
        X = dataset_encoded.drop(['LoS', 'entgeltbetrag', 'drg'], axis=1).values
        
        train_x, test_x, train_y, test_y = train_test_split(X, Y,
                                                                    test_size=0.2, random_state=7)
        
        model = lightgbm.LGBMRegressor()

        param_names = [p['name'] for p in HYPERPARAMETERS]
        integer_params = [p['name'] for p in HYPERPARAMETERS if p.get('integer', False)]

        counter = 0

        def hyperparams_objective(params):
            nonlocal counter
            counter += 1

            start_time = time.time()

            params_dict = dict(zip(param_names, params))
            for p in integer_params:
                params_dict[p] = int(params_dict[p])
                
            model.set_params(**params_dict)
            split = ShuffleSplit(n_splits=5, train_size=0.9)
            score = cross_val_score(model, train_x, train_y, cv=split,
                                    scoring="neg_mean_squared_error")

            cv_time = time.time() - start_time
            logger.info(f'ITERATION: {counter}, time: {cv_time}, score: {score}')
            logger.info(f'params: {params_dict}')

            return -score.mean()

        logger.info('hyperparameter optimization started')

        search_space = [p['range'] for p in HYPERPARAMETERS]
        x0 = [p['value'] for p in HYPERPARAMETERS]

        res_gp = gp_minimize(
            hyperparams_objective, search_space,
            x0=x0, n_calls=100)
        print(res_gp)
        dump(res_gp, 'res_dump', store_objective=False)
        
        optimal_params = dict(zip(param_names, res_gp.x))
        for p in integer_params:
                optimal_params[p] = int(optimal_params[p])
        optimal_model = lightgbm.LGBMRegressor()
        optimal_model.set_params(**optimal_params)
        optimal_model.fit(train_x, train_y)
        mae = mean_absolute_error(test_y, optimal_model.predict(test_x))
        
        with open('result.txt', 'w') as out_file:
            print('params', res_gp.x, 'cv_score', res_gp.fun, 'mae', mae,
            file=out_file) 

if __name__ == '__main__':
    task = OptimizeHyperparams()
    task.run()