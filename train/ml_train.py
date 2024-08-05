from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.all import ExponentialSmoothing,AutoETS,all_estimators,BATS,PolynomialTrendForecaster,VAR,CINNForecaster
from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
from sktime.regression.kernel_based import RocketRegressor
from sktime.regression.interval_based import TimeSeriesForestRegressor
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.regression.deep_learning.resnet import ResNetRegressor
from sktime.datasets import load_airline
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.performance_metrics.forecasting import (mean_absolute_error,
mean_absolute_percentage_error,mean_squared_error,
mean_squared_percentage_error)
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.performance_metrics.forecasting import (MeanSquaredError,
                                                    MedianSquaredScaledError,
                                                     MeanAbsoluteError,
                                                    MeanSquaredPercentageError,
                                                    MeanAbsolutePercentageError)
from sktime.regression.deep_learning.inceptiontime import InceptionTimeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.naive import NaiveForecaster
from sktime.split import ExpandingWindowSplitter,SlidingWindowSplitter
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.model_evaluation import evaluate
from sktime.utils.plotting import plot_series
from sktime.utils import mlflow_sktime
import numpy as np
import mlflow
import optuna
import pandas as pd
from utils.registrys import TransformerRegistry,ModelRegistry
from utils.gen_data import generate_synthetic_panel_data
from configs import configs





GLOBAL_TRACK=[]

with mlflow.start_run():

    def objective(trial):


        data = generate_synthetic_panel_data()
        results = []
        transformer, transformer_name, transformer_params = TransformerRegistry.get_transformer(trial)
        model, model_name, model_params = ModelRegistry.get_model(trial)

        params={
            'transformer': transformer_name,
            'model': model_name,
            **transformer_params,
            **model_params
        }
        with mlflow.start_run(nested=True):
            for key,val in params.items():
                mlflow.log_param(key,val)


        for ctr in data.country.drop_duplicates().tolist():
            country_df=data.query("country==@ctr")

            y = country_df['default_rate']
            X = country_df.drop(columns=['default_rate', 'country'])



        # Build the pipeline
            if 'reduce' in model_name:
                forecaster = TransformedTargetForecaster(steps=[
                    ('transformer', transformer),
                    ('forecaster', make_reduction(model, window_length=configs.WINDOW_LENGTH_REDUCTION, strategy='recursive'))
                ])
            else:
                forecaster = TransformedTargetForecaster(steps=[
                    ('transformer', transformer),
                    ('forecaster', model)
                ])


            cv = SlidingWindowSplitter(window_length=y.shape[0]//5, fh=[i + 1 for i in range(10)],step_length=30)


            # Cross-validation




            for k, (train_idx,test_idx) in enumerate(cv.split(y)):
                y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
                x_train_cv, x_test_cv = X.iloc[train_idx], X.iloc[test_idx]
                # Fit the model

                forecaster.fit(y_train_cv,X= x_train_cv)
                ##
                y_pred = forecaster.predict(fh=[i + 1 for i in range(10)])

                # Calculate metrics
                metrics = {}
                metrics['Fold'] = k
                metrics['MeanSquaredError'] = mean_squared_error(y_test_cv, y_pred)
                metrics['MeanAbsoluteError'] = mean_absolute_error(y_test_cv, y_pred)
                metrics['MeanSquaredPercentageError'] = MeanSquaredPercentageError()(y_test_cv, y_pred)
                metrics['MeanAbsolutePercentageError'] = MeanAbsolutePercentageError()(y_test_cv, y_pred)
                metrics['MedianSquaredScaledError'] = MedianSquaredScaledError()(y_test_cv, y_pred, y_train=y_train_cv)
                metrics['Country']=ctr
                metrics['model']=model_name
                metrics['transformer'] = transformer_name
                metrics['model_params']=str(model_params)
                metrics['transformer_params'] = str(transformer_params)
                results.append(pd.DataFrame(metrics,index=[0]))
        cmb = pd.concat(results)
        GLOBAL_TRACK.append(cmb)

        with mlflow.start_run(nested=True):
            mlflow.log_metric('MeanSquaredError', cmb['MeanSquaredError'].mean())
            mlflow.log_metric('MeanAbsoluteError', cmb['MeanAbsoluteError'].mean())
            mlflow.log_metric('MeanSquaredPercentageError', cmb['MeanSquaredPercentageError'].mean())
            mlflow.log_metric('MeanAbsolutePercentageError', cmb['MeanAbsolutePercentageError'].mean())
            mlflow.log_metric('MedianSquaredScaledError', cmb['MedianSquaredScaledError'].mean())


        return cmb['MeanSquaredError'].mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Optimize hyperparameters with Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)  # Adjust the number of trials

    optuna.logging.set_verbosity(optuna.logging.WARNING)


