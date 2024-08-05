from sktime.transformations.series.detrend import Detrender
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.forecasting.arima import AutoARIMA
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
import optuna
from optuna.trial import Trial

# Transformer Registry with hyperparameters
class TransformerRegistry:
    transformers = {
        'detrender': (Detrender,{}),
        'exponent': (ExponentTransformer,{'power':optuna.distributions.FloatDistribution( 0.1, 2.0)}),
        'boxcox': (BoxCoxTransformer,{'lambda_fixed':optuna.distributions.FloatDistribution( -2, 2)})
    }


    @staticmethod
    def get_transformer(trial: Trial):
        name = trial.suggest_categorical('transformer', list(TransformerRegistry.transformers.keys()))
        params_set = TransformerRegistry.transformers[name][1]
        params={}
        for param_name, param_space in params_set.items():
            if isinstance(param_space, list):
                params[param_name] = trial.suggest_categorical(param_name, param_space)
            else:
                params[param_name] = trial._suggest(param_name, param_space)
        return TransformerRegistry.transformers[name][0](**params), name, params


# Model Registry with hyperparameters
class ModelRegistry:
    models = {
        "NaiveForecaster": (NaiveForecaster, {
            "strategy": ["last", "mean"],
            "window_length": optuna.distributions.IntUniformDistribution(10, 50)
        })}

    @staticmethod
    def get_model(trial: Trial):
        name = trial.suggest_categorical('model', list(ModelRegistry.models.keys()))
        params_set=ModelRegistry.models[name][1]
        params={}
        for param_name, param_space in params_set.items():
            if isinstance(param_space, list):
                params[param_name] = trial.suggest_categorical(param_name, param_space)
            else:
                params[param_name] = trial._suggest(param_name, param_space)

        return ModelRegistry.models[name][0](**params), name, params