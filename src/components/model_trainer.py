import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_object, evaluate_model
import os
import sys

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info('Spliting dependent and independent variable')
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:, -1]
                )
            
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet()
                }
            
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)

            logging.info(f'Model Report: {model_report}')

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info(f'Best Model Name: {best_model}')
            logging.info(f'Best Model Score: {best_model_score}')

            save_object(file_path= self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            
        except Exception as e:
            logging.info('Error happened while Model Training')
            raise CustomException(e, sys)