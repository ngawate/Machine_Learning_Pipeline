import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation Started')

            numerical_col = ['carat', 'depth', 'table', 'x', 'y', 'z']
            cathegorical_col = ['cut', 'color', 'clarity']
            
            #['Fair: 1']
            cut_cath = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            #['I1':1 ,'SI2':2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8]
            clarity_cath = ['I1' ,'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
            #'J':1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7
            color_cath = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
            logging.info('Pipeline initiated')

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cath_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder', OrdinalEncoder(categories=[cut_cath, color_cath, clarity_cath])),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_col),
                    ('cath_pipeline', cath_pipeline, cathegorical_col)
                ]
            )

            return preprocessor
            logging.info('Pipeline Successfully Completed')

        except Exception as e:
            logging.info('Error in Data Transformation -->', e)
            raise CustomException(e, sys)
        
    def initiate_data_tranformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read Test and Train Data')
            logging.info(f'Train head: \n{train_df.head().to_string()}')
            logging.info(f'Test head: \n{test_df.head().to_string()}')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_columns = [target_column_name, 'id']

            #X_train, X_test, y_train, y_test
            X_train = train_df.drop(columns=drop_columns, axis=1)
            y_train = train_df[target_column_name]
            X_test = test_df.drop(columns=drop_columns, axis=1)
            y_test = test_df[target_column_name]

            #Data Trasformation
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            logging.info('Applying preproccessing object on training and testing dataset')

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle file saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
  
        except Exception as e:
            logging.info('Error occured in initiate_data_tranformation -->', e)
            raise CustomException(e, sys)