import sys
import os
from src.utils import save_obj
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import custom_exception
from src.logger import logging
from sklearn.preprocessing import OneHotEncoder,StandardScaler
@dataclass
class Datatransformationconfig:
    preprocessor_obj_file_path=os.path.join('articraft','processor.pkl')
class Data_transformation:
    def __init__(self):
        self.data_transformation_config=Datatransformationconfig()
    def get_data_transform(self):
        try:
            numerical_column=['writing_score','reading_score']
            categorical_columns=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course',
            ]
            num_pipeline=Pipeline(
                steps=[
                
                    ('impute',SimpleImputer(strategy='median')),
                    ('scalar',StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder',OneHotEncoder())
                ]
            )
            logging.info(f'categorical columns:{categorical_columns}')
            logging.info(f"numerical data:{numerical_column}")
            preprocessor=ColumnTransformer(
                transformers=[
                ('num_pipelinne',num_pipeline,numerical_column),
                ('cat_pipeline',cat_pipeline,categorical_columns)
            ])
        
            return preprocessor
        except Exception as e:
            raise e
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('train adn test data initiated')
            logging.info('obtaining preprocessing object')
            preprocessor_obj=self.get_data_transform()
            target_column='math_score'
            numerical_column=['writing_score','reading_score']
            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_featurea_train_df=train_df[target_column]
            input_featurer_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]
            logging.info('applying preprocessing object on training data frame and test data frame')
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_featurer_test_df)
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_featurea_train_df)

            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info('save preproccessing object')
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise custom_exception(e,sys)
            
