import os
import sys
import pandas as pd
from src.exception import custom_exception
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
@dataclass 
class Dataingestionconfig:
    train_data_path:str=os.path.join('articraft','train.csv')
    test_data_path:str=(os.path.join('articraft','test.csv'))
    raw_data_path:str=(os.path.join('articraft','data.csv'))
class dataingestion:
    def __init__(self):
        self.ingestion_config=Dataingestionconfig()
    def initiate_data_ingestion(self):
        logging.info("entered the data ingestion method or components")
        try:
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info("read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('train and test split initaiated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("ingestion of the data has completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise custom_exception(e,sys)
if __name__=="__main__":
    obj=dataingestion()
    obj.initiate_data_ingestion()