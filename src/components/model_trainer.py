import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model
from xgboost import XGBClassifier
from dataclasses import dataclass
import sys
import os
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


@dataclass
class ModelTrainerConfig:
    model_file_path= os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def model_building(self,train_arr,test_arr,preprocessor_obj_file):

        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train = train_arr[:,:-1]
            Y_train = train_arr[:,-1]
            X_test  = test_arr[:,:-1]
            Y_test  = test_arr[:,-1]
            models={
                'xgb_clf' : XGBClassifier()
                }
            
            report = evaluate_model(models,X_train,Y_train,X_test,Y_test)
            logging.info(f'Model Report : {report}')
            print(report.keys())
            print(report.values())
            best_model_score= max(sorted(report.values()))
            print(best_model_score)

            best_model_name= list(report.keys())[list(report.values()).index(best_model_score)]
            best_model= models[best_model_name]

            
            print(f'Best Model Found , Model Name : {best_model_name} ,  Accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')


            save_object(
                file_path=self.model_trainer_config.model_file_path,
                obj= best_model
            )


        except Exception as e:
            logging.info("model creation is having some error")
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj= DataIngestion()
    t1,t2= obj.initiate_data_ingestion()
    obj1= DataTransformation()
  
    train,test,pre= obj1.initaite_data_transformation(t1,t2)
    obj2= ModelTrainer()
    obj2.model_building(train,test,pre)