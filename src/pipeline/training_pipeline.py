import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=="__main__":
    obj= DataIngestion()
    t1,t2= obj.initiate_data_ingestion()
    obj1= DataTransformation()
  
    train,test,pre= obj1.initaite_data_transformation(t1,t2)
    obj2= ModelTrainer()
    obj2.model_building(train,test,pre)