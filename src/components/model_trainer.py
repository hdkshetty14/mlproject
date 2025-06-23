import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test data.")
            
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1])
            
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGB": XGBRegressor(),
                "CatBoost Regression": CatBoostRegressor(verbose=False),
                "AdaBoost Regression": AdaBoostRegressor()
            }
            
            params={
                "Ridge Regression": {'alpha':[5,10,15,20]},
                
                "K Neighbors Regressor": {
                    'n_neighbors':[5,10,15,20,30], 
                    'algorithm': ["auto", "ball_tree", "kd_tree", "brute"],
                    'p': [1,2,3]},
                
                "Decision Tree": {
                    'criterion': ["squared_error", "friedman_mse", "absolute_error", "poison"],
                    'splitter': ['best', 'random'],
                    'max_depth': [1,2,3,4,5,10,15,20,25],
                    'max_features': ["auto", "sqrt", "log2"]},
                
                "Random Forest": {
                    'n_estimators': [100,150,200,250,300],
                    'criterion': ["squared_error", "absolute_error", "friedman_mse", "poison"],
                    'max_depth': [None,1,2,3,4,5,10,15,20,25],
                    'min_samples_split': [2,5,10,15,20,25],
                    'min_samples_leaf': [2,5,10,15,20,25],
                    'max_features': ['auto', 1,2,5,10,15,20,25]},
                
                "Gradient Boosting": {
                    'loss': ["squared_error", "absolute_error", "huber", "quantile"],
                    'learning_rate': [0.1, 0.5, 0.7, 1],
                    'n_estimators': [100,150,200,250,300],
                    'criterion': ["squared_error", "friedman_mse"],
                    'max_depth': [None,1,2,3,4,5,10,15,20,25],
                    'min_samples_split': [2,5,10,15,20,25],
                    'min_samples_leaf': [2,5,10,15,20,25],
                    'max_features': ['auto', 1,2,5,10,15,20,25]},
                
                "AdaBoost Regression": {
                    'loss': ["linear", "square", "exponential"],
                    'learning_rate': [0.1, 0.5, 0.7, 1],
                    'n_estimators': [100,150,200,250,300]},
                
                "XGB": {
                    'learning_rate': [0.1, 0.5, 0.7, 1],
                    'max_depth': [None,1,2,3,4,5,10,15,20,25]}}
            
            model_report:dict=evaluate_models(X_train=X_train, 
                                             y_train=y_train, 
                                             X_test=X_test,
                                             y_test=y_test,
                                             models=models,
                                             param=params)
            
            best_model_score=max(sorted(model_report.values()))
            
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found.")
            
            logging.info("Best found model on training and testing dataset.")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )    
            
            predicted=best_model.predict(X_test)    
            
            r2=r2_score(y_test, predicted)
            return r2
        
        except Exception as e:
            raise CustomException(e, sys)

