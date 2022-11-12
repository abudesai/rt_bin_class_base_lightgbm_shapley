
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 


import lightgbm as lgb

model_fname = "model_params.save"
MODEL_NAME = "bin_class_base_lightgbm_shapley"

class Classifier(): 
    
    def __init__(self, boosting_type="gbdt", n_estimators = 250, num_leaves=31, learning_rate=1e-1, **kwargs) -> None:
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.model = self.build_model(**kwargs)     
        
        
    def build_model(self, **kwargs): 
        model = lgb.LGBMClassifier(            
            num_class =1,
            boosting_type = self.boosting_type,
            num_leaves = self.num_leaves,
            learning_rate = self.learning_rate,
            n_estimators = self.n_estimators,
            **kwargs, 
            random_state=42
        )
        return model
    
    
    def fit(self, train_X, train_y):        
        self.model.fit(train_X, train_y, verbose=20)        
        
    
    def predict(self, X): 
        preds = self.model.predict(X)
        return preds 
    
    
    def predict_proba(self, X): 
        preds = self.model.predict_proba(X)
        return preds 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)   
    
    
    def save(self, model_path): 
        joblib.dump(self.model, os.path.join(model_path, model_fname))
        


    @classmethod
    def load(cls, model_path):         
        classifier = joblib.load(os.path.join(model_path, model_fname))
        # print("where the load function is getting the model from: "+ os.path.join(model_path, model_fname))        
        return classifier


def save_model(model, model_path):
    # print(os.path.join(model_path, model_fname))
    joblib.dump(model, os.path.join(model_path, model_fname)) #this one works
    # print("where the save_model function is saving the model to: " + os.path.join(model_path, model_fname))
    

def load_model(model_path): 
    model = joblib.load(os.path.join(model_path, model_fname))   
    try: 
        model = joblib.load(os.path.join(model_path, model_fname))   
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model
