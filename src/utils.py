import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]

            # Train the model
            model.fit(x_train, y_train)

            # Predicting the test set results
            y_pred = model.predict(x_test)

            # Evaluating the model
            r2_square = r2_score(y_test, y_pred)

            report[model_name] = r2_square

        return report

    except Exception as e:
        raise CustomException(e, sys)    