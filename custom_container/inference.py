import os
import json
import joblib
import torch
import xgboost as xgb 
import pickle 
import numpy as np 
from sklearn.base import BaseEstimator 

def model_fn(model_dir):
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    

    if not os.path.exists(metadata_path) as f:
        raise FileNotFoundError("Missing model_metadata.json")
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    framework = metadata["model"]
    if framework == "sklearn":
        model = pickle.load(os.path.join(model_dir, "model.pkl"))
    elif framework == "xgboost":
        model = pickle.load(os.path.join(model_dir, "model.pkl"))
    elif framework == "pytorch":
        model = torch.load(os.path.join(model_dir, "model.pkl"), map_location = torch.device("cpu"))
        model.eval()
    else:
        raise ValueError(f"Unsupported Framework: {framework}")
    
    return {"model": model, "framework": framework}

def input_fn(request_body, content_type = "text/csv"):
    if content_type == "text/csv":
        data = np.array([list(map(float, row.split(","))) for row in request_body.strip().split("\n")])
        return data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
    
def predict_fn(input_data, model_bundle):
    model = model_bundle["model"]
    framework = model_bundle["framework"]

    if framework == "sklearn":
        return model.predict(input_data)
    elif framework == "xgboost":
        return model.predict(input_data)
    elif framework == "pytorch":
        tensor = torch.tensor(input_data, dtype=torch.float32)
        with torch.no_grad():
            return model(tensor).numpy()
    else:
        raise ValueError("Unsupported model in prediction")
    
def output_fn(prediction, content_type="text/csv"):
    if content_type = "text/csv":
        return "\"n".join(map(str, prediction.tolist()))
    else:
        raise ValueError(f"Unsupported content_type: {content_type}")