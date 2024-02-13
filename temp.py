import pickle
import pandas as pd

with open(f"processed_dataset/features_amp.pkl", "rb") as f:
    df = pickle.load(f)