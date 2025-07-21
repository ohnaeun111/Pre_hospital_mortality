import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Set dataset path (Modify this if necessary)
DATA_PATH = os.getenv('DATASET_PATH', 'your/private/dataset/path')

def load_data():
    """
    Load dataset from a private source.
    Modify the file path as needed.
    """
    df = pd.read_table(os.path.join(DATA_PATH, 'data.txt'), sep=',', low_memory=False)

    # Drop unnecessary columns
    X_features = df.drop(['survive'], axis=1).values
    y_label = df['survive'].values
    return X_features, y_label

def preprocess_data(X, y):
    """
    Split data into training and testing sets.
    """
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=77)
