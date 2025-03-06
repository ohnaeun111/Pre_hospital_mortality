from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

def get_models():
    """
    Initialize three classifiers: XGBoost, LightGBM, and Random Forest
    """
    model1 = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4, scale_pos_weight=12, max_delta_step=1, min_child_weight=1)
    model2 = LGBMClassifier(n_estimators=50, learning_rate=0.1, max_depth=-1, min_split_gain=0.01, scale_pos_weight=11, reg_lambda=1, min_child_weight=1)
    model3 = RandomForestClassifier(n_estimators=50, max_depth=9, min_samples_split=3, min_samples_leaf=4, max_features='sqrt', class_weight={0: 0.34, 1: 0.66}, max_leaf_nodes=40, random_state=77)
    
    return model1, model2, model3
