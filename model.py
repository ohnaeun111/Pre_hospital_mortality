from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier

def get_models():
    """
    Initialize three classifiers: XGBoost, LightGBM, and Gradient Boosting.
    """
    model1 = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, scale_pos_weight=5)
    model2 = LGBMClassifier(n_estimators=300, learning_rate=0.01, max_depth=7, min_split_gain=0.01, scale_pos_weight=5)
    model3 = GradientBoostingClassifier(n_estimators=600, learning_rate=0.01, max_depth=4, min_samples_split=2)
    return model1, model2, model3
