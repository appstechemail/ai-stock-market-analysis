# =========================
# IMPORT LIBRARIES
# =========================
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# =========================
# TRAIN MODELS
# =========================
def train_models(X_train, y_train, X_train_scaled):

    models = {}

    # Logistic Regression
    lr_model = LogisticRegression(
        C=0.5,
        max_iter=1000,
        class_weight="balanced"
    )
    lr_model.fit(X_train_scaled, y_train)
    models["lr"] = lr_model

    # Extra Trees
    et_model = ExtraTreesClassifier(n_estimators=200, random_state=42)
    et_model.fit(X_train, y_train)
    models["et"] = et_model

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        class_weight="balanced"
    )
    rf_model.fit(X_train_scaled, y_train)
    models["rf"] = rf_model

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    models["xgb"] = xgb_model

    # LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    lgb_model.fit(X_train, y_train)
    models["lgb"] = lgb_model

    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        verbose=0,
        random_seed=42
    )
    cat_model.fit(X_train, y_train)
    models["cat"] = cat_model

    # Neural Network
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        alpha=0.001,
        max_iter=500,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    models["mlp"] = mlp

    return models
