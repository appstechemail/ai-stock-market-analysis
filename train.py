
import pandas as pd

def get_model_probabilities(models, X_test, X_test_scaled):

    lr_model = models["lr"]
    et_model = models["et"]
    rf_model = models["rf"]
    xgb_model = models["xgb"]
    lgb_model = models["lgb"]
    cat_model = models["cat"]
    mlp_model = models["mlp"]

    lr_proba = lr_model.predict_proba(X_test_scaled)[:,1]

    et_proba = pd.Series(
        et_model.predict_proba(X_test)[:,1],
        index=X_test.index
    )

    rf_proba = pd.Series(
        rf_model.predict_proba(X_test_scaled)[:,1],
        index=X_test.index
    )

    xgb_proba = pd.Series(
        xgb_model.predict_proba(X_test)[:,1],
        index=X_test.index
    )

    lgb_proba = lgb_model.predict_proba(X_test)[:,1]
    cat_proba = cat_model.predict_proba(X_test)[:,1]

    mlp_proba = pd.Series(
        mlp_model.predict_proba(X_test_scaled)[:,1],
        index=X_test.index
    )

    return {
        "lr": lr_proba,
        "et": et_proba,
        "rf": rf_proba,
        "xgb": xgb_proba,
        "lgb": lgb_proba,
        "cat": cat_proba,
        "mlp": mlp_proba
    }


# =====================================================
# ############# GET PREDICTIONS #######################
# =====================================================

def get_predictions(models, X_test, X_test_scaled, ensemble_proba, threshold):

    return {
        "lr": models["lr"].predict(X_test_scaled),
        "et": models["et"].predict(X_test),
        "rf": models["rf"].predict(X_test_scaled),
        "xgb": models["xgb"].predict(X_test),
        "lgb": models["lgb"].predict(X_test),
        "cat": models["cat"].predict(X_test),
        "mlp": models["mlp"].predict(X_test_scaled),
        "ensemble": (ensemble_proba > threshold).astype(int)
    }

