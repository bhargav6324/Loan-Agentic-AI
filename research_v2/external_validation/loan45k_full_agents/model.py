"""Frozen 45K model feature contract and trainer."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ["person_gender", "person_education", "person_home_ownership",
               "loan_intent", "previous_loan_defaults_on_file"]
NUMERIC = ["person_age", "person_income", "person_emp_exp", "loan_amnt",
           "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
           "credit_score"]
FEATURES = CATEGORICAL + NUMERIC


def fit_model(train, seed: int):
    model = Pipeline([
        ("prepare", ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", "passthrough", NUMERIC),
        ])),
        ("model", RandomForestClassifier(n_estimators=100, random_state=seed,
                                         n_jobs=-1, class_weight="balanced")),
    ])
    model.fit(train[FEATURES], train["loan_status"])
    return model
