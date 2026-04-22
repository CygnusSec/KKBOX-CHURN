from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from config import TARGET, TEST_SIZE, RANDOM_STATE

def train_model(df):
    X = df.drop([TARGET, 'msno'], axis=1)
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model.fit(X_train, y_train)

    return model, X_val, y_val