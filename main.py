from src.load_data import load_data
from src.preprocess import clean_members, clean_transactions
from src.feature_engineering import (
    create_transaction_features,
    merge_all,
    encode_features,
    create_log_features
)
from src.model import train_model
from src.evaluate import evaluate
from src.logs_processing import process_logs


def main():
    # 1. Load data
    train, members, transactions = load_data()

    # 2. Preprocess
    members = clean_members(members)
    transactions = clean_transactions(transactions)

    # 3. Transaction features
    trans_feat = create_transaction_features(transactions)

    # 4. Merge base data (TẠO df Ở ĐÂY)
    df = merge_all(train, members, trans_feat)

    # 5. Process logs (nặng → để sau)
    logs_feat = process_logs("../data/user_logs_v2.csv")

    # 6. Feature engineering logs
    logs_feat = create_log_features(logs_feat)

    # 7. Merge logs vào df
    df = df.merge(logs_feat, on='msno', how='left')

    # 8. Encode categorical
    df = encode_features(df)

    # 9. Fill missing
    df = df.fillna(0)

    # 10. Train model
    model, X_val, y_val = train_model(df)

    # 11. Evaluate
    evaluate(model, X_val, y_val)


if __name__ == "__main__":
    main()