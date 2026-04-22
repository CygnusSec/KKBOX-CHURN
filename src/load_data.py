import pandas as pd
from config import DATA_PATH

def load_data():
    train = pd.read_csv(DATA_PATH + "train_v2.csv")
    members = pd.read_csv(DATA_PATH + "members_v3.csv")
    transactions = pd.read_csv(DATA_PATH + "transactions_v2.csv")

    return train, members, transactions