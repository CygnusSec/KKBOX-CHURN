import pandas as pd

def create_transaction_features(transactions):
    agg = transactions.groupby('msno').agg({
        'actual_amount_paid': ['mean', 'sum'],
        'payment_plan_days': ['mean'],
        'is_cancel': ['sum'],
        'is_auto_renew': ['mean']
    })

    agg.columns = ['_'.join(col) for col in agg.columns]

    # Ratio features
    agg['cancel_rate'] = agg['is_cancel_sum'] / (agg['payment_plan_days_mean'] + 1)
    agg['payment_per_day'] = agg['actual_amount_paid_sum'] / (agg['payment_plan_days_mean'] + 1)

    return agg.reset_index()


def merge_all(train, members, trans_feat):
    df = train.merge(members, on='msno', how='left')
    df = df.merge(trans_feat, on='msno', how='left')

    return df


def encode_features(df):
    df = pd.get_dummies(df, columns=['city', 'gender'], dummy_na=True)
    return df

def create_log_features(logs_df):
    # thêm ratio feature
    logs_df['completion_rate'] = (
        logs_df['num_100_sum'] /
        (logs_df['num_25_sum'] + logs_df['num_50_sum'] +
         logs_df['num_75_sum'] + logs_df['num_985_sum'] + 1)
    )

    return logs_df