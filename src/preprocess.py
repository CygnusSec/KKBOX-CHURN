import pandas as pd

def clean_members(members):
    # Fix age
    members['bd'] = members['bd'].clip(10, 80)

    # Fill gender
    members['gender'] = members['gender'].fillna('unknown')

    return members


def clean_transactions(transactions):
    # Remove duplicates
    transactions = transactions.drop_duplicates()

    # Clip payment
    transactions['actual_amount_paid'] = transactions['actual_amount_paid'].clip(0, 5000)

    # Convert datetime
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
    transactions['membership_expire_date'] = pd.to_datetime(transactions['membership_expire_date'])

    return transactions