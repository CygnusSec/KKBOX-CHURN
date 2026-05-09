#!/usr/bin/env python3
"""
Create monthly label files (user_label_201702.csv and user_label_201703.csv)
from existing transaction data.

Logic based on WSDMChurnLabeller.scala:
- For users with membership expiring in Feb 2017 (201702):
  - Check if they renewed within 30 days after expiration
  - If no renewal or renewal gap >= 30 days → churn = 1
  - If renewed within 30 days → churn = 0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Loading data...")
transactions = pd.read_csv("data/transactions_v2.csv")

# Convert dates
print("Converting dates...")
transactions["transaction_date"] = pd.to_datetime(
    transactions["transaction_date"].astype(str), format="%Y%m%d", errors="coerce"
)
transactions["membership_expire_date"] = pd.to_datetime(
    transactions["membership_expire_date"].astype(str), format="%Y%m%d", errors="coerce"
)

# Sort by user and transaction date
transactions = transactions.sort_values(["msno", "transaction_date", "membership_expire_date"])

print(f"Total transactions: {len(transactions):,}")

def create_monthly_labels(cutoff_date, expire_start, expire_end, output_file):
    """
    Create churn labels for users whose membership expires in a specific month.
    
    Parameters:
    -----------
    cutoff_date : str
        Date to split history vs future (e.g., "2017-01-31")
    expire_start : str
        Start of expiration window (e.g., "2017-02-01")
    expire_end : str
        End of expiration window (e.g., "2017-02-28")
    output_file : str
        Output CSV filename
    """
    print(f"\n{'='*60}")
    print(f"Creating {output_file}")
    print(f"Cutoff date: {cutoff_date}")
    print(f"Expiration window: {expire_start} to {expire_end}")
    print(f"{'='*60}")
    
    cutoff = pd.to_datetime(cutoff_date)
    exp_start = pd.to_datetime(expire_start)
    exp_end = pd.to_datetime(expire_end)
    
    # Split into history (before cutoff) and future (after cutoff)
    history = transactions[transactions["transaction_date"] <= cutoff].copy()
    future = transactions[transactions["transaction_date"] > cutoff].copy()
    
    print(f"History transactions: {len(history):,}")
    print(f"Future transactions: {len(future):,}")
    
    # Get last expiration date for each user in history period
    print("\nCalculating last expiration dates...")
    last_expire = (
        history
        .sort_values(["msno", "transaction_date", "membership_expire_date"])
        .groupby("msno")
        .tail(1)
        [["msno", "membership_expire_date"]]
        .rename(columns={"membership_expire_date": "last_expire"})
    )
    
    # Filter users whose membership expires in target month
    candidates = last_expire[
        (last_expire["last_expire"] >= exp_start) & 
        (last_expire["last_expire"] <= exp_end)
    ].copy()
    
    print(f"Users with membership expiring in target month: {len(candidates):,}")
    
    # Join with future transactions
    print("\nJoining with future transactions...")
    joined = candidates.merge(future, on="msno", how="left")
    
    # Users with no future activity → churn
    no_activity = joined[joined["transaction_date"].isna()][["msno"]].drop_duplicates()
    no_activity["is_churn"] = 1
    print(f"Users with no future activity (churn): {len(no_activity):,}")
    
    # Users with future activity → check renewal gap
    with_activity = joined[joined["transaction_date"].notna()].copy()
    
    if len(with_activity) > 0:
        print("\nCalculating renewal gaps...")
        # Calculate gap between last expiration and first future transaction
        renewal_gaps = (
            with_activity
            .sort_values(["msno", "transaction_date"])
            .groupby("msno")
            .agg({
                "last_expire": "first",
                "transaction_date": "first"
            })
            .reset_index()
        )
        
        renewal_gaps["gap_days"] = (
            renewal_gaps["transaction_date"] - renewal_gaps["last_expire"]
        ).dt.days
        
        # Renewal within 30 days → not churn
        valid_renewals = renewal_gaps[renewal_gaps["gap_days"] < 30][["msno"]].copy()
        valid_renewals["is_churn"] = 0
        print(f"Users with valid renewal (gap < 30 days): {len(valid_renewals):,}")
        
        # Late renewal (>= 30 days) → churn
        late_renewals = renewal_gaps[renewal_gaps["gap_days"] >= 30][["msno"]].copy()
        late_renewals["is_churn"] = 1
        print(f"Users with late renewal (gap >= 30 days): {len(late_renewals):,}")
        
        # Combine all results
        result = pd.concat([no_activity, valid_renewals, late_renewals], ignore_index=True)
    else:
        result = no_activity
    
    # Remove duplicates (keep first)
    result = result.drop_duplicates(subset=["msno"], keep="first")
    
    print(f"\nTotal labeled users: {len(result):,}")
    print(f"Churn rate: {result['is_churn'].mean():.2%}")
    print(f"  - Churn (1): {(result['is_churn'] == 1).sum():,}")
    print(f"  - Not churn (0): {(result['is_churn'] == 0).sum():,}")
    
    # Save to CSV
    result.to_csv(f"data/{output_file}", index=False)
    print(f"\n✅ Saved to data/{output_file}")
    
    return result


# Create February 2017 labels (201702)
# History: Jan 2017 (up to Jan 31)
# Expiration window: Feb 1-28, 2017
# Future: After Jan 31, 2017
df_201702 = create_monthly_labels(
    cutoff_date="2017-01-31",
    expire_start="2017-02-01",
    expire_end="2017-02-28",
    output_file="user_label_201702.csv"
)

# Create March 2017 labels (201703)
# History: Feb 2017 (up to Feb 28)
# Expiration window: Mar 1-31, 2017
# Future: After Feb 28, 2017
df_201703 = create_monthly_labels(
    cutoff_date="2017-02-28",
    expire_start="2017-03-01",
    expire_end="2017-03-31",
    output_file="user_label_201703.csv"
)

print("\n" + "="*60)
print("✅ All label files created successfully!")
print("="*60)
print("\nFiles created:")
print("  - data/user_label_201702.csv")
print("  - data/user_label_201703.csv")
print("\nYou can now use these files in your training pipeline:")
print('  df_train = pd.read_csv("data/user_label_201702.csv")')
print('  df_cv = pd.read_csv("data/user_label_201703.csv")')
