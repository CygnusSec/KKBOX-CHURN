import pandas as pd

def process_logs(file_path, chunksize=5_000_000):
    agg_list = []

    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        chunk['date'] = pd.to_datetime(chunk['date'])

        agg = chunk.groupby('msno').agg({
            'total_secs': ['sum', 'mean'],
            'num_unq': ['mean'],
            'num_25': ['sum'],
            'num_50': ['sum'],
            'num_75': ['sum'],
            'num_985': ['sum'],
            'num_100': ['sum']
        })

        agg.columns = ['_'.join(col) for col in agg.columns]
        agg_list.append(agg)

    # combine all chunks
    full_agg = pd.concat(agg_list)
    full_agg = full_agg.groupby(full_agg.index).sum()

    return full_agg.reset_index()