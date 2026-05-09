import pandas as pd
import glob
import os

station = 'rkpuram'
files = glob.glob(f'hourly_data/{station}/*.csv')
dfs = []
for f in files:
    df = pd.read_csv(f)
    df.columns = df.columns.str.strip()
    dfs.append(df)

df = pd.concat(dfs)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp')

target = 'Ozone (µg/m³)'
print(f"Station: {station}")
print(f"Total rows: {len(df)}")
df['Year'] = df['Timestamp'].dt.year
yearly_stats = df.groupby('Year')[target].agg(['mean', 'count', lambda x: x.isna().sum()])
yearly_stats.columns = ['Mean_O3', 'Total_Rows', 'Missing_Rows']
print("\nYearly Stats:")
print(yearly_stats)
