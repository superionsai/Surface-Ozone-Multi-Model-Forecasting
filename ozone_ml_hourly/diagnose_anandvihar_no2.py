import pandas as pd
import glob
import os

station = 'anandvihar'
files = glob.glob(f'hourly_data/{station}/*.csv')
dfs = []
for f in files:
    df = pd.read_csv(f)
    df.columns = df.columns.str.strip()
    dfs.append(df)

df = pd.concat(dfs)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp')

o3_col = 'Ozone (µg/m³)'
no2_col = 'NO2 (µg/m³)'

df['Year'] = df['Timestamp'].dt.year
stats = df.groupby('Year').agg({
    o3_col: 'mean',
    no2_col: 'mean'
})
print(f"Station: {station}")
print(stats)
