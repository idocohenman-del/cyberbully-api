import os
import pandas as pd

data_folder = "data"

for filename in os.listdir(data_folder):
    if filename.endswith(".csv"):
        filepath = os.path.join(data_folder, filename)
        try:
            df = pd.read_csv(filepath, nrows=2)
            print(f"\n{filename}")
            print(f"  Columns: {list(df.columns)}")
        except Exception as e:
            print(f"\n{filename} — ERROR: {e}")