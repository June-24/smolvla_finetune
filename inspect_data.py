import pandas as pd
import json

data_root = "/mnt/c/Users/Moham/OneDrive/Desktop/Claude Projects/smolvla_finetune/data"
for chunk in ["chunk_0to9", "chunk_10to19", "chunk_20to29", "chunk_30to39"]:
    df = pd.read_parquet(f"{data_root}/{chunk}/data.parquet")
    with open(f"{data_root}/{chunk}/task_names.json") as f:
        tasks = json.load(f)
    print(f"{chunk}: {len(df)} rows, {df['episode_index'].nunique()} episodes, {len(tasks)} tasks")
    print(f"  columns: {list(df.columns)}")
print("Done")
