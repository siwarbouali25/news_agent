# scripts/merge_csv.py
import os, glob, pandas as pd
from datetime import datetime

IN_DIR  = "staging"
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "articles.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# read new shard CSVs
files = glob.glob(os.path.join(IN_DIR, "**", "*.csv"), recursive=True)
if not files:
    print("No CSVs found in staging/")
    raise SystemExit(0)

dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"Loaded {f} -> {len(df)} rows")
    except Exception as e:
        print(f"Skip {f}: {e}")

if not dfs:
    print("No readable CSVs.")
    raise SystemExit(0)

# combine all new runs
all_df = pd.concat(dfs, ignore_index=True)

# 🟢 load existing articles if file exists
if os.path.exists(OUT_CSV) and os.path.getsize(OUT_CSV) > 0:
    old_df = pd.read_csv(OUT_CSV)
    print(f"Loaded existing CSV with {len(old_df)} rows")
    all_df = pd.concat([old_df, all_df], ignore_index=True)

# normalize columns
cols = ["id_article","title","content","url","category","source","image","published_date","content_hash"]
for c in cols:
    if c not in all_df.columns:
        all_df[c] = None
all_df = all_df[cols]

# dedupe (keep the earliest copy)
before = len(all_df)
all_df.drop_duplicates(subset=["content_hash"], inplace=True, keep="first")
all_df.drop_duplicates(subset=["url", "published_date"], inplace=True, keep="first")
after = len(all_df)
print(f"Deduped {before} -> {after} rows")

# sort newest first
all_df.sort_values(by=["published_date","title"], ascending=[False, True], inplace=True)
all_df.reset_index(drop=True, inplace=True)

# write merged CSV
all_df.to_csv(OUT_CSV, index=False)
print(f"💾 Updated {OUT_CSV} with {len(all_df)} total rows at {datetime.utcnow().isoformat()}Z")
