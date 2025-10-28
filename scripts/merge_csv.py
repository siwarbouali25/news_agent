# scripts/merge_csv.py
import os, glob, pandas as pd
from datetime import datetime

IN_DIR  = "staging"                 # where artifacts are downloaded
OUT_DIR = "data"                    # committed to git
OUT_CSV = os.path.join(OUT_DIR, "articles.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# read all shard CSVs
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

all_df = pd.concat(dfs, ignore_index=True)

# keep a consistent column order
cols = ["id_article","title","content","url","category","source","image","published_date","content_hash"]
for c in cols:
    if c not in all_df.columns:
        all_df[c] = None
all_df = all_df[cols]

# dedupe: prefer content_hash, then url+published_date
before = len(all_df)
all_df.drop_duplicates(subset=["content_hash"], inplace=True, keep="first")
all_df.drop_duplicates(subset=["url", "published_date"], inplace=True, keep="first")
after = len(all_df)
print(f"Deduped {before} -> {after} rows")

# sort newest first (string YYYY/MM/DD sorts OK; add secondary key title for stability)
all_df.sort_values(by=["published_date","title"], ascending=[False, True], inplace=True)
all_df.reset_index(drop=True, inplace=True)

# write single CSV
all_df.to_csv(OUT_CSV, index=False)
print(f"Wrote {OUT_CSV} with {len(all_df)} rows at {datetime.utcnow().isoformat()}Z")
