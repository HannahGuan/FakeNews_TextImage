from pathlib import Path
import os, re, shutil, sys, urllib.parse, json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageFile
import requests
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configuration
DATA_ROOT = Path("data/fakeddit")
IMAGES_DIR = DATA_ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("train_sampled_with_images.csv")

# Load data
def load_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".csv",):
        return pd.read_csv(path)
    if suf in (".tsv",):
        return pd.read_csv(path, sep="\t")
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suf}")

df = load_any(SOURCE)
print(f"[LOAD] {SOURCE} -> {df.shape[0]} rows, {df.shape[1]} cols")

# Required source columns we’ll use
need = {"id", "clean_title", "image_url", "2_way_label", "hasImage"}
missing = [c for c in need if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns in source: {missing}")

# Normalize booleans and labels
def is_true_scalar(x):
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in ("true", "1", "t", "y", "yes")

df["hasImage_bool"] = df["hasImage"].apply(is_true_scalar)

df = df[df["hasImage_bool"] & df["image_url"].notna() & df["id"].notna()].copy()

# Ensure 2_way_label is 0/1
def to_bin(y):
    try:
        v = int(str(y).strip())
        return v if v in (0,1) else None
    except Exception:
        return None

df["2_way_label"] = df["2_way_label"].apply(to_bin)
df = df[df["2_way_label"].notna()].copy()
df["2_way_label"] = df["2_way_label"].astype(int)

# Download
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
TIMEOUT = 12

def real_url(u: str) -> str:
    # decode %2e, etc.
    u = urllib.parse.unquote(u)
    # strip Reddit redirect pattern ?url=
    if "reddit.com/media?url=" in u:
        q = urllib.parse.urlparse(u).query
        params = urllib.parse.parse_qs(q)
        if "url" in params: 
            return params["url"][0]
    return u

def download_one(row) -> tuple[str,bool,str|None]:
    rid = str(row["id"])
    url = real_url(str(row["image_url"]))
    # force .jpg extension in our store
    dst = IMAGES_DIR / f"{rid}.jpg"
    if dst.exists():
        return (rid, True, None)
    try:
        r = session.get(url, stream=True, timeout=TIMEOUT)
        r.raise_for_status()
        ct = r.headers.get("content-type","")
        if not ct.startswith("image/"):
            return (rid, False, f"not-image({ct})")
        im = Image.open(r.raw)
        # convert to RGB/JPEG-friendly
        if im.mode not in ("RGB","L"):
            im = im.convert("RGB")
        im.save(dst, "JPEG", quality=95)
        return (rid, True, None)
    except Exception as e:
        return (rid, False, f"{type(e).__name__}: {e}")

rows = df[["id","image_url"]].drop_duplicates(subset=["id"]).to_dict(orient="records")
print(f"[DL] Downloading {len(rows)} images to {IMAGES_DIR} ...")
fails = []
ok = 0
with ThreadPoolExecutor(max_workers=12) as ex:
    for rid, success, err in ex.map(download_one, rows):
        if success: ok += 1
        else: fails.append((rid, err))
print(f"[DL] Done. ok={ok}, failed={len(fails)}")
if fails:
    (DATA_ROOT / "failed_downloads.tsv").write_text(
        "id\terror\n" + "\n".join([f"{i}\t{e}" for i,e in fails]),
        encoding="utf-8"
    )
    print(f"[DL] Fail log: {DATA_ROOT/'failed_downloads.tsv'}")

# Build train/val
# Keep only rows whose image exists
def img_exists(r):
    return (IMAGES_DIR / f"{str(r['id'])}.jpg").exists()

df = df[df.apply(img_exists, axis=1)].copy()
df["image_path"] = "images/" + df["id"].astype(str) + ".jpg"
df = df[["image_path","clean_title","2_way_label"]].reset_index(drop=True)

# Split: 80/20 (val will be split again into val/test inside train.py)
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["2_way_label"]
)

DATA_ROOT.mkdir(parents=True, exist_ok=True)
train_df.to_csv(DATA_ROOT / "train.csv", index=False)
val_df.to_csv(DATA_ROOT / "val.csv", index=False)

print("[SAVE] ", DATA_ROOT / "train.csv", f"({len(train_df)})")
print("[SAVE] ", DATA_ROOT / "val.csv",   f"({len(val_df)})")

print("\nSample rows:")
print(train_df.head().to_string(index=False))
