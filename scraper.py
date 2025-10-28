# scraper.py
# Config-driven news scraper with sharding + CSV append.
# - Dynamic source from domain
# - No author/tags
# - published_date -> YYYY/MM/DD (fallback = today UTC)
# - Feeds provided via YAML; supports many URLs per category
# - Parallelizable via --shards / --shard-idx

import os, sys, time, csv, hashlib, json, argparse, yaml
import requests, feedparser, pandas as pd
from bs4 import BeautifulSoup
from readability import Document
from dateutil import parser as dtparse
from datetime import datetime, timezone
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

# ===================== DEFAULTS =====================
MAX_PER_FEED  = 60
PAUSE_SECONDS = 1.2
TIMEOUT       = 20
OUTPUT_CSV    = "out/articles.csv"
DATE_FMT      = "%Y/%m/%d"

HEADERS = {
    "User-Agent": "news-hourly-scraper/1.0 (+contact@example.com)",
    "Accept-Language": "en;q=0.9, fr;q=0.8"
}

STRIP_QUERY_PARAMS = {
    "utm_source","utm_medium","utm_campaign","utm_term","utm_content",
    "at_medium","at_campaign","at_custom1","ns_mchannel","ns_source","ns_campaign"
}

# ===================== ARGS & CONFIG =====================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML file with feeds and options")
    ap.add_argument("--out", help="override output CSV path")
    ap.add_argument("--shards", type=int, default=1, help="total shard count")
    ap.add_argument("--shard-idx", type=int, default=0, help="0-based shard index")
    return ap.parse_args()

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def choose_shard(items, shards, idx):
    return [itm for i, itm in enumerate(items) if i % max(1, shards) == idx]

# ===================== UTILS =====================
def normalize_url(u: str) -> str:
    p = urlparse(u)
    q = {k: v for k, v in parse_qsl(p.query, keep_blank_values=True)
         if k not in STRIP_QUERY_PARAMS}
    clean = p._replace(
        scheme=p.scheme.lower(),
        netloc=p.netloc.lower(),
        path=p.path.rstrip("/"),
        query=urlencode(q, doseq=True),
        fragment=""
    )
    return urlunparse(clean)

def fetch(url, timeout=TIMEOUT, retries=2):
    last_err = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(1.0 + i)
            else:
                raise last_err

def clean_join(paras):
    out = []
    for p in paras:
        txt = p.get_text(" ", strip=True)
        if not txt or len(txt) < 3:
            continue
        cls = " ".join(p.get("class", [])).lower()
        if any(bad in cls for bad in ["promo","share","related","advert","cookie"]):
            continue
        bad = False
        for anc in p.parents:
            if getattr(anc, "name", None) in ("figure","figcaption","aside","header","footer","nav"):
                bad = True; break
            acl = " ".join(anc.get("class", [])).lower() if hasattr(anc, "get") else ""
            if any(x in acl for x in ["promo","related","share","advert","cookie"]):
                bad = True; break
        if bad:
            continue
        out.append(txt)
    return "\n\n".join(out).strip()

def format_date(dt_obj):
    return dt_obj.strftime(DATE_FMT) if dt_obj else None

def parse_or_today(date_raw: str | None):
    if date_raw:
        try:
            return format_date(dtparse.parse(date_raw))
        except Exception:
            pass
    return format_date(datetime.now(timezone.utc))

# ===================== EXTRACTION =====================
def parse_article(url, category):
    try:
        html = fetch(url).text
    except Exception as e:
        print(f"[skip fetch] {url} -> {e}")
        return None

    soup = BeautifulSoup(html, "lxml")

    # Canonical & normalized URLs
    canonical = (soup.find("link", rel="canonical") or {}).get("href") or url
    canonical = normalize_url(canonical)
    norm_url  = normalize_url(url)

    # Title
    h1 = soup.select_one("h1")
    title = h1.get_text(strip=True) if h1 else (soup.find("meta", property="og:title") or {}).get("content") or ""
    if not title:
        return None

    # Image (optional)
    image = (soup.find("meta", property="og:image") or {}).get("content")

    # Published date (fallback to scrape day)
    date_raw = None
    for tag, attrs, attr in [
        ("meta", {"property": "article:published_time"}, "content"),
        ("meta", {"name": "OriginalPublicationDate"}, "content"),
        ("time", {}, "datetime"),
    ]:
        el = soup.find(tag, attrs)
        if el and el.get(attr):
            date_raw = el.get(attr); break
    published_date = parse_or_today(date_raw)  # YYYY/MM/DD

    # Body: Readability → generic selectors → JSON-LD → AMP
    content_text = ""
    try:
        content_html = Document(html).summary(html_partial=True)
        content_text = BeautifulSoup(content_html, "lxml").get_text(" ", strip=True)
    except Exception:
        pass

    # 2) Generic selectors (ASCII quotes only)
    if len(content_text) < 800:
        paras = (
            soup.select('[data-component="text-block"] p')
            or soup.select('article p')
            or soup.select('main p')
            or soup.select('[class*="RichTextComponentWrapper"] p')
        )
        if paras:
            txt = clean_join(paras)
            if len(txt) > len(content_text):
                content_text = txt

    # 3) JSON-LD articleBody
    if len(content_text) < 800:
        for s in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(s.string or "")
            except Exception:
                continue
            objs = data if isinstance(data, list) else [data]
            for obj in objs:
                if isinstance(obj, dict) and obj.get("@type") in ("NewsArticle", "Article"):
                    body = obj.get("articleBody")
                    if isinstance(body, str) and len(body) > len(content_text):
                        content_text = body.strip()
            if len(content_text) >= 800:
                break

    # 4) AMP fallback
    if len(content_text) < 800:
        amp = (soup.find("link", rel="amphtml") or {}).get("href")
        if amp:
            try:
                amp_html = fetch(amp).text
                amp_soup = BeautifulSoup(amp_html, "lxml")
                amp_paras = (
                    amp_soup.select("article p")
                    or amp_soup.select("main p")
                    or amp_soup.select("p")
                )
                amp_text = clean_join(amp_paras)
                if len(amp_text) > len(content_text):
                    content_text = amp_text
            except Exception:
                pass

    if len(content_text.strip()) < 200:
        return None

    # De-dup keys
    id_source = canonical or norm_url
    id_article = hashlib.sha1(id_source.encode()).hexdigest()[:12]
    content_hash = hashlib.sha1((title + "|" + content_text[:4000]).encode("utf-8", "ignore")).hexdigest()

    # Dynamic source from domain
    src = urlparse(canonical or norm_url).netloc
    if src.startswith("www."):
        src = src[4:]

    return {
        "id_article": id_article,
        "title": title,
        "content": content_text.strip(),
        "url": canonical or norm_url,
        "category": category,
        "source": src,
        "image": image,
        "published_date": published_date,   # YYYY/MM/DD
        "content_hash": content_hash,
    }

# ===================== STORAGE =====================
def ensure_csv(path):
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        pd.DataFrame(columns=[
            "id_article","title","content","url","category","source","image","published_date","content_hash"
        ]).to_csv(path, index=False)

def load_existing_keys(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set(), set()
    try:
        df = pd.read_csv(path, usecols=["id_article","content_hash"])
        return set(df["id_article"].astype(str)), set(df["content_hash"].astype(str))
    except Exception:
        try:
            df = pd.read_csv(path, usecols=["id_article"])
            return set(df["id_article"].astype(str)), set()
        except Exception:
            return set(), set()

# ===================== MAIN =====================
def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    feeds_cfg = cfg.get("feeds", {})
    if not feeds_cfg:
        print("No feeds in config."); sys.exit(1)

    global MAX_PER_FEED, PAUSE_SECONDS, OUTPUT_CSV
    MAX_PER_FEED  = int(cfg.get("max_per_feed", MAX_PER_FEED))
    PAUSE_SECONDS = float(cfg.get("pause_seconds", PAUSE_SECONDS))
    OUTPUT_CSV    = args.out or cfg.get("output_csv", OUTPUT_CSV)

    # Flatten (category, url) list
    flat = []
    for cat, urls in feeds_cfg.items():
        if isinstance(urls, (list, tuple, set)):
            flat += [(cat, u) for u in urls]
        else:
            flat.append((cat, urls))

    # Drop exact duplicates and shard
    flat = list(dict.fromkeys(flat))  # preserve order, remove dup pairs
    flat = choose_shard(flat, args.shards, args.shard_idx)

    ensure_csv(OUTPUT_CSV)
    seen_ids, seen_content = load_existing_keys(OUTPUT_CSV)
    seen_run_ids, seen_run_content = set(), set()
    new_rows = []

    for category, feed_url in flat:
        print(f"[feed] {category} → {feed_url}")
        feed = feedparser.parse(feed_url)
        for e in feed.entries[:MAX_PER_FEED]:
            link = e.get("link")
            if not link:
                continue
            link = normalize_url(link)
            try:
                row = parse_article(link, category)
                if not row:
                    continue
                if (row["id_article"] in seen_ids) or (row["id_article"] in seen_run_ids):
                    continue
                if (row["content_hash"] in seen_content) or (row["content_hash"] in seen_run_content):
                    continue

                new_rows.append(row)
                seen_run_ids.add(row["id_article"])
                seen_run_content.add(row["content_hash"])
                print(f"✓ {row['title'][:80]}…")
                time.sleep(PAUSE_SECONDS)
            except Exception as ex:
                print("[skip]", link, "->", ex)

    if new_rows:
        pd.DataFrame(new_rows).to_csv(
            OUTPUT_CSV, mode="a", header=False, index=False, quoting=csv.QUOTE_MINIMAL
        )
        print(f"💾 Appended {len(new_rows)} new rows to {OUTPUT_CSV}")
    else:
        print("No new rows.")

if __name__ == "__main__":
    pd.set_option("display.width", 160)
    main()
