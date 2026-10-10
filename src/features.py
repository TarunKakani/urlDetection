"""
Shared URL preprocessing for the phishing URL detector.

Every phase imports from this file (Phase 4 feature extraction, Phase 7 evaluation,
Phase 8 predict.py), so train, test and the live demo all see exactly the same
transformation.

Two input modes:
  "host" (main model)  -> only the hostname, lowercased, leading "www." removed.
                          PhiUSIIL's legitimate URLs are all https + www + no path,
                          so a rule on those three alone scores 99.6%. Host mode
                          removes all three shortcuts.
  "url"  (comparison)  -> the full URL, scheme removed, lowercased. Used only to
                          show how much the shortcut inflates results.

  A. Character tokenization       -> build_vocab, encode, decode, save_vocab, load_vocab
  B. Handcrafted lexical features -> lexical_features, lexical_frame
  C. Host-level deduplication     -> to_host_dataset
"""
import ipaddress
import json
import math
import re
from collections import Counter
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tldextract

# Same settings as the Phase 3 split: free-hosting platforms (github.io,
# 000webhostapp.com, ...) are treated as public suffixes.
_extract = tldextract.TLDExtract(include_psl_private_domains=True)

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
MODES = ("host", "url")


# ---------------------------------------------------------------------------
# Normalization: what each model actually sees
# ---------------------------------------------------------------------------
def strip_scheme(url: str) -> str:
    """Remove 'http://' / 'https://' (or any scheme)."""
    return _SCHEME_RE.sub("", str(url).strip())


def normalize(url: str) -> str:
    """Full-URL mode text: scheme removed, lowercased. PhiUSIIL's URLs contain no
    uppercase letters, so lowercasing makes real-world URLs match the training data."""
    return strip_scheme(url).lower()


def host_of(url: str) -> str:
    """Host-mode text: the hostname only, lowercased, with a leading 'www.'
    removed (every legitimate PhiUSIIL URL has www, so it is a shortcut)."""
    u = str(url).strip()
    try:
        host = urlsplit(u if "://" in u else "http://" + u).hostname or ""
    except ValueError:
        host = strip_scheme(u).split("/")[0].split("@")[-1].split(":")[0]
    host = host.lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def to_text(url: str, mode: str = "host") -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return host_of(url) if mode == "host" else normalize(url)


# ---------------------------------------------------------------------------
# A. Character tokenization
# ---------------------------------------------------------------------------
PAD, UNK = 0, 1


def build_vocab(urls, mode: str = "host", min_count: int = 5) -> dict:
    """Map each character seen at least `min_count` times to an integer id.
    Ids 0 and 1 are reserved for padding and unknown characters.
    Call this on the TRAINING URLs only."""
    counts = Counter(ch for u in urls for ch in to_text(u, mode))
    chars = sorted(ch for ch, c in counts.items() if c >= min_count)
    return {ch: i + 2 for i, ch in enumerate(chars)}


def encode(urls, vocab: dict, max_len: int, mode: str = "host") -> np.ndarray:
    """Turn URLs into a (n, max_len) int16 array. Long inputs are truncated at
    the end, short ones padded with 0 at the end. Unknown characters -> UNK (1)."""
    X = np.zeros((len(urls), max_len), dtype=np.int16)
    for i, u in enumerate(urls):
        ids = [vocab.get(ch, UNK) for ch in to_text(u, mode)[:max_len]]
        X[i, : len(ids)] = ids
    return X


def decode(row, vocab: dict) -> str:
    """Inverse of encode for one row, for sanity checks. UNK shows as '?'."""
    inv = {i: ch for ch, i in vocab.items()}
    return "".join(inv.get(int(i), "?") for i in row if i != PAD)


def save_vocab(vocab: dict, max_len: int, mode: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"mode": mode, "max_len": max_len, "vocab": vocab},
                  f, ensure_ascii=False, indent=1)


def load_vocab(path: str):
    """Returns (vocab, max_len, mode) -- use all three when encoding new URLs."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["vocab"], d["max_len"], d["mode"]


# ---------------------------------------------------------------------------
# B. Handcrafted lexical features
# ---------------------------------------------------------------------------
SUSPICIOUS_WORDS = [
    "login", "signin", "logon", "verify", "secure", "account", "update",
    "bank", "confirm", "password", "billing", "wallet", "suspend", "unlock",
    "webscr", "authenticate", "recover",
]

# Cheap TLDs that appear disproportionately in phishing. Based on public
# "most abused TLD" reports (e.g. Spamhaus) -- cite the one you use.
ABUSED_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "online", "site",
    "icu", "buzz", "live", "shop", "work", "rest", "cyou", "sbs", "cfd",
}

# Fixed list so every split gets the same one-hot columns in the same order.
TLD_CATEGORIES = ["ip", "abused", "com", "org_net", "edu_gov", "cctld", "other_gtld"]


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def _path_len(url: str) -> int:
    u = str(url).strip()
    try:
        p = urlsplit(u if "://" in u else "http://" + u)
    except ValueError:
        return 0
    return len(p.path.rstrip("/")) + len(p.query) + len(p.fragment)


def _tld_category(suffix: str, is_ip: bool) -> str:
    if is_ip:
        return "ip"
    labels = suffix.lower().split(".") if suffix else []
    tld = labels[-1] if labels else ""
    if tld in ABUSED_TLDS:
        return "abused"
    if tld == "com":
        return "com"
    if tld in {"org", "net"}:
        return "org_net"
    if any(l in {"edu", "gov", "mil", "ac", "int"} for l in labels):
        return "edu_gov"
    if len(tld) == 2:
        return "cctld"
    return "other_gtld"


def lexical_features(url: str, mode: str = "host") -> dict:
    """Handcrafted features for one URL.

    Both modes: character counts, entropy and keyword hits on the text the
    model sees, plus host structure (lengths, subdomains, IP, TLD category).
    "url" mode adds the path/scheme features, which are the shortcut ones."""
    raw = str(url).strip()
    text = to_text(raw, mode)
    host = host_of(raw)
    ip = _is_ip(host)
    ext = _extract(host)

    feats = {
        "len": len(text),
        "host_len": len(host),
        "sld_len": 0 if ip else len(ext.domain),  # the 'paypal' in paypal.com
        "n_dots": text.count("."),
        "n_hyphens": text.count("-"),
        "n_digits": sum(ch.isdigit() for ch in text),
        "is_ip": int(ip),
        "n_subdomains": 0 if ip or not ext.subdomain else len(ext.subdomain.split(".")),
        "entropy": round(shannon_entropy(text), 4),
        "n_suspicious": sum(text.count(w) for w in SUSPICIOUS_WORDS),
        "tld_cat": _tld_category(ext.suffix, ip),
    }
    if mode == "url":
        feats.update({
            "path_len": _path_len(raw),
            "n_slashes": text.count("/"),
            "n_at": text.count("@"),
            "is_https": int(raw.lower().startswith("https://")),
        })
    return feats


def lexical_frame(urls, mode: str = "host") -> pd.DataFrame:
    """Feature table for many URLs, with tld_cat one-hot encoded over the fixed
    TLD_CATEGORIES list (so train, val and test always share the same columns)."""
    df = pd.DataFrame([lexical_features(u, mode) for u in urls])
    df["tld_cat"] = pd.Categorical(df["tld_cat"], categories=TLD_CATEGORIES)
    return pd.get_dummies(df, columns=["tld_cat"], prefix="tld", dtype=int)


# ---------------------------------------------------------------------------
# C. Host-level deduplication (host mode only)
# ---------------------------------------------------------------------------
def to_host_dataset(df: pd.DataFrame, url_col: str = "URL", label_col: str = "label"):

    """One row per unique host.

    Many phishing URLs share a host and differ only in path; in host mode they
    become identical inputs, so only one is kept. Hosts that appear with BOTH
    labels (usually a compromised legitimate site serving phishing at a path)
    can't be separated by a host-only model, so they are dropped and returned
    separately for the report.

    Returns (deduped_df, conflicts_df)."""
    d = df.copy()
    d["host"] = d[url_col].map(host_of)
    n_labels = d.groupby("host")[label_col].transform("nunique")
    conflicts = d[n_labels > 1]
    d = d[n_labels == 1].drop_duplicates("host").reset_index(drop=True)
    return d, conflicts