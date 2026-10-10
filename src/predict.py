"""
predict.py -- score URLs with the host-only phishing models.

Usage:
    python predict.py https://secure-login.paypa1-verify.top/x
    python predict.py example.com github.com another.site
    python predict.py --file urls.txt
    python predict.py --json paypal-login.xyz

Needs, relative to this file (all produced by Phases 4-7):
    features.py
    data/features/host/vocab.json
    models/host_cnn.pt
    models/host_bilstm.pt
    models/thresholds.json        (last cell of the Phase 7 notebook)

How a verdict is reached
    The CNN is the primary model; the BiLSTM score is shown alongside it.
    Two thresholds come from Phase 7, and neither was tuned on test data:
      * strict  -- 1% FPR on 2,000 independent long-tail Tranco domains
      * lenient -- 1% FPR on the PhiUSIIL validation split
    Above strict            -> PHISHING
    Between lenient/strict  -> SUSPICIOUS
    Below lenient           -> LIKELY LEGITIMATE
    Hosts on shared hosting platforms (vercel.app, github.io, pages.dev, ...)
    get CAN'T JUDGE instead: Phase 7 showed the models flag ~99% of legitimate
    sites there, so a verdict from the hostname alone would be meaningless.
"""

import argparse
import json
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import tldextract

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F  # noqa: E402

# tldextract logs a full traceback when it can't download the suffix list
# (e.g. no internet during a demo); it then falls back to its bundled copy.
logging.getLogger("tldextract").setLevel(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Model definitions -- identical to the Phase 6 notebook, so the saved
# state dicts load without changes.
# ---------------------------------------------------------------------------
class CharCNN(nn.Module):
    def __init__(self, n_vocab, emb=32, filters=128, kernels=(3, 5, 7), hidden=64, drop=0.5):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, emb, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb, filters, k, padding=k // 2) for k in kernels)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(filters * len(kernels), hidden),
                                  nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        mask = (x != 0).unsqueeze(1)
        h = self.emb(x).transpose(1, 2)
        pooled = [torch.relu(c(h)).masked_fill(~mask, 0).amax(dim=2) for c in self.convs]
        return self.head(torch.cat(pooled, dim=1)).squeeze(1)


class CharBiLSTM(nn.Module):
    def __init__(self, n_vocab, emb=32, hidden=64, drop=0.5):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * hidden, 1))

    def forward(self, x):
        lengths = (x != 0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(x), lengths,
                                                   batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(torch.cat([h[0], h[1]], dim=1)).squeeze(1)


def _sigmoid(z: float) -> float:
    # float64 sigmoid: no saturation at 1.0 the way float32 torch.sigmoid has
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class Detector:
    VERDICTS = {
        "phishing":   "PHISHING",
        "suspicious": "SUSPICIOUS",
        "legit":      "LIKELY LEGITIMATE",
        "platform":   "CAN'T JUDGE (shared hosting platform)",
        "invalid":    "INVALID INPUT",
    }

    def __init__(self, root: str = HERE):
        def path(*p):
            full = os.path.join(root, *p)
            if not os.path.exists(full):
                raise FileNotFoundError(
                    f"Missing {os.path.relpath(full, root)}. "
                    + ("Run the last cell of the Phase 7 notebook to create it."
                       if p[-1] == "thresholds.json" else
                       "Run predict.py from the project folder that holds models/ and data/."))
            return full

        self.vocab, self.max_len, mode = F.load_vocab(path("../data", "features", "host", "vocab.json"))
        if mode != "host":
            raise ValueError(f"vocab.json was built in '{mode}' mode; predict.py needs 'host'.")

        self.nets = {"cnn": self._load(CharCNN, path("../models", "host_cnn.pt")),
                     "bilstm": self._load(CharBiLSTM, path("../models", "host_bilstm.pt"))}

        with open(path("../models", "thresholds.json")) as f:
            thr = json.load(f)
        # Compare everything in logit space: strict thresholds sit at logits of 6-9,
        # where float32 probabilities lose resolution.
        self.thr = {n: {"strict": float(thr[n]["tranco_logit_1pct"]),
                        "lenient": _logit(float(thr[n]["val_prob_1pct"]))}
                    for n in self.nets}

        # Same Public Suffix List handling as Phase 7 (downloads once and caches;
        # falls back to the snapshot bundled with tldextract when offline)
        self._icann = tldextract.TLDExtract(include_psl_private_domains=False)
        self._priv = tldextract.TLDExtract(include_psl_private_domains=True)

    @staticmethod
    def _load(cls, p):
        sd = torch.load(p, map_location="cpu")
        net = cls(sd["emb.weight"].shape[0])
        net.load_state_dict(sd)
        return net.eval()

    def _logits(self, net, hosts):
        X = F.encode(hosts, self.vocab, self.max_len, "host")
        X = np.where(X < net.emb.num_embeddings, X, F.UNK)
        with torch.inference_mode():          # per call: Streamlit scores in a worker thread
            return net(torch.from_numpy(X).long()).double().numpy()

    def _level(self, model, z):
        t = self.thr[model]
        return "phishing" if z > t["strict"] else "suspicious" if z > t["lenient"] else "legit"

    def predict(self, urls):
        """Score a list of URLs. Returns one dict per URL, in order."""
        urls = [str(u).strip() for u in urls]
        hosts = [F.to_text(u, "host") if u else "" for u in urls]
        ok = [i for i, h in enumerate(hosts) if h and "." in h and not any(c.isspace() for c in h)]

        z = {n: np.full(len(urls), np.nan) for n in self.nets}
        if ok:
            for n, net in self.nets.items():
                z[n][ok] = self._logits(net, [hosts[i] for i in ok])

        out = []
        for i, (u, h) in enumerate(zip(urls, hosts)):
            if i not in ok:
                out.append({"url": u, "host": h, "verdict": "invalid",
                            "label": self.VERDICTS["invalid"], "notes": ["No hostname found."]})
                continue

            priv, icann = self._priv(h), self._icann(h)
            platform = priv.suffix if priv.suffix != icann.suffix and priv.domain else None

            r = {"url": u, "host": h, "platform": platform, "notes": []}
            for n in self.nets:
                r[f"{n}_prob"] = _sigmoid(z[n][i])
                r[f"{n}_logit"] = float(z[n][i])
                r[f"{n}_level"] = self._level(n, z[n][i])

            if platform:
                r["verdict"] = "platform"
                r["notes"].append(
                    f"Subdomain of {platform}, a shared hosting platform. Anyone can create "
                    f"one, so the name says almost nothing; in evaluation the models flagged "
                    f"nearly every legitimate site on such platforms. Check the page itself.")
            else:
                r["verdict"] = r["cnn_level"]

            if "xn--" in h:
                r["notes"].append("Internationalised (non-Latin) domain. These can be homograph "
                                  "attacks, but legitimate IDNs are also over-flagged by this model.")
            if not platform and r["cnn_level"] != r["bilstm_level"]:
                r["notes"].append(f"The models disagree: BiLSTM says "
                                  f"{self.VERDICTS[r['bilstm_level']].lower()}.")
            r["label"] = self.VERDICTS[r["verdict"]]
            out.append(r)
        return out


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def _print(results):
    for r in results:
        print(f"\n{r['url']}")
        print(f"  host     {r['host'] or '-'}")
        print(f"  verdict  {r['label']}")
        if r["verdict"] != "invalid":
            print(f"  CNN      {r['cnn_prob']:.4f}   (primary)")
            print(f"  BiLSTM   {r['bilstm_prob']:.4f}")
        for note in r["notes"]:
            print(f"  note     {note}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Host-only phishing URL detector.")
    ap.add_argument("urls", nargs="*", help="URLs or hostnames to score")
    ap.add_argument("--file", help="text file with one URL per line")
    ap.add_argument("--json", action="store_true", help="print results as JSON")
    args = ap.parse_args(argv)

    urls = list(args.urls)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            urls += [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if not urls:
        ap.error("give at least one URL, or --file")

    try:
        det = Detector()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")

    results = det.predict(urls)
    if args.json:
        print(json.dumps(results, indent=1))
    else:
        _print(results)


if __name__ == "__main__":
    main()