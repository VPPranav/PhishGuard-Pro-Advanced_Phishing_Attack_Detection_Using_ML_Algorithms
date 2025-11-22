# feature_engineering.py

import re
import tldextract


def jaccard(a: str, b: str) -> float:
    """Compute Jaccard similarity between sets of characters."""
    A, B = set(a), set(b)
    if len(A.union(B)) == 0:
        return 0.0
    return len(A.intersection(B)) / len(A.union(B))


def compute_numeric_features(url: str):
    """
    Reproduces EXACT numeric feature logic used in your notebook.
    """

    ext = tldextract.extract(url)
    mld = ext.domain or ""
    ps = ext.suffix or ""

    # Numeric features used in your pipeline
    mld_res = len(mld)
    mld_ps_res = len(ps)

    card_rem = sum(c.isdigit() for c in url)

    ratio_Rrem = card_rem / (len(url) + 1)
    ratio_Arem = sum(c.isalpha() for c in url) / (len(url) + 1)

    tokens_R = re.sub(r"[^a-zA-Z0-9]", "", url)
    tokens_A = re.sub(r"[^a-zA-Z]", "", url)

    jaccard_RR = jaccard(tokens_R, tokens_R)
    jaccard_RA = jaccard(tokens_R, tokens_A)
    jaccard_AR = jaccard(tokens_A, tokens_R)
    jaccard_AA = jaccard(tokens_A, tokens_A)

    jaccard_ARrd = jaccard(tokens_A, mld)
    jaccard_ARrem = jaccard(tokens_A, url)

    return {
        "mld_res": mld_res,
        "mld.ps_res": mld_ps_res,
        "card_rem": card_rem,
        "ratio_Rrem": ratio_Rrem,
        "ratio_Arem": ratio_Arem,
        "jaccard_RR": jaccard_RR,
        "jaccard_RA": jaccard_RA,
        "jaccard_AR": jaccard_AR,
        "jaccard_AA": jaccard_AA,
        "jaccard_ARrd": jaccard_ARrd,
        "jaccard_ARrem": jaccard_ARrem,
    }
