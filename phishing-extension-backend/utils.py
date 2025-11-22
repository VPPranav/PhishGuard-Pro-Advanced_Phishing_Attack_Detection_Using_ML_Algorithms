# # utils.py
# import pandas as pd
# import numpy as np
# import logging

# LOG = logging.getLogger("uvicorn.error")

# def _get_transformer_last(est):
#     """
#     If est is a Pipeline, return its last step (est.steps[-1][1]), else return est.
#     """
#     try:
#         steps = getattr(est, "steps", None)
#         if steps:
#             return steps[-1][1]
#     except Exception:
#         pass
#     return est


# def build_input_row(url: str, model):
#     """
#     Build a single-row pandas DataFrame matching the training feature names
#     expected by the model's ColumnTransformer. This function inspects the
#     model.named_steps['prep'] (ColumnTransformer) and splits columns into
#     text vs numeric, then fills text cols with strings and numeric cols with 0.
#     """

#     # Try to find ColumnTransformer named 'prep' (common in our notebook pipeline)
#     try:
#         prep = None
#         # model might be a Pipeline; search for ColumnTransformer inside named_steps
#         if hasattr(model, "named_steps"):
#             for n, step in model.named_steps.items():
#                 # heuristics: name contains 'prep' or 'pre' or step is ColumnTransformer-like
#                 if n.lower().startswith("prep") or n.lower().startswith("pre"):
#                     prep = step
#                     break
#             # if not found by name, try to locate first ColumnTransformer instance
#             if prep is None:
#                 for n, step in model.named_steps.items():
#                     # inspect attributes typical of ColumnTransformer
#                     if hasattr(step, "transformers_") and hasattr(step, "feature_names_in_"):
#                         prep = step
#                         break

#         # as fallback, maybe model itself is a ColumnTransformer
#         if prep is None and hasattr(model, "transformers_"):
#             prep = model

#         if prep is None:
#             raise RuntimeError("Could not find ColumnTransformer (prep) inside the saved pipeline.")
#     except Exception as e:
#         LOG.exception("Error locating ColumnTransformer in model: %s", e)
#         raise

#     # get the ordered feature names used during training
#     try:
#         all_cols = list(prep.feature_names_in_)
#     except Exception:
#         # fallback: try to combine transformers' columns
#         cols = []
#         for name, transformer, c in getattr(prep, "transformers_", []):
#             if c is None:
#                 continue
#             if isinstance(c, (list, tuple)):
#                 cols.extend(list(c))
#         all_cols = cols

#     # Determine which columns are numeric vs text by inspecting transformers_
#     numeric_cols = set()
#     text_cols = set()

#     try:
#         for name, transformer, cols in getattr(prep, "transformers_", []):
#             if cols is None:
#                 continue
#             # normalize cols list
#             cols_list = list(cols) if isinstance(cols, (list, tuple, np.ndarray)) else [cols]

#             # get last estimator if pipeline
#             last = _get_transformer_last(transformer)

#             # Heuristics to detect text transformers
#             # If last has vocabulary_ or is instance of known text vectorizers -> text
#             is_text = False
#             try:
#                 # avoid importing sklearn heavy modules here; check attributes
#                 if hasattr(last, "vocabulary_") or hasattr(last, "ngram_range"):
#                     is_text = True
#             except Exception:
#                 pass

#             # If name looks like 'text' or contains 'url', consider text
#             if (isinstance(name, str) and ("text" in name.lower() or "url" in name.lower())):
#                 is_text = True

#             # assign
#             for c in cols_list:
#                 if is_text:
#                     text_cols.add(c)
#                 else:
#                     numeric_cols.add(c)
#     except Exception as e:
#         LOG.exception("Error inspecting transformers_: %s", e)
#         # fallback: treat columns containing 'url' as text, others numeric
#         for c in all_cols:
#             if "url" in c.lower() or "domain" in c.lower() or "host" in c.lower():
#                 text_cols.add(c)
#             else:
#                 numeric_cols.add(c)

#     # Build row with correct types:
#     row = {}
#     for c in all_cols:
#         if c in text_cols:
#             # If this is the URL column, put the actual url. Otherwise empty string
#             row[c] = url if "url" in c.lower() else ""
#         elif c in numeric_cols:
#             row[c] = 0
#         else:
#             # unknown column: safe fallback - numeric 0
#             row[c] = 0

#     df = pd.DataFrame([row], columns=all_cols)
#     return df
# utils.py
# utils.py
# Lightweight helper: build DataFrame in exact order and safe dtypes.

# utils.py
# utils.py
# utils.py

from feature_engineering import compute_numeric_features


def extract_features(url: str):
    """
    EXACT FEATURE SCHEMA as expected by your saved ColumnTransformer.
    """

    # CRITICAL — Your training used FULL URL string inside TF-IDF
    domain_raw = url.strip()

    # compute numeric feature block
    numeric = compute_numeric_features(url)

    # ranking used in your training — constant 10,000,000
    ranking_value = 10_000_000

    features = {
        "domain": domain_raw,     # passed to TfidfVectorizer
        "ranking": ranking_value, # numeric column in training
    }

    # Merge numeric features
    features.update(numeric)

    return features
