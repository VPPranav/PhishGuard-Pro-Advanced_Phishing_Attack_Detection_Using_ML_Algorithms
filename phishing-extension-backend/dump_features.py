import joblib

model_path = "model/phishing_model.pkl"
art = joblib.load(model_path)

if isinstance(art, dict):
    model = art["model"]
else:
    model = art

# find ColumnTransformer
prep = None
for name, step in model.named_steps.items():
    if hasattr(step, "feature_names_in_"):
        prep = step
        break

if prep is None:
    raise RuntimeError("ColumnTransformer not found in pipeline.")

print("\n=== FEATURE NAMES USED FOR TRAINING ===\n")
for col in prep.feature_names_in_:
    print(col)
