# # main.py
# import logging
# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import joblib
# from utils import build_input_row
# import traceback

# LOG = logging.getLogger("uvicorn.error")

# app = FastAPI(title="Phishing Detector API")

# # Allow extension/browser requests (preflight)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],    # change to specific origin(s) for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load model artifact
# MODEL_PATH = "model/phishing_model.pkl"  # ensure this file exists

# try:
#     artifacts = joblib.load(MODEL_PATH)
#     model = artifacts.get("model", None) or artifacts  # support both dict or direct estimator
#     threshold = artifacts.get("best_threshold", 0.5) if isinstance(artifacts, dict) else 0.5
#     LOG.info("Model loaded from %s", MODEL_PATH)
# except Exception as e:
#     LOG.exception("Failed to load model during startup: %s", e)
#     # If model fails to load, keep running but endpoints will return error
#     model = None
#     threshold = 0.5


# @app.get("/")
# def root():
#     return {"status": "Phishing detector backend running"}


# @app.post("/predict")
# async def predict(request: Request):
#     """
#     Expects JSON: {"url": "https://example.com"}
#     Returns: {"url":..., "probability": float, "verdict": "phishing"|"safe"}
#     """
#     if model is None:
#         return JSONResponse(status_code=500, content={"error": "Model not loaded on server"})

#     try:
#         body = await request.json()
#         url = body.get("url", None)
#         if not url:
#             return JSONResponse(status_code=400, content={"error": "Missing 'url' in request body"})

#         # build a DataFrame row correctly typed (uses model internal ColumnTransformer)
#         X = build_input_row(url, model)

#         # predict (run in same thread - scikit-learn is not async)
#         proba = float(model.predict_proba(X)[0][1])
#         verdict = "phishing" if proba >= threshold else "safe"

#         return {"url": url, "probability": proba, "verdict": verdict}

#     except Exception as exc:
#         # Log full traceback for debugging
#         LOG.error("Exception during /predict: %s", exc)
#         LOG.error(traceback.format_exc())
#         return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})
# main.py
# main.py
# main.py
# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from utils import extract_features

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model/phishing_model.pkl"
loaded = joblib.load(MODEL_PATH)
model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
print("✔ Model loaded successfully.")


@app.post("/predict")
def predict(payload: dict):
    try:
        url = payload.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="URL missing")

        feats = extract_features(url)
        X = pd.DataFrame([feats])
        proba = float(model.predict_proba(X)[0][1])   # probability of phishing

        # verdict logic
        if proba < 0.001:
            verdict = "legitimate"
        elif proba > 0.5:
            verdict = "phishing"
        else:
            verdict = "suspicious"

        # confidence score = probability matching the verdict
        confidence_score = (
            1 - proba if verdict == "legitimate"
            else proba if verdict == "phishing"
            else abs(0.5 - proba) * 2    # suspicious confidence scaling
        )

        return {
            "url": url,
            "verdict": verdict,
            "confidence_score": round(confidence_score * 100, 2)  # in %
        }

    except Exception as e:
        print("❌ Prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))
