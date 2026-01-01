from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.models import StockRequest, MLRequest, TARequest
import backend.service as service
import uvicorn

app = FastAPI(title="Stock Prediction API")

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only, restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Stock Prediction API is running"}

@app.post("/api/fetch_data")
def fetch_data(req: StockRequest):
    data = service.get_stock_data(req.ticker, req.start_date, req.end_date, req.interval)
    if "error" in data:
        raise HTTPException(status_code=400, detail=data["error"])
    return data

@app.post("/api/fundamentals")
def fundamentals(req: StockRequest):
    # Fundamentals might only need ticker, but for consistency we use StockRequest or we could make a specific one
    # service.get_fundamentals only needs ticker
    data = service.get_fundamentals(req.ticker)
    if "error" in data:
        raise HTTPException(status_code=400, detail=data["error"])
    return data

@app.post("/api/ml")
def machine_learning(req: MLRequest):
    data = service.run_ml_model(req.ticker, req.start_date, req.end_date, req.model)
    if "error" in data:
        raise HTTPException(status_code=400, detail=data["error"])
    return data

@app.post("/api/ta")
def technical_analysis(req: TARequest):
    data = service.run_ta(req.ticker, req.start_date, req.end_date, req.indicator)
    if "error" in data:
        raise HTTPException(status_code=400, detail=data["error"])
    return data

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
