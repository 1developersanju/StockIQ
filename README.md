# Stock Market Dashboard

A modern, mobile-first dashboard for stock market analysis. Built with a decoupled architecture featuring a FastAPI backend and a high-performance Vanilla JS frontend.

## 🚀 Features

- **Market Overview**: Interactive Price History with Line, Area, and Candlestick modes.
- **Advanced Controls**: Dynamic Date Range presets (1M, 3M, 6M, 1Y, 5Y) and Custom Intervals.
- **Technical Overlays**: Real-time Moving Average (SMA 20, 50, 200) calculations.
- **Machine Learning**: Prophet time-series forecasting and Decision Tree classifiers.
- **Technical Analysis**: EMA, RSI, MACD, and Bollinger Bands indicators.
- **Responsive Design**: Polished SaaS-style UI optimized for all devices.

## 🏗️ Tech Stack

- **Backend**: FastAPI
- **Frontend**: Vanilla HTML/JS/CSS, Plotly.js
- **Data**: yfinance API

---

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r backend/requirements.txt
```

### 2. Run the Application

You need two terminals running simultaneously:

#### Terminal 1: Backend
```bash
python3 -m uvicorn backend.main:app --reload
```

#### Terminal 2: Frontend
```bash
python3 -m http.server -d frontend 8080
```
Visit `http://localhost:8080` to view the dashboard.

---

## 📁 Project Structure
- `backend/`: API logic, data services, and ML models.
- `frontend/`: UI components, styles, and dashboard logic.
- `README.md`: Project documentation.
# StockIQ
