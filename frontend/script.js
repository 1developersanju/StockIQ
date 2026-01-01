const API_URL = "http://localhost:8000/api";

// --- UI Elements ---
const el = {
    // Layout
    sidebar: document.getElementById('app-sidebar'),
    overlay: document.getElementById('drawer-overlay'),
    openDrawer: document.getElementById('open-drawer'),
    closeDrawer: document.getElementById('close-drawer'),
    runBtn: document.getElementById('runBtn'),

    // Config Inputs
    ticker: document.getElementById('ticker'),
    startDate: document.getElementById('startDate'),
    endDate: document.getElementById('endDate'),
    mlModel: document.getElementById('mlModel'),
    taIndicator: document.getElementById('taIndicator'),
    interval: document.getElementById('interval'),
    showVolume: document.getElementById('showVolume'),
    chartType: document.getElementById('chartType'),
    sma20: document.getElementById('sma20'),
    sma50: document.getElementById('sma50'),
    sma200: document.getElementById('sma200'),
    presetBtns: document.querySelectorAll('.preset-btn'),

    // Config Sections
    homeSpecific: document.getElementById('home-specific-config'),
    mlSpecific: document.getElementById('ml-specific-config'),
    taSpecific: document.getElementById('ta-specific-config'),

    // Nav Items
    navItems: document.querySelectorAll('.nav-item'),

    // Views
    emptyState: document.getElementById('empty-state'),
    resultsArea: document.getElementById('results-area'),
    viewHome: document.getElementById('view-Home'),
    viewML: document.getElementById('view-ML'),
    viewTA: document.getElementById('view-TA'),

    // Dynamic Display
    displayTicker: document.getElementById('display-ticker'),
    mlDisplayTicker: document.getElementById('ml-display-ticker'),
    mlDisplayModel: document.getElementById('ml-display-model'),
    taDisplayTicker: document.getElementById('ta-display-ticker'),
    taDisplayIndicator: document.getElementById('ta-display-indicator'),

    // Detailed Content
    fundamentals: document.getElementById('fundamentals-container'),
    mlDescription: document.getElementById('ml-description'),
    mlMetrics: document.getElementById('ml-metrics'),
    taDescription: document.getElementById('ta-description'),

    // Feedback
    errorMsg: document.getElementById('error-msg'),
    errorText: document.getElementById('error-text'),
    loader: document.querySelector('.loader')
};

// --- App State ---
let currentView = 'Home';
let lastPriceData = null;

// --- Initialization ---
function init() {
    // Default Dates
    if (!el.endDate.value) {
        el.endDate.valueAsDate = new Date();
    }

    // Event Listeners
    el.openDrawer.addEventListener('click', () => toggleDrawer(true));
    el.closeDrawer.addEventListener('click', () => toggleDrawer(false));
    el.overlay.addEventListener('click', () => toggleDrawer(false));

    el.navItems.forEach(item => {
        item.addEventListener('click', (e) => switchView(e.currentTarget.dataset.view));
    });

    el.runBtn.addEventListener('click', runAnalysis);
    el.ticker.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') runAnalysis();
    });

    // Instant Chart Updates
    [el.chartType, el.sma20, el.sma50, el.sma200, el.showVolume].forEach(control => {
        control.addEventListener('change', () => {
            if (lastPriceData && currentView === 'Home') {
                renderOverviewCharts(el.ticker.value, lastPriceData);
            }
        });
    });

    el.presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            el.presetBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            updateDatesFromPreset(btn.dataset.range);
        });
    });

    // Initial Date Preset
    updateDatesFromPreset('1Y');

    // Handle Window Resize for Charts
    window.addEventListener('resize', () => {
        const charts = document.querySelectorAll('.js-plotly-plot');
        charts.forEach(chart => Plotly.Plots.resize(chart));
    });
}

// --- Date Helpers ---
function updateDatesFromPreset(range) {
    const end = new Date();
    const start = new Date();

    switch (range) {
        case '1M': start.setMonth(end.getMonth() - 1); break;
        case '3M': start.setMonth(end.getMonth() - 3); break;
        case '6M': start.setMonth(end.getMonth() - 6); break;
        case '1Y': start.setFullYear(end.getFullYear() - 1); break;
        case '5Y': start.setFullYear(end.getFullYear() - 5); break;
    }

    el.startDate.valueAsDate = start;
    el.endDate.valueAsDate = end;
}

// --- UI Logic ---
function toggleDrawer(open) {
    if (open) {
        el.sidebar.classList.add('open');
        el.overlay.classList.add('visible');
    } else {
        el.sidebar.classList.remove('open');
        el.overlay.classList.remove('visible');
    }
}

function switchView(viewName) {
    currentView = viewName;

    // Update Nav Active State
    el.navItems.forEach(item => {
        const isActive = item.dataset.view === viewName;
        item.classList.toggle('active', isActive);
    });

    // Update Specific Config Visibility
    el.homeSpecific.classList.toggle('hidden', viewName !== 'Home');
    el.mlSpecific.classList.toggle('hidden', viewName !== 'Machine Learning');
    el.taSpecific.classList.toggle('hidden', viewName !== 'Technical Analysis');

    // On mobile, close drawer after selection if it's not a config change
    if (window.innerWidth < 1024) {
        // toggleDrawer(false); // We actually want them to click "Run" after selection maybe?
    }
}

function setLoading(isLoading) {
    el.runBtn.disabled = isLoading;
    el.loader.classList.toggle('hidden', !isLoading);
    const label = el.runBtn.querySelector('.btn-label');
    if (isLoading) {
        label.style.opacity = '0.5';
        label.innerText = 'Analyzing...';
    } else {
        label.style.opacity = '1';
        label.innerText = 'Run Analysis';
    }
}

function showError(msg) {
    if (msg) {
        el.errorText.innerText = msg;
        el.errorMsg.classList.remove('hidden');
    } else {
        el.errorMsg.classList.add('hidden');
    }
}

// --- Analysis Logic ---
async function runAnalysis() {
    showError(null);
    setLoading(true);

    const ticker = el.ticker.value.toUpperCase();
    const start = el.startDate.value;
    const end = el.endDate.value;

    if (!ticker) {
        showError("Invalid ticker symbol");
        setLoading(false);
        return;
    }

    try {
        // Prepare UI for results
        el.emptyState.classList.add('hidden');
        el.resultsArea.classList.remove('hidden');

        // Hide all views first
        [el.viewHome, el.viewML, el.viewTA].forEach(v => v.classList.add('hidden'));

        if (currentView === 'Home') {
            await handleHomeRequest(ticker, start, end);
        } else if (currentView === 'Machine Learning') {
            await handleMLRequest(ticker, start, end);
        } else if (currentView === 'Technical Analysis') {
            await handleTARequest(ticker, start, end);
        }

        // Auto-close drawer on mobile results
        if (window.innerWidth < 1024) {
            toggleDrawer(false);
        }
    } catch (err) {
        console.error(err);
        showError(err.message || "An error occurred during analysis.");
    } finally {
        setLoading(false);
    }
}

// --- API Request Handlers ---

async function handleHomeRequest(ticker, start, end) {
    el.viewHome.classList.remove('hidden');
    el.displayTicker.innerText = ticker;
    const interval = el.interval ? el.interval.value : '1d';

    // 1. Fetch Price Data
    const priceData = await fetchAPI('/fetch_data', { ticker, start_date: start, end_date: end, interval });
    lastPriceData = priceData;
    renderOverviewCharts(ticker, priceData);

    // 2. Fetch Fundamentals
    try {
        const fundData = await fetchAPI('/fundamentals', { ticker, start_date: start, end_date: end });
        renderFundamentals(fundData);
    } catch (e) {
        el.fundamentals.innerHTML = `<p class="error-text">Fundamental data metrics unavailable for this ticker.</p>`;
    }
}

async function handleMLRequest(ticker, start, end) {
    el.viewML.classList.remove('hidden');
    const model = el.mlModel.value;

    el.mlDisplayTicker.innerText = ticker;
    el.mlDisplayModel.innerText = model;
    el.mlDescription.innerText = getModelDescription(model);

    const data = await fetchAPI('/ml', { ticker, start_date: start, end_date: end, model });
    renderMLResults(data);
}

async function handleTARequest(ticker, start, end) {
    el.viewTA.classList.remove('hidden');
    const indicator = el.taIndicator.value;

    el.taDisplayTicker.innerText = ticker;
    el.taDisplayIndicator.innerText = indicator;
    el.taDescription.innerText = getIndicatorDescription(indicator);

    const data = await fetchAPI('/ta', { ticker, start_date: start, end_date: end, indicator });
    renderTAChart(indicator, data);
}

// --- API Helper ---
async function fetchAPI(endpoint, body) {
    const res = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });

    if (!res.ok) {
        const errorDetail = await res.json().catch(() => ({ detail: "API Error" }));
        throw new Error(errorDetail.detail || `Server error: ${res.status}`);
    }
    return res.json();
}

// --- Chart Rendering ---

function getChartTheme() {
    return {
        font: { family: 'Inter, sans-serif', color: '#6B7280', size: 12 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 20, r: 20, b: 40, l: 50 },
        xaxis: { gridcolor: '#F3F4F6', zeroline: false },
        yaxis: { gridcolor: '#F3F4F6', zeroline: false }
    };
}

function renderOverviewCharts(ticker, data) {
    const chartType = el.chartType ? el.chartType.value : 'Candlestick';
    const traces = [];

    // 1. Price Trace
    if (chartType === 'Candlestick') {
        traces.push({
            x: data.dates,
            open: data.open,
            high: data.high,
            low: data.low,
            close: data.close,
            type: 'candlestick',
            increasing: { line: { color: '#10B981' } },
            decreasing: { line: { color: '#EF4444' } },
            name: ticker
        });
    } else {
        traces.push({
            x: data.dates,
            y: data.close,
            type: 'scatter',
            mode: 'lines',
            fill: chartType === 'Area' ? 'tozeroy' : 'none',
            name: 'Price',
            line: { color: '#6366F1', width: 2 }
        });
    }

    // 2. Overlays
    if (el.sma20 && el.sma20.checked) traces.push(calculateSMATrace(data, 20, '#F59E0B'));
    if (el.sma50 && el.sma50.checked) traces.push(calculateSMATrace(data, 50, '#10B981'));
    if (el.sma200 && el.sma200.checked) traces.push(calculateSMATrace(data, 200, '#6366F1'));

    Plotly.newPlot('price-chart', traces, {
        ...getChartTheme(),
        yaxis: { ...getChartTheme().yaxis, title: 'Price (USD)' },
        xaxis: { ...getChartTheme().xaxis, rangeslider: { visible: false } }
    }, { responsive: true, displayModeBar: false });

    // 2. Volume Chart (optional)
    const volumeContainer = document.getElementById('volume-chart-card');
    if (el.showVolume && el.showVolume.checked) {
        if (!volumeContainer) {
            const newCard = document.createElement('div');
            newCard.id = 'volume-chart-card';
            newCard.className = 'card full-width';
            newCard.innerHTML = `
                <div class="card-header"><h3>Trading Volume</h3></div>
                <div id="volume-chart" class="chart-container"></div>
            `;
            // Insert after price chart
            document.getElementById('price-chart').closest('.card').after(newCard);
        }

        const volumeColors = data.close.map((c, i) => c >= data.open[i] ? '#10B981' : '#EF4444');

        const volumeTrace = {
            x: data.dates,
            y: data.volume,
            type: 'bar',
            name: 'Volume',
            marker: { color: volumeColors, opacity: 0.6 }
        };

        Plotly.newPlot('volume-chart', [volumeTrace], {
            ...getChartTheme(),
            yaxis: { ...getChartTheme().yaxis, title: 'Volume' }
        }, { responsive: true, displayModeBar: false });
    } else if (volumeContainer) {
        volumeContainer.remove();
    }
}

function renderFundamentals(data) {
    el.fundamentals.innerHTML = '';
    const items = [
        { label: "P/E Ratio", key: "pe_ratio" },
        { label: "ROE", key: "roe" },
        { label: "P/B Ratio", key: "pb_ratio" },
        { label: "Debt/Equity", key: "debt_to_equity" },
        { label: "Current Ratio", key: "current_ratio" },
        { label: "ROA", key: "return_on_assets" },
        { label: "P/S Ratio", key: "price_to_sales" },
        { label: "Dividend Yield", key: "dividend_yield" }
    ];

    items.forEach(item => {
        let val = data[item.key];
        if (val === undefined || val === null) val = 'N/A';

        const div = document.createElement('div');
        div.className = 'fundamental-item';
        div.innerHTML = `
            <span class="fundamental-label">${item.label}</span>
            <span class="fundamental-value">${val}</span>
        `;
        el.fundamentals.appendChild(div);
    });
}

function renderMLResults(data) {
    // 1. Forecast Chart
    let traces = [];
    if (data.model === 'LSTM') {
        traces.push({ x: data.train_dates, y: data.train_close, name: 'Training', line: { color: '#D1D5DB' } });
        traces.push({ x: data.test_dates, y: data.test_close, name: 'Actual', line: { color: '#374151' } });
        traces.push({ x: data.test_dates, y: data.test_predictions, name: 'Predicted', line: { color: '#6366F1', width: 2 } });

        renderMetrics([
            { label: 'RMSE', value: data.rmse.toFixed(4) },
            { label: 'MAPE %', value: data.mape.toFixed(2) }
        ]);
    } else {
        const trainY = data.model === 'Prophet' ? data.train_y : data.train_close;
        const validY = data.model === 'Prophet' ? data.valid_y : data.valid_close;

        traces.push({ x: data.train_dates, y: trainY, name: 'Historical', line: { color: '#D1D5DB' } });
        traces.push({ x: data.valid_dates, y: validY, name: 'Actual', line: { color: '#374151' } });
        traces.push({ x: data.valid_dates, y: data.valid_predictions, name: 'Forecasted', line: { color: '#6366F1', width: 2 } });

        const metricLabel = data.model === 'Prophet' ? 'RMSE' : 'Score';
        const metricVal = data.model === 'Prophet' ? data.rms.toFixed(4) : data.score.toFixed(4);
        renderMetrics([{ label: metricLabel, value: metricVal }]);
    }

    Plotly.newPlot('ml-chart', traces, getChartTheme(), { responsive: true, displayModeBar: false });
}

function renderTAChart(indicator, data) {
    const traces = [{
        x: data.dates,
        y: data.close,
        name: 'Price',
        line: { color: '#D1D5DB', width: 1 }
    }];

    if (indicator === 'EMA') {
        traces.push({ x: data.dates, y: data.short_ema, name: 'Short EMA', line: { color: '#6366F1' } });
        traces.push({ x: data.dates, y: data.mid_ema, name: 'Mid EMA', line: { color: '#10B981' } });
        traces.push({ x: data.dates, y: data.long_ema, name: 'Long EMA', line: { color: '#F59E0B' } });
    } else if (indicator === 'RSI') {
        traces.push({ x: data.dates, y: data.rsi, name: 'RSI', line: { color: '#6366F1' } });
        // Add Overbought/Oversold lines
        const shapes = [
            { type: 'line', y0: 30, y1: 30, x0: data.dates[0], x1: data.dates[data.dates.length - 1], line: { color: '#EF4444', dash: 'dash' } },
            { type: 'line', y0: 70, y1: 70, x0: data.dates[0], x1: data.dates[data.dates.length - 1], line: { color: '#EF4444', dash: 'dash' } }
        ];
        Plotly.newPlot('ta-chart', traces, { ...getChartTheme(), shapes }, { responsive: true });
        return;
    } else if (indicator === 'MACD') {
        traces.push({ x: data.dates, y: data.macd, name: 'MACD', line: { color: '#6366F1' } });
        traces.push({ x: data.dates, y: data.signal, name: 'Signal', line: { color: '#F59E0B' } });
    } else if (indicator === 'Bollinger Band') {
        traces.push({ x: data.dates, y: data.sma, name: 'SMA', line: { color: '#9CA3AF' } });
        traces.push({ x: data.dates, y: data.upper, name: 'Upper', line: { color: '#10B981', dash: 'dot' } });
        traces.push({ x: data.dates, y: data.lower, name: 'Lower', line: { color: '#EF4444', dash: 'dot' } });
    }

    Plotly.newPlot('ta-chart', traces, getChartTheme(), { responsive: true, displayModeBar: false });
}

function renderMetrics(metrics) {
    el.mlMetrics.innerHTML = '';
    metrics.forEach(m => {
        const div = document.createElement('div');
        div.className = 'metric-item';
        div.innerHTML = `<span class="metric-label">${m.label}</span><span class="metric-value">${m.value}</span>`;
        el.mlMetrics.appendChild(div);
    });
}

// --- Text Helpers ---
function getModelDescription(model) {
    const descriptions = {
        'LSTM': "Long Short-Term Memory (LSTM) is a recurrent neural network that learns from sequences of price data to capture long-term dependencies in market trends.",
        'Tree Classifier': "Decision Tree regression splits stock data into nested decision nodes to approximate price movements based on historical feature correlations.",
        'Prophet': "Prophet is an additive model that decomposes time-series data into trend, seasonality, and holiday effects to predict future price ranges."
    };
    return descriptions[model] || "";
}

function getIndicatorDescription(indicator) {
    const descriptions = {
        'EMA': "The Exponential Moving Average (EMA) tracks the trend by weighting recent prices more heavily, making it more responsive to new market info.",
        'RSI': "The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements, identifying overbought or oversold conditions.",
        'MACD': "Moving Average Convergence Divergence (MACD) shows the relationship between two moving averages, signaling strength, direction, and momentum.",
        'Bollinger Band': "Bollinger Bands represent volatility using standard deviations above and below a central moving average, indicating potential price breakouts."
    };
    return descriptions[indicator] || "";
}

function calculateSMATrace(data, window, color) {
    const prices = data.close;
    const sma = [];
    for (let i = 0; i < prices.length; i++) {
        if (i < window - 1) {
            sma.push(null);
        } else {
            const slice = prices.slice(i - window + 1, i + 1);
            const avg = slice.reduce((a, b) => a + b, 0) / window;
            sma.push(avg);
        }
    }
    return {
        x: data.dates,
        y: sma,
        name: `SMA ${window}`,
        line: { color, width: 1.5, dash: 'dashdot' }
    };
}

// Start the app
init();
