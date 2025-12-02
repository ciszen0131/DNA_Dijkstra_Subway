# France Government Debt Policy Analysis and Prediction Model

A comprehensive analysis and prediction framework to examine the impact of French government debt policies by comparing France's financial indicators with Portugal and Ireland.

## 📋 Overview

This project analyzes and predicts the effects of government debt policy changes using data from IMF, OECD, and bond yields. It leverages advanced time series models (ARIMA, SARIMAX, LSTM) and policy simulation techniques to provide actionable insights for policy decision-making.

### Key Features

- **Multi-country Comparison**: Compares France, Portugal, and Ireland economic indicators
- **Advanced Forecasting**: ARIMA, SARIMAX, and LSTM models for time series prediction
- **Policy Simulation**: Monte Carlo simulations for uncertainty analysis
- **Interactive Visualizations**: Comprehensive dashboards for economic indicator comparison

## 📁 Repository Structure

```
├── data/                          # Data storage directory
├── notebooks/                     # Jupyter notebooks for analysis
│   └── analysis_notebook.ipynb    # Main analysis notebook
├── src/                           # Source code
│   ├── __init__.py
│   ├── data_preparation.py        # Data loading and preprocessing
│   ├── visualization.py           # Visualization utilities
│   └── models/                    # Predictive models
│       ├── __init__.py
│       ├── time_series_models.py  # ARIMA and SARIMAX implementations
│       ├── lstm_model.py          # LSTM neural network models
│       └── policy_simulation.py   # Policy simulation framework
├── requirements.txt               # Python dependencies
├── 프랑스_정부_부채_수준_예측_및_정책_효과_분석.ipynb  # Original analysis notebook
└── README.md                      # This file
```

## 🔑 Key Economic Indicators

The analysis focuses on the following indicators:

1. **Government Fiscal Balance** (% of GDP)
2. **Unemployment Rates** (%)
3. **10-Year Bond Yield Rates** (%)
4. **Government Debt-to-GDP Ratio** (%)
5. **Real GDP Growth Rate** (%)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ciszen0131/DNA_Dijkstra_Subway.git
   cd DNA_Dijkstra_Subway
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 📊 Data Sources

- **IMF World Economic Outlook**: Fiscal balance, debt-to-GDP ratios
- **OECD Economic Indicators**: Unemployment rates, GDP growth
- **Bond Yields Data**: 10-year government bond yields

## 🚀 Usage

### 1. Data Preparation

```python
from src.data_preparation import DataLoader, DataPreprocessor

# Load data
loader = DataLoader()
data = loader.create_sample_data()  # Or load from files

# Preprocess data
preprocessor = DataPreprocessor()
clean_data = preprocessor.handle_missing_values(data['fiscal'])
```

### 2. Visualization

```python
from src.visualization import EconomicVisualizer

visualizer = EconomicVisualizer()

# Create comprehensive dashboard
fig = visualizer.create_dashboard(data, save_path='dashboard.png')

# Plot specific comparisons
visualizer.plot_bond_yields_comparison(data['bond_yields'])
visualizer.plot_fiscal_balance_trends(data['fiscal'])
```

### 3. Time Series Forecasting

#### ARIMA Model
```python
from src.models import ARIMAModel

# Initialize and fit ARIMA model
arima = ARIMAModel(order=(1, 1, 1))
arima.fit(france_fiscal_series, auto_order=True)

# Generate forecast
forecast = arima.predict(steps=24)
lower, upper = arima.get_confidence_interval(steps=24)
```

#### SARIMAX Model (with seasonality)
```python
from src.models import SARIMAXModel

# Initialize SARIMAX with seasonal components
sarimax = SARIMAXModel(
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)
sarimax.fit(monthly_series, exog=policy_indicators)

# Forecast with exogenous variables
forecast = sarimax.predict(steps=24, exog_forecast=future_policy)
```

#### LSTM Neural Network
```python
from src.models import LSTMPredictor

# Train LSTM model
lstm = LSTMPredictor(
    sequence_length=12,
    n_units=[64, 32],
    dropout_rate=0.2
)
lstm.fit(series, epochs=100, batch_size=32)

# Generate predictions
forecast = lstm.predict(steps=24)
```

### 4. Policy Simulation

```python
from src.models import PolicySimulator

simulator = PolicySimulator()

# Define policy scenarios
scenarios = [
    {'name': 'fiscal_consolidation', 'policy_type': 'fiscal_consolidation', 'intensity': 1.0},
    {'name': 'structural_reforms', 'policy_type': 'structural_reforms', 'intensity': 1.5},
]

# Run simulations
results = simulator.run_policy_scenarios(base_series, scenarios, periods=24)

# Generate policy report
report = simulator.generate_policy_report()
print(report)
```

## 📈 Model Selection Rationale

| Model | Use Case | Strengths |
|-------|----------|-----------|
| **ARIMA** | Short-term forecasting of stationary series | Simple, interpretable, works well with small datasets |
| **SARIMAX** | Seasonal data with external factors | Captures seasonality and policy intervention effects |
| **LSTM** | Complex non-linear patterns | Captures long-term dependencies, handles multivariate data |
| **Monte Carlo** | Uncertainty quantification | Provides probability distributions of outcomes |

## 📝 Policy Insights

### Key Findings

1. **Fiscal Consolidation Effects**:
   - Gradual debt reduction of 1-2% annually is achievable
   - Short-term unemployment impact typically recovers within 2-3 years

2. **Benchmark Comparison**:
   - Ireland's recovery demonstrates effectiveness of structural reforms
   - Portugal shows sustainable debt reduction path through fiscal discipline

3. **Post-Pandemic Considerations**:
   - Initial debt spike expected but recoverable
   - Structural reforms enhance long-term sustainability

### Recommendations

1. Implement measured fiscal consolidation (0.5-1% GDP annually)
2. Combine with targeted structural reforms
3. Maintain flexibility for economic shocks
4. Use Portugal/Ireland experiences as policy benchmarks

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- IMF World Economic Outlook Database
- OECD Economic Outlook
- European Central Bank Statistics
- Blanchard, O., & Leigh, D. (2013). "Growth Forecast Errors and Fiscal Multipliers"

---

## Additional Files

### Original Subway Navigation (DNA.py)

The repository also contains `DNA.py`, an implementation of Dijkstra's algorithm for Seoul subway navigation between Lines 1 and 2