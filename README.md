# 📈 Time Series Forecasting of 911 Emergency Calls Using Facebook Prophet

A portfolio time series forecasting project using **Facebook Prophet** to analyze and forecast emergency 911 call volume trends.  
This notebook demonstrates real-world forecasting techniques on timestamped incident data from Montgomery County, PA.

Use this link to access the notebook in Google Colab:  
[Open Colab Notebook](https://github.com/yourusername/911-call-forecasting-prophet/blob/main/annotated_911_prophet_notebook.ipynb)

[Prophet](https://facebook.github.io/prophet/) is an open-source time series forecasting tool developed by Facebook's Core Data Science team.  
It is specifically designed for:

- **Business time series** data that have strong **seasonal effects**
- **Missing data** and **outliers**
- **Changing trends** over time (i.e., changepoints)

Prophet is favored because it provides:
- **High interpretability** (you can see trend, yearly/weekly seasonality)
- **Automatic handling** of missing values and holidays
- **Minimal configuration**, making it ideal for analysts and developers
- 
Prophet is an **additive model**, where the forecast is built from multiple components y(t) = g(t) + s(t) + h(t) + εt
Where:
- `g(t)` is the **trend** function modeling non-periodic changes
- `s(t)` captures **seasonal effects** (weekly, yearly)
- `h(t)` models effects of **holidays**
- `εt` is the **error term** (noise)
Prophet uses a combination of:
- **Piecewise linear or logistic growth** for trend (`g(t)`)
- **Fourier series** to model seasonality (`s(t)`)
- **Dummy variables** for user-defined holidays (`h(t)`)
Unlike ARIMA or SARIMA, which require manual differencing and ACF/PACF tuning, Prophet lets the model **learn patterns from data with minimal tuning**.

### 💡 Why Prophet for This Project?

- 📅 911 calls exhibit **daily, weekly, and yearly seasonality**
- ⚠️ Data may have **missing days or outliers** (holidays, weather events)
- 🔧 Prophet allows us to **add holidays** (e.g., New Year, Christmas)
- 📊 We want **interpretable components** to explain spikes in call volume

This makes Prophet a perfect fit for the task — enabling us to forecast and **interpret** emergency call trends with confidence.


## 📌 Project Overview
This project presents a full time series analysis and forecasting pipeline applied to **911 emergency call data** from Montgomery County, Pennsylvania. Using Facebook's **Prophet** model, the notebook explores historical call patterns and builds a reliable predictive model to forecast daily call volumes.
The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/mchirico/montcoalert), originally uploaded by Mike Chirico. Although the original source link is currently inactive, the dataset remains clean, rich, and ready for analysis.

### 🎯 Objectives
- Analyze 911 emergency call data to uncover daily, weekly, and yearly patterns.
- Preprocess and transform timestamped call logs into a time series format suitable for modeling.
- Apply the **Prophet** model to forecast future call volumes.
- Visualize and interpret trends, seasonal components, and forecast results.
- Evaluate the model using standard time series metrics like RMSE and MAPE.

### 📦 Dataset Summary
- **Rows**: ~100,000+ emergency call records  
- **Key Columns**:
  - `timeStamp`: timestamp of the call (used as time index)
  - `title`: call type (e.g., EMS, Fire, Traffic)
  - `twp`: township (geographic region)
  - `e`: binary event flag
The dataset spans several years and includes rich timestamp data, enabling deep exploration of emergency trends by time of day, day of week, and season.

### 🔍 Why This Project?
Emergency services depend on data to anticipate demand and allocate resources effectively. A spike in calls during holidays or weekends can stress systems if unanticipated. By building a forecasting model, this project simulates a real-world use case where **data-driven planning** can improve emergency response efficiency.
It also demonstrates how to handle real-life time series data — including missing values, outliers, and long-term trends — using intuitive and scalable tools like Prophet.

### 🛠️ Tools & Libraries Used
- **Pandas / NumPy** – for data manipulation
- **Seaborn / Matplotlib** – for exploratory visualizations
- **Prophet** – for forecasting model
- **Sklearn** – for RMSE, MAPE, and error metrics
This project makes extensive use of **Prophet's ability** to decompose the time series into interpretable components (trend, yearly/weekly seasonality, holidays), and provides high-quality plots with minimal effort.
