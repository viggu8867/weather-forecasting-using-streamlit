
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

# Set page config
st.set_page_config(page_title="Weather Forecasting Dashboard", layout="wide")

# Load external CSS
css_path = ".streamlit/style.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("CSS file not found. Using default styling.")

# Title and summary metrics
st.title("🌦️ Weather Forecasting Dashboard")
st.write("Explore historical weather data and forecast future temperatures.")
month_columns = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# Load default dataset
try:
    df = pd.read_csv("weather_dataset.csv")
except FileNotFoundError:
    st.error("Default dataset 'weather_dataset.csv' not found. Please upload a CSV file.")
    df = None

# Sidebar for file upload and month selection
with st.sidebar:
    st.header("Data & Settings")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    selected_months = st.multiselect("Select Months for Analysis", month_columns, default=['JAN'])

# Load uploaded dataset if provided
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

# Data preprocessing
if df is not None:
    # Check for missing values
    if df.isnull().any().any():
        st.error("Missing values detected! Imputing with mean...")
        df = df.fillna(df.mean(numeric_only=True))

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    avg_temp = df[month_columns].mean().mean()
    min_year = df['YEAR'][df[month_columns].mean(axis=1).idxmin()]
    max_year = df['YEAR'][df[month_columns].mean(axis=1).idxmax()]
    with col1:
        st.metric("Average Temperature", f"{avg_temp:.2f} °C")
    with col2:
        st.metric("Coldest Year", min_year)
    with col3:
        st.metric("Warmest Year", max_year)

    # Transform data for modeling
    df1 = pd.melt(df, id_vars='YEAR', value_vars=month_columns, var_name='Month', value_name='Temprature')
    df1['Date'] = pd.to_datetime(df1['Month'] + ' ' + df1['YEAR'].astype(str), format='%b %Y')
    df1['Trend'] = np.arange(len(df1))
    df1['Prev_Temp'] = df1['Temprature'].shift(1).fillna(df1['Temprature'].mean())
    df1 = pd.get_dummies(df1, columns=['Month'])

    # Train Random Forest model
    X = df1.drop(columns=['Temprature', 'Date'])
    y = df1['Temprature']
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(train_x, train_y)
    test_y_pred = rf.predict(test_x)

    # Save model
    joblib.dump(rf, 'weather_model.pkl')

    # Tabs for navigation
    tabs = st.tabs(["📊 Data Overview", "📈 Visualizations", "📉 Statistics", "🔮 Forecast", "📅 Time Series"])

    with tabs[0]:
        st.subheader("Dataset Preview")
        st.write(df.head())
        st.write(f"Shape: {df.shape}")

    with tabs[1]:
        st.subheader("Visualizations")
        col1, col2 = st.columns([1, 2])
        with col1:
            plot_type = st.selectbox("Select Plot Type", ["Histogram", "Boxplot", "Scatterplot"])
        with col2:
            if plot_type == "Histogram":
                fig = go.Figure()
                for month in selected_months:
                    fig.add_trace(go.Histogram(x=df[month], name=month, opacity=0.5))
                fig.update_layout(title="Histogram of Selected Months", xaxis_title="Temperature (°C)", yaxis_title="Count", barmode='overlay')
                st.plotly_chart(fig, alt="Histogram of temperatures for selected months")
                if st.button("Download Histogram"):
                    fig.write_image("histogram.png")
                    with open("histogram.png", "rb") as file:
                        st.download_button(label="Download Image", data=file, file_name="histogram.png", mime="image/png")
            elif plot_type == "Boxplot":
                fig = go.Figure()
                for month in selected_months:
                    fig.add_trace(go.Box(y=df[month], name=month))
                fig.update_layout(title="Boxplot of Selected Months", yaxis_title="Temperature (°C)")
                st.plotly_chart(fig, alt="Boxplot of temperatures for selected months")
                if st.button("Download Boxplot"):
                    fig.write_image("boxplot.png")
                    with open("boxplot.png", "rb") as file:
                        st.download_button(label="Download Image", data=file, file_name="boxplot.png", mime="image/png")
            elif plot_type == "Scatterplot":
                fig = go.Figure()
                for month in selected_months:
                    fig.add_trace(go.Scatter(x=df['YEAR'], y=df[month], mode='markers', name=month))
                fig.update_layout(title="Scatterplot of Selected Months vs Year", xaxis_title="Year", yaxis_title="Temperature (°C)")
                st.plotly_chart(fig, alt="Scatterplot of temperatures vs year for selected months")
                if st.button("Download Scatterplot"):
                    fig.write_image("scatterplot.png")
                    with open("scatterplot.png", "rb") as file:
                        st.download_button(label="Download Image", data=file, file_name="scatterplot.png", mime="image/png")

    with tabs[2]:
        st.subheader("Descriptive Statistics")
        with st.expander("View Statistics"):
            st.write(df.describe())
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df[month_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    with tabs[3]:
        st.subheader("🔮 Temperature Forecast")
        forecast_year = st.number_input("Enter Year to Forecast", min_value=2018, max_value=2050, value=2018)
        if forecast_year < 2018:
            st.error("Please select a year after 2017.")
        elif st.button("Predict"):
            with st.spinner("Generating Forecast..."):
                model = joblib.load('weather_model.pkl')
                future_data = []
                for year in range(2018, forecast_year + 1):
                    temp_df = pd.DataFrame({'YEAR': [year]*12, 'Month': month_columns})
                    temp_df['Trend'] = np.arange(len(df1), len(df1) + 12)
                    temp_df['Prev_Temp'] = df1['Temprature'].iloc[-12:].values
                    temp_df = pd.get_dummies(temp_df, columns=['Month'])
                    # Ensure columns match training data
                    for col in X.columns:
                        if col not in temp_df.columns:
                            temp_df[col] = 0
                    temp_df = temp_df[X.columns]  # Reorder columns
                    pred = model.predict(temp_df)
                    future_data.append(pd.DataFrame({'Year': year, 'Month': month_columns, 'Temprature': pred}))
                future_df = pd.concat(future_data, ignore_index=True)
                st.success("Forecast generated successfully!")
                st.write(future_df)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=future_df['Year'], y=future_df['Temprature'], mode='lines+markers', name='Forecasted Temperature'))
                fig.update_layout(title=f"Temperature Forecast (2018-{forecast_year})", xaxis_title='Year', yaxis_title='Temperature (°C)')
                st.plotly_chart(fig, alt=f"Forecasted temperatures from 2018 to {forecast_year}")
                if st.button("Download Forecast Plot"):
                    fig.write_image("forecast.png")
                    with open("forecast.png", "rb") as file:
                        st.download_button(label="Download Image", data=file, file_name="forecast.png", mime="image/png")

    with tabs[4]:
        st.subheader("Time Series Analysis")
        year_range = st.slider("Select Year Range", min_value=int(df['YEAR'].min()), max_value=int(df['YEAR'].max()), value=(1901, 2017))
        filtered_df = df1[df1['YEAR'].between(year_range[0], year_range[1])]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Temprature'], mode='lines+markers', name='Historical Temperature'))
        z = np.polyfit(filtered_df['YEAR'], filtered_df['Temprature'], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(x=filtered_df['Date'], y=p(filtered_df['YEAR']), name='Trend Line', line=dict(dash='dash')))
        fig.update_layout(title='Historical Temperature with Trend', xaxis_title='Date', yaxis_title='Temperature (°C)')
        st.plotly_chart(fig, alt="Time-series plot of historical temperatures with trend line")
        if st.button("Download Time Series Plot"):
            fig.write_image("timeseries.png")
            with open("timeseries.png", "rb") as file:
                st.download_button(label="Download Image", data=file, file_name="timeseries.png", mime="image/png")
        st.subheader("Model Performance")
        st.write(f"R² Score: {r2_score(test_y, test_y_pred):.4f}")
        st.write(f"MAE: {mean_absolute_error(test_y, test_y_pred):.4f}")
        # Residual plot
        residuals = test_y - test_y_pred
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_y, y=residuals, mode='markers', name='Residuals'))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(title="Residual Plot", xaxis_title="Actual Temperature (°C)", yaxis_title="Residuals")
        st.plotly_chart(fig, alt="Residual plot of model predictions")
else:
    st.error("Please upload a valid CSV file to proceed.")
 
