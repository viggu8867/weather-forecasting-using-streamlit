import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Page config
st.set_page_config(page_title="🌤️ Weather Data Explorer", layout="wide")

# App title
st.title("🌤️ Weather Data Interactive Explorer")
st.caption("Upload your CSV, choose months to analyze, generate plots, and download processed data.")

# Sidebar - File Upload
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("📤 Upload CSV file", type="csv")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Load default or uploaded
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")
else:
    st.sidebar.info("ℹ️ Using default Weather.csv")
    df = pd.read_csv("Weather.csv")

# Identify month columns
month_columns = [col for col in df.columns if col.strip().lower() in ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']]

if not month_columns:
    st.error("❌ No month columns (Jan-Dec) found in your data!")
    st.stop()

# Sidebar - Month selection
selected_months = st.sidebar.multiselect(
    "📅 Choose month columns to analyze",
    month_columns,
    default=month_columns
)

if not selected_months:
    st.warning("⚠️ Please select at least one month column in the sidebar to see results!")
    st.stop()

# Sidebar - Plot type
plot_type = st.sidebar.selectbox(
    "📊 Choose plot type",
    ["Histogram", "Boxplot", "Scatterplot"]
)

# Sidebar - Heatmap color
heatmap_color = st.sidebar.color_picker("🎨 Pick heatmap color", "#FF5733")

# Sidebar - Download button
buffer = io.BytesIO()
df.to_csv(buffer, index=False)
buffer.seek(0)
st.sidebar.download_button(
    label="⬇️ Download CSV",
    data=buffer,
    file_name="Processed_Weather.csv",
    mime="text/csv"
)

# Filtered Data
selected_data = df[selected_months]

# Tabs
tabs = st.tabs(["🏠 Home", "🗂️ Data Preview", "📈 Statistics", "🧩 Correlation", "🎨 Visualizations"])

# --- Home Tab ---
with tabs[0]:
    st.header("🏠 Welcome to the Weather Data Interactive Explorer")
    st.markdown("""
    - Upload your weather CSV data
    - Choose **one or more months** (Jan-Dec) to analyze
    - Preview and explore your dataset
    - Generate descriptive statistics
    - View correlations
    - Build custom plots
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/1163/1163661.png", width=150)
    st.success("✨ Get started by exploring the tabs above!")

# --- Data Preview Tab ---
with tabs[1]:
    st.header("🗂️ Data Preview")
    st.dataframe(df, use_container_width=True)

# --- Statistics Tab ---
with tabs[2]:
    st.header("📈 Descriptive Statistics")
    if st.checkbox("Show descriptive statistics", True):
        st.write(selected_data.describe())
    if st.checkbox("Show data types"):
        st.write(selected_data.dtypes)

# --- Correlation Tab ---
with tabs[3]:
    st.header("🧩 Correlation Heatmap")
    if len(selected_months) >= 2:
        corr = selected_data.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", cbar_kws={'label': 'Correlation'})
        ax.set_title("Correlation Heatmap", fontsize=16, color=heatmap_color)
        st.pyplot(fig)
    elif len(selected_months) == 1:
        st.info("ℹ️ Correlation needs at least two selected months. Select more in the sidebar.")
    else:
        st.warning("⚠️ Please select at least one month column.")

# --- Visualizations Tab ---
with tabs[4]:
    st.header("🎨 Custom Visualizations")
    if not selected_months:
        st.warning("⚠️ Please select at least one month column.")
    else:
        if plot_type == "Histogram":
            st.subheader("📊 Histograms")
            for col in selected_months:
                fig, ax = plt.subplots()
                sns.histplot(selected_data[col].dropna(), bins=30, kde=True, color=heatmap_color, ax=ax)
                ax.set_title(f"Histogram of {col}")
                st.pyplot(fig)

        elif plot_type == "Boxplot":
            st.subheader("📦 Boxplot")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=selected_data, palette="Set2", ax=ax)
            ax.set_title("Boxplot of Selected Months")
            st.pyplot(fig)

        elif plot_type == "Scatterplot":
            st.subheader("📈 Scatterplot")
            if len(selected_months) >= 2:
                x_axis = st.selectbox("X-axis", selected_months, index=0)
                y_axis = st.selectbox("Y-axis", selected_months, index=1)
                fig, ax = plt.subplots()
                sns.scatterplot(x=selected_data[x_axis], y=selected_data[y_axis], color=heatmap_color, ax=ax)
                ax.set_title(f"{y_axis} vs {x_axis}")
                st.pyplot(fig)
            else:
                st.warning("⚠️ Please select at least two month columns.")

