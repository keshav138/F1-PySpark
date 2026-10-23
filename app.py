"""app.py - Streamlit Dashboard for F1 PySpark Project"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel



# ----------------------------- Configuration -----------------------------
st.set_page_config(page_title="F1 Race Analytics", layout="wide", page_icon="🏎️")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")

PRE_RACE_FEATURES = ["grid_pos", "quali_pos", "driver_age", "driver_podium_rate", "driver_form", "constructor_form"]
LIVE_FEATURES = ["lap_frac", "lap", "position", "grid_pos", "pos_change", "recent_pace_delta", "pits_so_far"]

# ----------------------------- Resource Caching -----------------------------

@st.cache_resource
def init_spark():
    """Initialize SparkSession without Streamlit caching to avoid Py4J gateway errors."""
    spark = (
        SparkSession.builder.appName("F1-Dashboard")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark

@st.cache_resource
def load_models(_spark_session):
    """Load the saved PySpark ML models without Streamlit caching."""
    pre_race, live = None, None
    try:
        pre_race = PipelineModel.load(os.path.join(OUT_DIR, "best_model"))
        live = PipelineModel.load(os.path.join(OUT_DIR, "live_model"))
    except Exception as e:
        st.sidebar.warning(f"Models not found. Run 02_ml.py and 03_ml.py first. Error: {e}")
    return pre_race, live

spark = init_spark()
pre_race_model, live_model = load_models(spark)

# ----------------------------- Helper Functions -----------------------------
def create_gauge(probability, title):
    """Creates a Plotly gauge chart for podium probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': title},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "silver"},
                {'range': [66, 100], 'color': "gold"}
            ],
        }
    ))
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ----------------------------- Sidebar Navigation -----------------------------
st.sidebar.title("F1 Analytics Engine 🏁")
page = st.sidebar.radio("Navigation", [
    "📊 SQL Analytics", 
    "🔮 Pre-Race Predictor", 
    "⏱️ Live Lap Telemetry", 
    "🧠 Driver Clusters & Metrics"
])

# ----------------------------- Pages -----------------------------

if page == "📊 SQL Analytics":
    st.title("Historical F1 Analytics (Spark SQL)")
    st.markdown("Insights generated via PySpark RDDs and Spark SQL aggregations.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Wins per Driver")
        try:
            wins_df = pd.read_csv(os.path.join(OUT_DIR, "sql_wins.csv"))
            fig = px.bar(wins_df.head(10), x='driver', y='wins', color='wins', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Run 01_rdd_sql.py to generate data.")

        st.subheader("Grid Position vs Avg Finish")
        try:
            grid_df = pd.read_csv(os.path.join(OUT_DIR, "sql_grid_vs_finish.csv"))
            fig = px.line(grid_df, x='grid_position', y='avg_finish', markers=True)
            fig.update_yaxes(autorange="reversed") # Lower finish is better
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    with col2:
        st.subheader("Constructor Dominance (2000+)")
        try:
            cons_df = pd.read_csv(os.path.join(OUT_DIR, "sql_constructor_points.csv"))
            top_cons = cons_df.groupby('constructor')['total_points'].sum().nlargest(5).index
            fig = px.line(cons_df[cons_df['constructor'].isin(top_cons)], 
                          x='year', y='total_points', color='constructor', markers=True)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
            
        st.subheader("Pit Stop Efficiency")
        try:
            pit_df = pd.read_csv(os.path.join(OUT_DIR, "sql_avg_pit_stops.csv"))
            fig = px.bar(pit_df, x='avg_pit_time_sec', y='constructor', orientation='h', color='avg_pit_time_sec')
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass


elif page == "🔮 Pre-Race Predictor":
    st.title("Pre-Race Podium Predictor")
    st.markdown("Adjust historical and qualifying variables to predict a driver's chance of finishing on the podium before the lights go out.")
    
    if pre_race_model:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Driver & Constructor Form")
            grid_pos = st.slider("Grid Position", 1, 22, 5)
            quali_pos = st.slider("Qualifying Position", 1, 22, 5)
            driver_age = st.slider("Driver Age (Years)", 18.0, 45.0, 26.0)
            driver_podium_rate = st.slider("Career Podium Rate", 0.0, 1.0, 0.15)
            driver_form = st.slider("Driver Form (Avg points last 5 races)", 0.0, 26.0, 10.0)
            constructor_form = st.slider("Constructor Form (Avg pts last 10 races)", 0.0, 44.0, 15.0)
            
            if st.button("Predict Pre-Race Probability"):
                # Create a single row Spark DataFrame
                input_data = [(float(grid_pos), float(quali_pos), float(driver_age), 
                               float(driver_podium_rate), float(driver_form), float(constructor_form))]
                df = spark.createDataFrame(input_data, PRE_RACE_FEATURES)
                
                # Inference
                pred = pre_race_model.transform(df)
                prob = pred.select("probability").head()[0][1]  # Index 1 is the positive class (Podium)
                
                with col2:
                    st.plotly_chart(create_gauge(prob, "Pre-Race Podium Probability"), use_container_width=True)
    else:
        st.error("Pre-Race model not loaded.")


elif page == "⏱️ Live Lap Telemetry":
    st.title("Live Race Simulator")
    st.markdown("Dynamically predict podium chances as the race unfolds based on track position, pace delta, and tyre age.")
    
    if live_model:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Current Race Telemetry")
            lap = st.number_input("Current Lap", min_value=1, max_value=80, value=30)
            lap_frac = st.slider("Race Completion %", 0.0, 1.0, lap/60.0)
            position = st.slider("Current Running Position", 1, 20, 4)
            grid_pos = st.slider("Original Grid Position", 1, 22, 6)
            pos_change = grid_pos - position
            st.metric("Places Gained/Lost", pos_change)
            
            recent_pace_delta = st.slider("Pace Delta vs Field (ms, negative is faster)", -2000.0, 2000.0, -200.0)
            pits_so_far = st.selectbox("Pit Stops Made", [0, 1, 2, 3, 4])
            
            if st.button("Predict Live Probability"):
                input_data = [(float(lap_frac), float(lap), float(position), float(grid_pos), 
                               float(pos_change), float(recent_pace_delta), float(pits_so_far))]
                df = spark.createDataFrame(input_data, LIVE_FEATURES)
                
                pred = live_model.transform(df)
                prob = pred.select("probability").head()[0][1]
                
                with col2:
                    st.plotly_chart(create_gauge(prob, "Live Podium Probability"), use_container_width=True)
    else:
        st.error("Live lap model not loaded.")


elif page == "🧠 Driver Clusters & Metrics":
    st.title("Machine Learning Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Driver Clusters (K-Means)")
        try:
            clusters = pd.read_csv(os.path.join(OUT_DIR, "cluster_summary.csv"))
            st.dataframe(clusters, use_container_width=True)
            st.markdown("""
            * **Cluster Profiles:** Groups drivers based on win rate, average finish, and grid consistency. 
            """)
        except Exception:
            st.info("Cluster data missing.")
            
    with col2:
        st.subheader("Feature Correlation Heatmap")
        try:
            st.image(os.path.join(OUT_DIR, "correlation_heatmap.png"), use_column_width=True)
        except Exception:
            st.info("Heatmap missing.")