import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import uuid
from models.crop_recommender import CropRecommender
from models.price_predictor import PricePredictor
from utils.visualizations import (
    create_crop_probability_chart,
    create_price_trend_chart,
    create_regional_price_comparison
)
from utils.recommendations import RecommendationEngine
from data.data_generator import generate_sample_data
from database.connection import get_database
from pages.farmer_profile import show_farmer_profile

# ✅ Streamlit Cloud runs automatically on port 8501 — no manual config needed
st.set_page_config(
    page_title="Smart Crop & Market Price Recommender",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
if 'crop_model' not in st.session_state:
    st.session_state.crop_model = None
if 'price_model' not in st.session_state:
    st.session_state.price_model = None
if 'recommendation_engine' not in st.session_state:
    st.session_state.recommendation_engine = None

@st.cache_data
def load_and_train_models():
    try:
        crop_data, price_data = generate_sample_data()
        crop_model = CropRecommender()
        crop_model.train(crop_data)

        price_model = PricePredictor()
        price_model.train(price_data)

        rec_engine = RecommendationEngine(crop_model, price_model)
        return crop_model, price_model, rec_engine, crop_data, price_data
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None


def main():
    st.title("🌾 Smart Crop & Market Price Recommender")
    st.markdown("### AI-powered farming decisions for maximum profitability")

    page = st.sidebar.selectbox("Navigate", ["🏠 Home", "👨‍🌾 Farmer Profile"])

    if page == "👨‍🌾 Farmer Profile":
        show_farmer_profile()
        return

    with st.spinner("Loading AI models..."):
        crop_model, price_model, rec_engine, crop_data, price_data = load_and_train_models()

    if crop_model is None:
        st.error("Failed to load models. Please refresh the page.")
        return

    st.sidebar.header("🌱 Farm Conditions")

    st.sidebar.subheader("Soil Nutrients")
    nitrogen = st.sidebar.slider("Nitrogen (N) - kg/ha", 0, 200, 90)
    phosphorus = st.sidebar.slider("Phosphorus (P) - kg/ha", 5, 150, 42)
    potassium = st.sidebar.slider("Potassium (K) - kg/ha", 5, 250, 43)
    ph = st.sidebar.slider("Soil pH", 3.5, 10.0, 6.5, 0.1)

    st.sidebar.subheader("Weather Conditions")
    temperature = st.sidebar.slider("Temperature (°C)", 8.0, 45.0, 25.0, 0.5)
    humidity = st.sidebar.slider("Humidity (%)", 14.0, 100.0, 80.0, 1.0)
    rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 150.0, 5.0)

    st.sidebar.subheader("Location")
    states = [
        'Maharashtra', 'Punjab', 'Uttar Pradesh',
        'Haryana', 'Gujarat', 'Rajasthan',
        'West Bengal', 'Tamil Nadu'
    ]
    selected_state = st.sidebar.selectbox("Select your state:", states)

    if st.sidebar.button("🔍 Get Recommendations", type="primary"):
        if crop_model is None or price_model is None or rec_engine is None:
            st.error("Models not loaded properly. Please refresh.")
            return

        input_features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

        with st.spinner("Analyzing soil and weather conditions..."):
            crop_predictions = crop_model.predict_with_probability(input_features)
            recommended_crop = crop_predictions[0]['crop']

        with st.spinner("Analyzing market data..."):
            recommendations = rec_engine.get_comprehensive_recommendation(
                input_features[0], selected_state
            )

            if 'farmer_id' in st.session_state and st.session_state.farmer_id and recommendations:
                db = get_database()
                session_id = str(uuid.uuid4())
                recommendation_data = {
                    'farmer_id': st.session_state.farmer_id,
                    'session_id': session_id,
                    'recommended_crop': recommendations['recommended_crop'],
                    'suitability_score': crop_predictions[0]['suitability_score'],
                    'market_score': recommendations['crop_analysis'][0]['market_score'],
                    'combined_score': recommendations['crop_analysis'][0]['combined_score'],
                    'best_market': recommendations['market_recommendation']['best_market'],
                    'expected_price': recommendations['market_recommendation']['expected_price'],
                    'profit_margin': recommendations['market_recommendation']['profit_margin'],
                    'soil_n': nitrogen,
                    'soil_p': phosphorus,
                    'soil_k': potassium,
                    'soil_ph': ph,
                    'temperature': temperature,
                    'humidity': humidity,
                    'rainfall': rainfall,
                    'state': selected_state
                }
                db.insert_recommendation(recommendation_data)
                st.success("✅ Recommendation saved to your profile!")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.header("🎯 Crop Recommendation")
            st.success(f"**Recommended Crop: {recommended_crop.title()}**")

            st.subheader("Crop Suitability Analysis")
            fig_crop = create_crop_probability_chart(crop_predictions[:5])
            st.plotly_chart(fig_crop, use_container_width=True)

            st.header("💰 Market Analysis")
            if recommendations:
                market_rec = recommendations['market_recommendation']
                st.info(f"**Best Market: {market_rec['best_market']}**")
                st.metric("Expected Price", f"₹{market_rec['expected_price']:.2f}/kg",
                          f"+{market_rec['profit_margin']:.1f}%")

                st.subheader("Price Trend Analysis")
                price_trend_data = price_model.get_price_trends(recommended_crop, selected_state)
                fig_trend = create_price_trend_chart(price_trend_data, recommended_crop)
                st.plotly_chart(fig_trend, use_container_width=True)

                st.subheader("Regional Price Comparison")
                regional_data = price_model.get_regional_prices(recommended_crop)
                fig_regional = create_regional_price_comparison(regional_data, recommended_crop)
                st.plotly_chart(fig_regional, use_container_width=True)

        with col2:
            st.header("📊 Input Summary")
            st.write(f"**Nitrogen:** {nitrogen} kg/ha")
            st.write(f"**Phosphorus:** {phosphorus} kg/ha")
            st.write(f"**Potassium:** {potassium} kg/ha")
            st.write(f"**pH Level:** {ph}")
            st.write(f"**Temperature:** {temperature}°C")
            st.write(f"**Humidity:** {humidity}%")
            st.write(f"**Rainfall:** {rainfall} mm")
            st.write(f"**Location:** {selected_state}")

            if recommendations:
                st.subheader("💡 Key Insights")
                for insight in recommendations.get('insights', []):
                    st.write(f"• {insight}")
                st.subheader("⚠️ Recommendations")
                for rec in recommendations.get('recommendations', []):
                    st.write(f"• {rec}")

    # --- Info Tabs ---
    st.header("📚 Learn More")
    tab1, tab2, tab3 = st.tabs(["About the System", "How it Works", "Data Sources"])

    with tab1:
        st.markdown("""
        ### Smart Crop & Market Price Recommender
        This AI system helps farmers make data-driven decisions for better yield and profit.
        """)

    with tab2:
        st.markdown("""
        ### How it Works
        Combines crop suitability and price prediction using machine learning models.
        """)

    with tab3:
        st.markdown("""
        ### Data Sources
        Uses synthetic and historical agricultural datasets for demonstration.
        """)

# ✅ Correct entry point — required for Streamlit Cloud
if __name__ == "__main__":
    main()