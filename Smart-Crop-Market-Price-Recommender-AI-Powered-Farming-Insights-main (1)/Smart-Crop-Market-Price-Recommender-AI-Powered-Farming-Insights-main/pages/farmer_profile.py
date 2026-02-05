import streamlit as st
import pandas as pd
from datetime import datetime, date
from database.connection import get_database
import uuid

def show_farmer_profile():
    """Display farmer profile management interface"""
    st.header("👨‍🌾 Farmer Profile")
    
    db = get_database()
    
    # Initialize session state for farmer login
    if 'farmer_id' not in st.session_state:
        st.session_state.farmer_id = None
    if 'farmer_data' not in st.session_state:
        st.session_state.farmer_data = None
    
    # Login/Registration tabs
    if st.session_state.farmer_id is None:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Login to Your Account")
            email = st.text_input("Email Address", key="login_email")
            
            if st.button("Login", type="primary"):
                if email:
                    farmer = db.get_farmer_by_email(email)
                    if farmer:
                        st.session_state.farmer_id = farmer['id']
                        st.session_state.farmer_data = farmer
                        st.success(f"Welcome back, {farmer['name']}!")
                        st.rerun()
                    else:
                        st.error("Email not found. Please register first.")
                else:
                    st.error("Please enter your email address.")
        
        with tab2:
            st.subheader("Create New Account")
            with st.form("farmer_registration"):
                name = st.text_input("Full Name*")
                email = st.text_input("Email Address*")
                phone = st.text_input("Phone Number")
                location = st.text_input("Farm Location (Village/City)")
                state = st.selectbox("State", 
                                   ['Maharashtra', 'Punjab', 'Uttar Pradesh', 'Haryana', 
                                    'Gujarat', 'Rajasthan', 'West Bengal', 'Tamil Nadu'])
                farm_size = st.number_input("Farm Size (acres)", min_value=0.1, value=1.0, step=0.1)
                
                col1, col2 = st.columns(2)
                with col1:
                    latitude = st.number_input("Latitude (optional)", value=0.0, format="%.6f")
                with col2:
                    longitude = st.number_input("Longitude (optional)", value=0.0, format="%.6f")
                
                submitted = st.form_submit_button("Register", type="primary")
                
                if submitted:
                    if name and email:
                        # Check if email already exists
                        existing_farmer = db.get_farmer_by_email(email)
                        if existing_farmer:
                            st.error("Email already registered. Please login instead.")
                        else:
                            farmer_data = {
                                'name': name,
                                'email': email,
                                'phone': phone,
                                'location': location,
                                'state': state,
                                'farm_size_acres': farm_size,
                                'latitude': latitude if latitude != 0.0 else None,
                                'longitude': longitude if longitude != 0.0 else None
                            }
                            
                            farmer_id = db.insert_farmer(farmer_data)
                            if farmer_id:
                                st.session_state.farmer_id = farmer_id
                                st.session_state.farmer_data = farmer_data
                                st.session_state.farmer_data['id'] = farmer_id
                                st.success(f"Welcome to Smart Crop Recommender, {name}!")
                                st.rerun()
                            else:
                                st.error("Registration failed. Please try again.")
                    else:
                        st.error("Please fill in all required fields (marked with *)")
    
    else:
        # Farmer is logged in - show dashboard
        farmer = st.session_state.farmer_data
        
        # Header with farmer info and logout
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"Logged in as: **{farmer['name']}** ({farmer['email']})")
        with col2:
            if st.button("Logout"):
                st.session_state.farmer_id = None
                st.session_state.farmer_data = None
                st.rerun()
        
        # Dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📋 Recommendation History", "🌾 Yield Tracking", "👤 Profile Settings"])
        
        with tab1:
            show_farmer_dashboard(db, farmer['id'])
        
        with tab2:
            show_recommendation_history(db, farmer['id'])
        
        with tab3:
            show_yield_tracking(db, farmer['id'])
        
        with tab4:
            show_profile_settings(farmer)

def show_farmer_dashboard(db, farmer_id):
    """Show farmer dashboard with key metrics"""
    st.subheader("Farm Performance Dashboard")
    
    # Get farmer's recent recommendations
    recommendations = db.get_farmer_recommendations(farmer_id, limit=5)
    yield_history = db.get_farmer_yield_history(farmer_id)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_recommendations = len(recommendations) if recommendations is not None else 0
        st.metric("Total Recommendations", total_recommendations)
    
    with col2:
        if yield_history is not None and not yield_history.empty:
            avg_satisfaction = yield_history['satisfaction_rating'].mean()
            st.metric("Avg Satisfaction", f"{avg_satisfaction:.1f}/5")
        else:
            st.metric("Avg Satisfaction", "N/A")
    
    with col3:
        if yield_history is not None and not yield_history.empty:
            total_profit = yield_history['net_profit'].sum()
            st.metric("Total Profit", f"₹{total_profit:,.0f}")
        else:
            st.metric("Total Profit", "₹0")
    
    with col4:
        if yield_history is not None and not yield_history.empty:
            successful_predictions = len(yield_history[yield_history['price_performance'] == 'Better than predicted'])
            accuracy_rate = (successful_predictions / len(yield_history)) * 100
            st.metric("Prediction Accuracy", f"{accuracy_rate:.1f}%")
        else:
            st.metric("Prediction Accuracy", "N/A")
    
    # Recent activity
    st.subheader("Recent Activity")
    if recommendations is not None and not recommendations.empty:
        recent_rec = recommendations.iloc[0]
        st.info(f"**Latest Recommendation:** {recent_rec['recommended_crop'].title()} "
               f"• Expected Price: ₹{recent_rec['expected_price']:.2f}/kg "
               f"• Profit Margin: {recent_rec['profit_margin']:.1f}%")
    else:
        st.info("No recommendations yet. Visit the main page to get your first crop recommendation!")
    
    # Performance chart
    if yield_history is not None and not yield_history.empty:
        st.subheader("Price Prediction Performance")
        import plotly.express as px
        
        fig = px.scatter(yield_history, 
                        x='predicted_price', 
                        y='actual_price_per_kg',
                        color='price_performance',
                        title='Predicted vs Actual Prices',
                        labels={'predicted_price': 'Predicted Price (₹/kg)', 
                               'actual_price_per_kg': 'Actual Price (₹/kg)'})
        
        # Add diagonal line for perfect prediction
        max_price = max(yield_history['predicted_price'].max(), yield_history['actual_price_per_kg'].max())
        fig.add_shape(type="line", x0=0, y0=0, x1=max_price, y1=max_price, 
                     line=dict(dash="dash", color="gray"))
        
        st.plotly_chart(fig, use_container_width=True)

def show_recommendation_history(db, farmer_id):
    """Show detailed recommendation history"""
    st.subheader("Recommendation History")
    
    recommendations = db.get_farmer_recommendations(farmer_id, limit=20)
    
    if recommendations is not None and not recommendations.empty:
        # Display recommendations in a nice format
        for idx, rec in recommendations.iterrows():
            with st.expander(f"{rec['recommended_crop'].title()} - {rec['created_at'].strftime('%Y-%m-%d')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Crop Details:**")
                    st.write(f"• Crop: {rec['recommended_crop'].title()}")
                    st.write(f"• Suitability Score: {rec['suitability_score']:.1f}%")
                    st.write(f"• Combined Score: {rec['combined_score']:.1f}%")
                    
                    st.write("**Soil Conditions:**")
                    st.write(f"• Nitrogen: {rec['soil_n']:.1f} kg/ha")
                    st.write(f"• Phosphorus: {rec['soil_p']:.1f} kg/ha")
                    st.write(f"• Potassium: {rec['soil_k']:.1f} kg/ha")
                    st.write(f"• pH: {rec['soil_ph']:.1f}")
                
                with col2:
                    st.write("**Market Analysis:**")
                    st.write(f"• Best Market: {rec['best_market']}")
                    st.write(f"• Expected Price: ₹{rec['expected_price']:.2f}/kg")
                    st.write(f"• Profit Margin: {rec['profit_margin']:.1f}%")
                    
                    st.write("**Weather Conditions:**")
                    st.write(f"• Temperature: {rec['temperature']:.1f}°C")
                    st.write(f"• Humidity: {rec['humidity']:.1f}%")
                    st.write(f"• Rainfall: {rec['rainfall']:.1f} mm")
    else:
        st.info("No recommendations found. Visit the main page to get your first recommendation!")

def show_yield_tracking(db, farmer_id):
    """Show yield tracking and input interface"""
    st.subheader("Yield Tracking")
    
    tab1, tab2 = st.tabs(["Add Yield Data", "Yield History"])
    
    with tab1:
        st.write("Track your actual farming outcomes to improve future recommendations.")
        
        # Get farmer's recommendations for dropdown
        recommendations = db.get_farmer_recommendations(farmer_id, limit=50)
        
        if recommendations is not None and not recommendations.empty:
            with st.form("yield_data_form"):
                # Select recommendation
                rec_options = [(f"{row['recommended_crop'].title()} - {row['created_at'].strftime('%Y-%m-%d')}", 
                              row['id']) for _, row in recommendations.iterrows()]
                
                selected_rec = st.selectbox(
                    "Select Recommendation",
                    options=rec_options,
                    format_func=lambda x: x[0]
                )
                
                crop_planted = st.text_input("Actual Crop Planted", 
                                           value=selected_rec[0].split(' - ')[0] if selected_rec else "")
                
                col1, col2 = st.columns(2)
                with col1:
                    planting_date = st.date_input("Planting Date")
                with col2:
                    harvest_date = st.date_input("Harvest Date")
                
                col1, col2 = st.columns(2)
                with col1:
                    actual_yield = st.number_input("Actual Yield (kg/acre)", min_value=0.0, step=10.0)
                    actual_price = st.number_input("Actual Price (₹/kg)", min_value=0.0, step=0.1)
                with col2:
                    total_costs = st.number_input("Total Costs (₹)", min_value=0.0, step=100.0)
                    satisfaction = st.slider("Satisfaction Rating", 1, 5, 3)
                
                notes = st.text_area("Notes (optional)", placeholder="Any additional observations...")
                
                submitted = st.form_submit_button("Add Yield Data", type="primary")
                
                if submitted:
                    if selected_rec and actual_yield > 0 and actual_price > 0:
                        total_revenue = actual_yield * actual_price
                        net_profit = total_revenue - total_costs
                        
                        yield_data = {
                            'farmer_id': farmer_id,
                            'recommendation_id': selected_rec[1],
                            'crop_planted': crop_planted,
                            'planting_date': planting_date,
                            'harvest_date': harvest_date,
                            'actual_yield_kg_per_acre': actual_yield,
                            'actual_price_per_kg': actual_price,
                            'total_revenue': total_revenue,
                            'total_costs': total_costs,
                            'net_profit': net_profit,
                            'satisfaction_rating': satisfaction,
                            'notes': notes
                        }
                        
                        yield_id = db.insert_yield_data(yield_data)
                        if yield_id:
                            st.success("Yield data added successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to add yield data. Please try again.")
                    else:
                        st.error("Please fill in all required fields.")
        else:
            st.info("No recommendations found. Get a recommendation first to track yield data.")
    
    with tab2:
        yield_history = db.get_farmer_yield_history(farmer_id)
        
        if yield_history is not None and not yield_history.empty:
            st.write(f"**{len(yield_history)} yield records found**")
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_yield = yield_history['actual_yield_kg_per_acre'].mean()
                st.metric("Avg Yield", f"{avg_yield:.0f} kg/acre")
            with col2:
                avg_price = yield_history['actual_price_per_kg'].mean()
                st.metric("Avg Price", f"₹{avg_price:.2f}/kg")
            with col3:
                total_profit = yield_history['net_profit'].sum()
                st.metric("Total Profit", f"₹{total_profit:,.0f}")
            
            # Display yield history
            display_cols = ['crop_planted', 'harvest_date', 'actual_yield_kg_per_acre', 
                           'actual_price_per_kg', 'net_profit', 'satisfaction_rating', 'price_performance']
            
            yield_display = yield_history[display_cols].copy()
            yield_display.columns = ['Crop', 'Harvest Date', 'Yield (kg/acre)', 
                                   'Price (₹/kg)', 'Net Profit (₹)', 'Satisfaction', 'Price Performance']
            
            st.dataframe(yield_display, use_container_width=True)
        else:
            st.info("No yield data recorded yet. Use the 'Add Yield Data' tab to track your farming outcomes.")

def show_profile_settings(farmer_data):
    """Show and allow editing of profile settings"""
    st.subheader("Profile Settings")
    
    st.info("Profile editing will be available in a future update. For now, contact support to modify your profile.")
    
    # Display current profile info
    st.write("**Current Profile:**")
    st.write(f"• Name: {farmer_data['name']}")
    st.write(f"• Email: {farmer_data['email']}")
    st.write(f"• Phone: {farmer_data.get('phone', 'Not provided')}")
    st.write(f"• Location: {farmer_data.get('location', 'Not provided')}")
    st.write(f"• State: {farmer_data['state']}")
    st.write(f"• Farm Size: {farmer_data.get('farm_size_acres', 0)} acres")

if __name__ == "__main__":
    show_farmer_profile()