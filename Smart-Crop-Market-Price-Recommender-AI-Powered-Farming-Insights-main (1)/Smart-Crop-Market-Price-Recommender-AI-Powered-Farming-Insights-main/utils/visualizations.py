import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_crop_probability_chart(crop_predictions):
    """Create a horizontal bar chart showing crop suitability probabilities"""
    
    crops = [pred['crop'].title() for pred in crop_predictions]
    probabilities = [pred['probability'] * 100 for pred in crop_predictions]
    
    # Color scale based on probability
    colors = ['#2E8B57' if p > 60 else '#FF6B35' if p > 30 else '#FFA500' for p in probabilities]
    
    fig = go.Figure(data=[
        go.Bar(
            y=crops,
            x=probabilities,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{p:.1f}%' for p in probabilities],
            textposition='inside',
            textfont=dict(color='white', size=12)
        )
    ])
    
    fig.update_layout(
        title='Crop Suitability Analysis',
        xaxis_title='Suitability Score (%)',
        yaxis_title='Crops',
        height=400,
        showlegend=False,
        xaxis=dict(range=[0, 100])
    )
    
    return fig

def create_price_trend_chart(price_data, commodity):
    """Create a line chart showing price trends over time"""
    
    if price_data.empty:
        # Create empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No price data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title=f'{commodity.title()} Price Trends',
            height=400
        )
        return fig
    
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=price_data['date'],
        y=price_data['price'],
        mode='lines+markers',
        name='Price',
        line=dict(color='#2E8B57', width=3),
        marker=dict(size=6)
    ))
    
    # Add moving average if enough data points
    if len(price_data) >= 7:
        moving_avg = price_data['price'].rolling(window=7, center=True).mean()
        fig.add_trace(go.Scatter(
            x=price_data['date'],
            y=moving_avg,
            mode='lines',
            name='7-day Moving Average',
            line=dict(color='#FF6B35', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=f'{commodity.title()} Price Trends (Last 30 Days)',
        xaxis_title='Date',
        yaxis_title='Price (₹/kg)',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_regional_price_comparison(regional_data, commodity):
    """Create a bar chart comparing prices across different regions"""
    
    if regional_data.empty:
        # Create empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No regional price data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title=f'{commodity.title()} Regional Price Comparison',
            height=400
        )
        return fig
    
    # Sort by price for better visualization
    regional_data = regional_data.sort_values('price', ascending=True)
    
    # Color code based on price level
    max_price = regional_data['price'].max()
    min_price = regional_data['price'].min()
    
    colors = []
    for price in regional_data['price']:
        if price >= max_price * 0.9:
            colors.append('#2E8B57')  # High price - green
        elif price <= min_price * 1.1:
            colors.append('#FF6B35')  # Low price - red
        else:
            colors.append('#FFA500')  # Medium price - orange
    
    fig = go.Figure(data=[
        go.Bar(
            x=regional_data['state'],
            y=regional_data['price'],
            marker=dict(color=colors),
            text=[f'₹{p:.2f}' for p in regional_data['price']],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f'{commodity.title()} Prices Across States',
        xaxis_title='State',
        yaxis_title='Price (₹/kg)',
        height=400,
        showlegend=False,
        xaxis=dict(tickangle=45)
    )
    
    return fig

def create_feature_importance_chart(feature_importance):
    """Create a chart showing feature importance from the ML model"""
    
    features = [item[0] for item in feature_importance]
    importance = [item[1] for item in feature_importance]
    
    fig = go.Figure(data=[
        go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker=dict(color='#2E8B57')
        )
    ])
    
    fig.update_layout(
        title='Feature Importance in Crop Recommendation',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        height=400
    )
    
    return fig

def create_profit_analysis_chart(crop_data, price_data):
    """Create a comprehensive profit analysis visualization"""
    
    # This would combine crop yield potential with market prices
    # For now, create a simple profit potential chart
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Crop Suitability vs Market Price', 'Seasonal Price Variation'),
        vertical_spacing=0.1
    )
    
    # Top subplot: Suitability vs Price scatter
    if not crop_data.empty and not price_data.empty:
        # Sample data for demonstration
        crops = ['Rice', 'Wheat', 'Cotton', 'Maize', 'Sugarcane']
        suitability = [85, 72, 68, 75, 80]
        avg_price = [45, 25, 65, 20, 35]
        
        fig.add_trace(
            go.Scatter(
                x=suitability,
                y=avg_price,
                mode='markers+text',
                text=crops,
                textposition='top center',
                marker=dict(size=15, color='#2E8B57'),
                name='Crops'
            ),
            row=1, col=1
        )
        
        # Bottom subplot: Seasonal price variation
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        seasonal_prices = [42, 44, 46, 48, 52, 55, 58, 54, 48, 42, 40, 41]
        
        fig.add_trace(
            go.Scatter(
                x=months,
                y=seasonal_prices,
                mode='lines+markers',
                line=dict(color='#FF6B35', width=3),
                marker=dict(size=8),
                name='Seasonal Price'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Profit Analysis Dashboard"
    )
    
    fig.update_xaxes(title_text="Suitability Score (%)", row=1, col=1)
    fig.update_yaxes(title_text="Average Price (₹/kg)", row=1, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Price (₹/kg)", row=2, col=1)
    
    return fig

def create_dashboard_summary(recommendations):
    """Create a summary dashboard with key metrics"""
    
    if not recommendations:
        fig = go.Figure()
        fig.add_annotation(
            text="No recommendations available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    # Create a summary with key metrics
    metrics = [
        "Recommended Crop",
        "Best Market",
        "Expected Price",
        "Profit Margin"
    ]
    
    values = [
        recommendations.get('crop', 'N/A'),
        recommendations.get('market_recommendation', {}).get('best_market', 'N/A'),
        f"₹{recommendations.get('market_recommendation', {}).get('expected_price', 0):.2f}",
        f"{recommendations.get('market_recommendation', {}).get('profit_margin', 0):.1f}%"
    ]
    
    colors = ['#2E8B57', '#FF6B35', '#FFA500', '#4169E1']
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics,
            y=[1, 1, 1, 1],  # Equal height bars
            marker=dict(color=colors),
            text=values,
            textposition='middle',
            textfont=dict(size=14, color='white')
        )
    ])
    
    fig.update_layout(
        title='Recommendation Summary',
        showlegend=False,
        height=300,
        yaxis=dict(showticklabels=False, showgrid=False),
        xaxis=dict(tickangle=0)
    )
    
    return fig
