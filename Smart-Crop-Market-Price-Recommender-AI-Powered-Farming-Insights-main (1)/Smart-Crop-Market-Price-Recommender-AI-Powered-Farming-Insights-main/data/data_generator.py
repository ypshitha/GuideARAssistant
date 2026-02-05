import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_data():
    """Generate representative crop and price datasets for training"""
    
    # Generate crop recommendation dataset
    crop_data = generate_crop_dataset()
    
    # Generate price prediction dataset
    price_data = generate_price_dataset()
    
    return crop_data, price_data

def generate_crop_dataset(n_samples=2200):
    """Generate crop recommendation dataset with realistic agricultural patterns"""
    
    crops = ['rice', 'wheat', 'cotton', 'maize', 'sugarcane', 'barley', 'groundnut', 'soybean']
    
    # Define crop-specific parameter ranges based on agricultural knowledge
    crop_profiles = {
        'rice': {
            'N': (80, 120), 'P': (35, 60), 'K': (35, 60),
            'temperature': (20, 35), 'humidity': (75, 95),
            'ph': (5.5, 7.0), 'rainfall': (150, 300)
        },
        'wheat': {
            'N': (50, 100), 'P': (25, 50), 'K': (25, 50),
            'temperature': (15, 25), 'humidity': (50, 70),
            'ph': (6.0, 7.5), 'rainfall': (50, 100)
        },
        'cotton': {
            'N': (80, 150), 'P': (40, 80), 'K': (60, 100),
            'temperature': (25, 35), 'humidity': (60, 80),
            'ph': (6.0, 8.0), 'rainfall': (75, 150)
        },
        'maize': {
            'N': (100, 150), 'P': (50, 80), 'K': (50, 80),
            'temperature': (20, 30), 'humidity': (60, 80),
            'ph': (6.0, 7.5), 'rainfall': (100, 200)
        },
        'sugarcane': {
            'N': (120, 200), 'P': (60, 100), 'K': (80, 120),
            'temperature': (25, 35), 'humidity': (70, 90),
            'ph': (6.0, 8.0), 'rainfall': (150, 250)
        },
        'barley': {
            'N': (40, 80), 'P': (20, 40), 'K': (20, 40),
            'temperature': (12, 22), 'humidity': (50, 70),
            'ph': (6.5, 7.5), 'rainfall': (40, 80)
        },
        'groundnut': {
            'N': (20, 40), 'P': (60, 100), 'K': (80, 120),
            'temperature': (25, 35), 'humidity': (65, 85),
            'ph': (6.0, 7.5), 'rainfall': (75, 150)
        },
        'soybean': {
            'N': (30, 60), 'P': (70, 110), 'K': (60, 100),
            'temperature': (22, 32), 'humidity': (70, 90),
            'ph': (6.0, 7.5), 'rainfall': (100, 200)
        }
    }
    
    data = []
    samples_per_crop = n_samples // len(crops)
    
    for crop in crops:
        profile = crop_profiles[crop]
        
        for _ in range(samples_per_crop):
            # Generate parameters within crop-specific ranges with some noise
            sample = {
                'N': np.random.normal(np.mean(profile['N']), (profile['N'][1] - profile['N'][0]) / 6),
                'P': np.random.normal(np.mean(profile['P']), (profile['P'][1] - profile['P'][0]) / 6),
                'K': np.random.normal(np.mean(profile['K']), (profile['K'][1] - profile['K'][0]) / 6),
                'temperature': np.random.normal(np.mean(profile['temperature']), (profile['temperature'][1] - profile['temperature'][0]) / 6),
                'humidity': np.random.normal(np.mean(profile['humidity']), (profile['humidity'][1] - profile['humidity'][0]) / 6),
                'ph': np.random.normal(np.mean(profile['ph']), (profile['ph'][1] - profile['ph'][0]) / 6),
                'rainfall': np.random.normal(np.mean(profile['rainfall']), (profile['rainfall'][1] - profile['rainfall'][0]) / 6),
                'label': crop
            }
            
            # Ensure values are within reasonable bounds
            sample['N'] = np.clip(sample['N'], 0, 200)
            sample['P'] = np.clip(sample['P'], 5, 150)
            sample['K'] = np.clip(sample['K'], 5, 250)
            sample['temperature'] = np.clip(sample['temperature'], 8, 45)
            sample['humidity'] = np.clip(sample['humidity'], 14, 100)
            sample['ph'] = np.clip(sample['ph'], 3.5, 10)
            sample['rainfall'] = np.clip(sample['rainfall'], 20, 300)
            
            data.append(sample)
    
    # Add some cross-crop samples for better model generalization
    for _ in range(n_samples - len(data)):
        crop = random.choice(crops)
        profile = crop_profiles[crop]
        
        # Add more variation for cross-crop samples
        sample = {
            'N': np.random.uniform(0, 200),
            'P': np.random.uniform(5, 150),
            'K': np.random.uniform(5, 250),
            'temperature': np.random.uniform(8, 45),
            'humidity': np.random.uniform(14, 100),
            'ph': np.random.uniform(3.5, 10),
            'rainfall': np.random.uniform(20, 300),
            'label': crop
        }
        data.append(sample)
    
    return pd.DataFrame(data)

def generate_price_dataset(n_samples=5000):
    """Generate price prediction dataset with seasonal and regional patterns"""
    
    commodities = ['rice', 'wheat', 'cotton', 'maize', 'sugarcane', 'barley', 'groundnut', 'soybean']
    states = ['Maharashtra', 'Punjab', 'Uttar Pradesh', 'Haryana', 'Gujarat', 'Rajasthan', 'West Bengal', 'Tamil Nadu']
    
    # Base prices for different commodities (per kg in INR)
    base_prices = {
        'rice': 45, 'wheat': 25, 'cotton': 65, 'maize': 20, 
        'sugarcane': 35, 'barley': 22, 'groundnut': 80, 'soybean': 55
    }
    
    # State price multipliers (based on productivity and market access)
    state_multipliers = {
        'Maharashtra': 1.1, 'Punjab': 0.95, 'Uttar Pradesh': 0.9,
        'Haryana': 0.98, 'Gujarat': 1.05, 'Rajasthan': 1.0,
        'West Bengal': 1.02, 'Tamil Nadu': 1.08
    }
    
    data = []
    start_date = datetime.now() - timedelta(days=365*2)  # 2 years of data
    
    for i in range(n_samples):
        commodity = random.choice(commodities)
        state = random.choice(states)
        
        # Generate random date within the range
        random_days = random.randint(0, 365*2)
        date = start_date + timedelta(days=random_days)
        
        # Calculate base price
        base_price = base_prices[commodity]
        
        # Apply state multiplier
        price = base_price * state_multipliers[state]
        
        # Add seasonal variation
        month = date.month
        if month in [10, 11, 12]:  # Post-harvest season - lower prices
            seasonal_factor = 0.85 + random.uniform(0, 0.1)
        elif month in [6, 7, 8]:  # Pre-harvest/monsoon - higher prices
            seasonal_factor = 1.15 + random.uniform(0, 0.1)
        else:
            seasonal_factor = 0.95 + random.uniform(0, 0.15)
        
        price *= seasonal_factor
        
        # Add random market fluctuation
        market_factor = random.uniform(0.8, 1.3)
        price *= market_factor
        
        # Generate supply and demand factors
        supply_volume = random.uniform(500, 2000)  # tonnes
        demand_factor = random.uniform(0.7, 1.4)
        
        # Market type affects price
        market_types = ['main_market', 'wholesale', 'export_hub']
        market_type = random.choice(market_types)
        market_name = f"{state}_{market_type}"
        
        if market_type == 'export_hub':
            price *= 1.2  # Export markets typically offer higher prices
        elif market_type == 'wholesale':
            price *= 1.1  # Wholesale markets offer moderate premiums
        
        sample = {
            'date': date,
            'commodity': commodity,
            'state': state,
            'market': market_name,
            'price_per_kg': round(max(price, base_price * 0.5), 2),  # Ensure minimum price
            'supply_volume': supply_volume,
            'demand_factor': demand_factor
        }
        
        data.append(sample)
    
    df = pd.DataFrame(data)
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

def add_noise_to_data(data, noise_level=0.05):
    """Add realistic noise to the generated data"""
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if col not in ['date']:
            noise = np.random.normal(0, data[col].std() * noise_level, len(data))
            data[col] = data[col] + noise
    
    return data
