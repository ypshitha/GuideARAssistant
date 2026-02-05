import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

class CropRecommender:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.scaler = StandardScaler()
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.crop_labels = []
        self.is_trained = False
    
    def train(self, data):
        """Train the crop recommendation model"""
        try:
            # Prepare features and target
            X = data[self.feature_names]
            y = data['label']
            
            self.crop_labels = y.unique().tolist()
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train the model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            
            print(f"Model trained successfully with accuracy: {accuracy:.3f}")
            return accuracy
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return None
    
    def predict(self, features):
        """Predict the best crop for given conditions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)
        return prediction[0]
    
    def predict_with_probability(self, features):
        """Predict crop with probability scores for all crops"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Create results with crop names and probabilities
        results = []
        for i, prob in enumerate(probabilities):
            results.append({
                'crop': self.model.classes_[i],
                'probability': prob,
                'suitability_score': prob * 100
            })
        
        # Sort by probability (descending)
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return results
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance))
        
        # Sort by importance
        sorted_importance = sorted(feature_importance.items(), key=lambda x: float(x[1]), reverse=True)
        
        return sorted_importance
    
    def get_crop_requirements(self, crop_name):
        """Get typical requirements for a specific crop"""
        # This is a simplified version - in production, this would come from agricultural databases
        crop_requirements = {
            'rice': {
                'N': '80-120', 'P': '40-60', 'K': '40-60',
                'temperature': '20-35°C', 'humidity': '70-90%',
                'ph': '5.5-7.0', 'rainfall': '150-300mm'
            },
            'wheat': {
                'N': '60-100', 'P': '30-50', 'K': '30-50',
                'temperature': '15-25°C', 'humidity': '50-70%',
                'ph': '6.0-7.5', 'rainfall': '50-100mm'
            },
            'cotton': {
                'N': '80-150', 'P': '40-80', 'K': '60-100',
                'temperature': '25-35°C', 'humidity': '60-80%',
                'ph': '6.0-8.0', 'rainfall': '75-150mm'
            },
            'maize': {
                'N': '100-150', 'P': '50-80', 'K': '50-80',
                'temperature': '20-30°C', 'humidity': '60-80%',
                'ph': '6.0-7.5', 'rainfall': '100-200mm'
            },
            'sugarcane': {
                'N': '120-200', 'P': '60-100', 'K': '80-120',
                'temperature': '25-35°C', 'humidity': '70-90%',
                'ph': '6.0-8.0', 'rainfall': '150-250mm'
            }
        }
        
        return crop_requirements.get(crop_name.lower(), {})
