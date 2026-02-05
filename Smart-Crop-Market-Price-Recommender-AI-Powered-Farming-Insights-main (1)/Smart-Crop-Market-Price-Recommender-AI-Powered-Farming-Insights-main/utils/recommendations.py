import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class RecommendationEngine:
    def __init__(self, crop_model, price_model):
        self.crop_model = crop_model
        self.price_model = price_model
    
    def get_comprehensive_recommendation(self, input_features, state):
        """Get comprehensive farming recommendations combining crop suitability and market analysis"""
        
        try:
            # Get crop recommendations
            crop_predictions = self.crop_model.predict_with_probability(input_features.reshape(1, -1))
            top_crops = crop_predictions[:3]  # Top 3 crop recommendations
            
            # Analyze each crop for market potential
            crop_analysis = []
            for crop_pred in top_crops:
                crop = crop_pred['crop']
                suitability = crop_pred['suitability_score']
                
                # Get market recommendations for this crop
                market_recs = self.price_model.get_market_recommendations(crop, state)
                
                if market_recs:
                    best_market = market_recs[0]
                    expected_price = best_market['expected_price']
                    
                    # Calculate profit potential (simplified)
                    base_price = self._get_base_price(crop)
                    profit_margin = ((expected_price - base_price) / base_price) * 100
                    
                    # Combined score: suitability + market potential
                    market_score = min((expected_price / base_price) * 50, 100)
                    combined_score = (suitability * 0.6) + (market_score * 0.4)
                    
                    crop_analysis.append({
                        'crop': crop,
                        'suitability_score': suitability,
                        'market_score': market_score,
                        'combined_score': combined_score,
                        'expected_price': expected_price,
                        'profit_margin': profit_margin,
                        'best_market': best_market['market']
                    })
            
            # Sort by combined score
            crop_analysis.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Get the best recommendation
            best_recommendation = crop_analysis[0] if crop_analysis else None
            
            if not best_recommendation:
                return None
            
            # Generate insights and recommendations
            insights = self._generate_insights(input_features, best_recommendation, state)
            recommendations = self._generate_recommendations(best_recommendation, crop_analysis)
            
            return {
                'recommended_crop': best_recommendation['crop'],
                'crop_analysis': crop_analysis,
                'market_recommendation': {
                    'best_market': best_recommendation['best_market'],
                    'expected_price': best_recommendation['expected_price'],
                    'profit_margin': best_recommendation['profit_margin']
                },
                'insights': insights,
                'recommendations': recommendations,
                'risk_assessment': self._assess_risks(input_features, best_recommendation)
            }
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return None
    
    def _get_base_price(self, crop):
        """Get base price for a crop (cost price approximation)"""
        base_prices = {
            'rice': 35, 'wheat': 20, 'cotton': 50, 'maize': 15,
            'sugarcane': 25, 'barley': 18, 'groundnut': 60, 'soybean': 40
        }
        return base_prices.get(crop.lower(), 30)
    
    def _generate_insights(self, input_features, best_rec, state):
        """Generate actionable insights based on the analysis"""
        insights = []
        
        # Soil condition insights
        n, p, k, temp, humidity, ph, rainfall = input_features
        
        if ph < 6.0:
            insights.append("Soil is acidic - consider lime application to improve pH")
        elif ph > 8.0:
            insights.append("Soil is alkaline - consider sulfur application to reduce pH")
        
        if n < 50:
            insights.append("Low nitrogen levels - apply nitrogen fertilizers before planting")
        elif n > 150:
            insights.append("High nitrogen levels - reduce nitrogen fertilizer to avoid excess vegetative growth")
        
        if rainfall < 75:
            insights.append("Low rainfall region - ensure adequate irrigation facilities")
        elif rainfall > 250:
            insights.append("High rainfall area - ensure proper drainage to prevent waterlogging")
        
        # Market insights
        if best_rec['profit_margin'] > 20:
            insights.append(f"Excellent market opportunity with {best_rec['profit_margin']:.1f}% profit margin")
        elif best_rec['profit_margin'] > 10:
            insights.append(f"Good market potential with {best_rec['profit_margin']:.1f}% profit margin")
        else:
            insights.append("Moderate market conditions - focus on cost optimization")
        
        # Seasonal insights
        current_month = datetime.now().month
        if current_month in [10, 11, 12, 1]:
            insights.append("Peak selling season - good time for harvest and sales")
        elif current_month in [3, 4, 5]:
            insights.append("Pre-monsoon season - prepare for planting")
        
        return insights
    
    def _generate_recommendations(self, best_rec, all_analysis):
        """Generate specific recommendations for the farmer"""
        recommendations = []
        
        # Crop-specific recommendations
        crop = best_rec['crop'].lower()
        
        if crop == 'rice':
            recommendations.append("Ensure proper water management - rice requires consistent flooding")
            recommendations.append("Apply phosphorus fertilizer during transplanting")
        elif crop == 'wheat':
            recommendations.append("Plant during winter season (Nov-Dec) for optimal yield")
            recommendations.append("Ensure adequate drainage to prevent fungal diseases")
        elif crop == 'cotton':
            recommendations.append("Monitor for bollworm pest - use integrated pest management")
            recommendations.append("Ensure adequate potassium for fiber quality")
        elif crop == 'maize':
            recommendations.append("Plant in rows for better sunlight exposure")
            recommendations.append("Side-dress with nitrogen during tasseling stage")
        elif crop == 'sugarcane':
            recommendations.append("Ensure 18-month crop cycle for optimal sugar content")
            recommendations.append("Apply organic matter to improve soil structure")
        
        # Market recommendations
        if best_rec['profit_margin'] > 15:
            recommendations.append(f"Target {best_rec['best_market']} for maximum profits")
        
        # Alternative crop suggestions
        if len(all_analysis) > 1:
            second_best = all_analysis[1]
            recommendations.append(f"Consider {second_best['crop']} as alternative with {second_best['suitability_score']:.1f}% suitability")
        
        # Timing recommendations
        recommendations.append("Monitor weather forecasts before planting")
        recommendations.append("Consider crop insurance for risk mitigation")
        
        return recommendations
    
    def _assess_risks(self, input_features, best_rec):
        """Assess various risks associated with the recommendation"""
        risks = []
        
        n, p, k, temp, humidity, ph, rainfall = input_features
        
        # Weather risks
        if temp > 35:
            risks.append("High temperature risk - may affect crop yield")
        if temp < 15:
            risks.append("Low temperature risk - may delay growth")
        
        if humidity > 90:
            risks.append("High humidity - increased disease pressure")
        if humidity < 40:
            risks.append("Low humidity - water stress risk")
        
        if rainfall > 250:
            risks.append("Excess rainfall risk - potential flooding")
        if rainfall < 50:
            risks.append("Drought risk - irrigation dependency")
        
        # Soil risks
        if ph < 5.5 or ph > 8.5:
            risks.append("Extreme pH levels - nutrient availability issues")
        
        # Market risks
        if best_rec['profit_margin'] < 5:
            risks.append("Low profit margin - market price volatility risk")
        
        # Overall risk level
        risk_score = len(risks)
        if risk_score <= 2:
            risk_level = "Low"
        elif risk_score <= 4:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        return {
            'risk_level': risk_level,
            'risk_factors': risks,
            'risk_score': risk_score
        }
    
    def get_seasonal_recommendations(self, state, month=None):
        """Get seasonal farming recommendations"""
        if month is None:
            month = datetime.now().month
        
        seasonal_advice = {
            1: "Winter season - ideal for wheat, barley, and rabi crops",
            2: "Late winter - prepare for harvest of rabi crops",
            3: "Spring season - post-harvest activities and summer crop preparation",
            4: "Pre-summer - sowing of summer crops like cotton, sugarcane",
            5: "Summer season - irrigation management critical",
            6: "Pre-monsoon - prepare for kharif crop sowing",
            7: "Monsoon onset - sow rice, maize, cotton",
            8: "Peak monsoon - monitor drainage and pest control",
            9: "Late monsoon - disease management in standing crops",
            10: "Post-monsoon - harvest kharif crops",
            11: "Winter preparation - sow rabi crops",
            12: "Early winter - winter crop management"
        }
        
        return seasonal_advice.get(month, "Season-specific advice not available")
    
    def compare_crop_profitability(self, crops, state):
        """Compare profitability of multiple crops"""
        profitability_analysis = []
        
        for crop in crops:
            try:
                # Get market price
                price = self.price_model.predict_price(crop, state, f"{state}_main_market")
                
                # Get base cost
                base_cost = self._get_base_price(crop)
                
                # Calculate profit margin
                profit_margin = ((price - base_cost) / base_cost) * 100
                
                profitability_analysis.append({
                    'crop': crop,
                    'market_price': price,
                    'production_cost': base_cost,
                    'profit_margin': profit_margin,
                    'revenue_per_acre': price * 1000  # Assuming 1000 kg/acre average yield
                })
            except:
                continue
        
        # Sort by profit margin
        profitability_analysis.sort(key=lambda x: x['profit_margin'], reverse=True)
        
        return profitability_analysis
