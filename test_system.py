#!/usr/bin/env python3
"""
Test script for the stress monitoring system
"""

import sys
import os
import requests
import json
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.stress_detector import StressDetector
from models.recommendation_engine import RecommendationEngine
from config import Config

def test_stress_detector():
    """Test the stress detection model"""
    print("Testing Stress Detector...")
    print("=" * 40)
    
    # Initialize detector
    detector = StressDetector("random_forest")
    
    # Train model if not exists
    if not os.path.exists(Config.MODEL_PATH):
        print("Training new model...")
        detector.train()
        detector.save_model(Config.MODEL_PATH, Config.SCALER_PATH)
    else:
        print("Loading existing model...")
        detector.load_model(Config.MODEL_PATH, Config.SCALER_PATH)
    
    # Test cases
    test_cases = [
        {
            'name': 'Low Stress Individual',
            'data': {
                'heart_rate': 70, 'sleep_hours': 8, 'activity_level': 0.7,
                'mood_score': 8, 'work_stress': 3, 'social_interaction': 7,
                'caffeine_intake': 1, 'exercise_frequency': 5, 'age': 25, 'gender': 0
            }
        },
        {
            'name': 'Medium Stress Individual',
            'data': {
                'heart_rate': 85, 'sleep_hours': 6, 'activity_level': 0.4,
                'mood_score': 5, 'work_stress': 6, 'social_interaction': 4,
                'caffeine_intake': 3, 'exercise_frequency': 2, 'age': 35, 'gender': 1
            }
        },
        {
            'name': 'High Stress Individual',
            'data': {
                'heart_rate': 95, 'sleep_hours': 4, 'activity_level': 0.2,
                'mood_score': 3, 'work_stress': 9, 'social_interaction': 2,
                'caffeine_intake': 5, 'exercise_frequency': 1, 'age': 40, 'gender': 1
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        print("-" * 30)
        
        try:
            prediction = detector.predict(test_case['data'])
            print(f"Stress Level: {prediction['stress_label']}")
            print(f"Confidence: {prediction['confidence']:.3f}")
            print(f"Probabilities:")
            for level, prob in prediction['probabilities'].items():
                print(f"  {level.capitalize()}: {prob:.3f}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n✅ Stress Detector test completed!")

def test_recommendation_engine():
    """Test the recommendation engine"""
    print("\nTesting Recommendation Engine...")
    print("=" * 40)
    
    try:
        # Initialize engine
        engine = RecommendationEngine()
        
        # Test data
        stress_data = {
            'stress_level': 1,
            'stress_label': 'Medium',
            'confidence': 0.85,
            'probabilities': {'low': 0.15, 'medium': 0.85, 'high': 0.0}
        }
        
        user_preferences = {
            'interests': ['reading', 'music', 'nature'],
            'available_time': '1-2 hours daily',
            'preferred_activities': ['walking', 'meditation', 'reading']
        }
        
        # Generate recommendations
        recommendations = engine.generate_recommendations(stress_data, user_preferences)
        
        print("Generated Recommendations:")
        for category, items in recommendations.items():
            print(f"\n{category.upper()}:")
            for item in items[:3]:  # Show first 3 items
                print(f"  - {item}")
        
        print("\n✅ Recommendation Engine test completed!")
        
    except ValueError as e:
        print(f"⚠️  Recommendation Engine test skipped: {e}")
        print("   (OpenAI API key required for full functionality)")

def test_api():
    """Test the API endpoints"""
    print("\nTesting API...")
    print("=" * 40)
    
    base_url = f"http://{Config.API_HOST}:{Config.API_PORT}"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ API Health: {health_data['status']}")
            print(f"   Models: {health_data['models']}")
        else:
            print(f"❌ API Health check failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ API not accessible: {e}")
        print("   Make sure to start the API server first: python run_app.py")
        return
    
    # Test stress prediction endpoint
    try:
        test_data = {
            "heart_rate": 85.0,
            "sleep_hours": 6.5,
            "activity_level": 0.4,
            "mood_score": 6.0,
            "work_stress": 7.0,
            "social_interaction": 5.0,
            "caffeine_intake": 2.0,
            "exercise_frequency": 3.0,
            "age": 30,
            "gender": 1
        }
        
        response = requests.post(f"{base_url}/api/stress/predict", 
                               json=test_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Stress Prediction: {result['stress_label']} (confidence: {result['confidence']:.3f})")
        else:
            print(f"❌ Stress prediction failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Stress prediction request failed: {e}")
    
    # Test recommendations endpoint
    try:
        stress_data = {
            "heart_rate": 85.0,
            "sleep_hours": 6.5,
            "activity_level": 0.4,
            "mood_score": 6.0,
            "work_stress": 7.0,
            "social_interaction": 5.0,
            "caffeine_intake": 2.0,
            "exercise_frequency": 3.0,
            "age": 30,
            "gender": 1
        }
        
        preferences = {
            "interests": ["reading", "music"],
            "available_time": "1-2 hours daily",
            "preferred_activities": ["walking", "meditation"]
        }
        
        response = requests.post(f"{base_url}/api/recommendations", 
                               json={"stress_data": stress_data, "user_preferences": preferences}, 
                               timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Recommendations generated: {len(result)} categories")
        else:
            print(f"❌ Recommendations failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Recommendations request failed: {e}")
    
    print("\n✅ API test completed!")

def main():
    """Run all tests"""
    print("🧠 Stress Monitoring System - Test Suite")
    print("=" * 50)
    
    # Test stress detector
    test_stress_detector()
    
    # Test recommendation engine
    test_recommendation_engine()
    
    # Test API (if running)
    test_api()
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("\nTo start the full system:")
    print("1. Set your OpenAI API key: export OPENAI_API_KEY='your_key_here'")
    print("2. Run the application: python run_app.py")
    print("3. Open your browser: http://localhost:8000")

if __name__ == "__main__":
    main()
