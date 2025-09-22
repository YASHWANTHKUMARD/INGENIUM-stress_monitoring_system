#!/usr/bin/env python3
"""
Training script for the stress monitoring system
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.stress_detector import StressDetector
from config import Config

def train_model(model_type="random_forest", n_samples=1000, save_model=True):
    """
    Train the stress detection model
    
    Args:
        model_type: Type of model to train ('random_forest', 'svm', 'neural_network')
        n_samples: Number of samples to generate for training
        save_model: Whether to save the trained model
    """
    print(f"Training {model_type} model for stress detection...")
    print(f"Generating {n_samples} training samples...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Initialize and train the model
    detector = StressDetector(model_type)
    
    # Train the model
    results = detector.train()
    
    print(f"\nTraining Results:")
    print(f"Model Type: {results['model_type']}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"\nClassification Report:")
    print(results['classification_report'])
    
    if save_model:
        # Save the model
        model_path = Config.MODEL_PATH
        scaler_path = Config.SCALER_PATH
        
        detector.save_model(model_path, scaler_path)
        print(f"\nModel saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
    
    return detector, results

def test_model(detector, test_cases=None):
    """
    Test the trained model with sample data
    
    Args:
        detector: Trained StressDetector instance
        test_cases: List of test cases, if None uses default test cases
    """
    if test_cases is None:
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
    
    print(f"\nTesting model with {len(test_cases)} test cases:")
    print("=" * 60)
    
    for test_case in test_cases:
        print(f"\nTest Case: {test_case['name']}")
        print("-" * 40)
        
        try:
            prediction = detector.predict(test_case['data'])
            
            print(f"Predicted Stress Level: {prediction['stress_label']}")
            print(f"Confidence: {prediction['confidence']:.3f}")
            print(f"Probabilities:")
            for level, prob in prediction['probabilities'].items():
                print(f"  {level.capitalize()}: {prob:.3f}")
                
        except Exception as e:
            print(f"Error testing case: {e}")

def main():
    parser = argparse.ArgumentParser(description='Train stress detection model')
    parser.add_argument('--model-type', choices=['random_forest', 'svm', 'neural_network'], 
                       default='random_forest', help='Type of model to train')
    parser.add_argument('--samples', type=int, default=1000, 
                       help='Number of training samples to generate')
    parser.add_argument('--no-save', action='store_true', 
                       help='Do not save the trained model')
    parser.add_argument('--test-only', action='store_true', 
                       help='Only test existing model, do not train')
    
    args = parser.parse_args()
    
    if args.test_only:
        # Load existing model and test
        try:
            detector = StressDetector(args.model_type)
            detector.load_model(Config.MODEL_PATH, Config.SCALER_PATH)
            print("Loaded existing model for testing...")
            test_model(detector)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please train a model first using: python train_model.py")
    else:
        # Train new model
        detector, results = train_model(
            model_type=args.model_type,
            n_samples=args.samples,
            save_model=not args.no_save
        )
        
        # Test the trained model
        test_model(detector)

if __name__ == "__main__":
    main()
