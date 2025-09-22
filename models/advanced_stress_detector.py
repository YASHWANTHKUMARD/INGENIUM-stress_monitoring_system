import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class AdvancedStressDetector:
    """
    Advanced ML-based stress detection system using ensemble methods
    """
    
    def __init__(self):
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None
        self.model_performance = {}
        
    def create_advanced_dataset(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Create advanced stress detection dataset with more features
        """
        np.random.seed(42)
        
        # Base features
        data = {
            'heart_rate': np.random.normal(75, 15, n_samples),
            'sleep_hours': np.random.normal(7.5, 1.5, n_samples),
            'activity_level': np.random.uniform(0, 1, n_samples),
            'mood_score': np.random.uniform(1, 10, n_samples),
            'work_stress': np.random.uniform(1, 10, n_samples),
            'social_interaction': np.random.uniform(1, 10, n_samples),
            'caffeine_intake': np.random.uniform(0, 5, n_samples),
            'exercise_frequency': np.random.uniform(0, 7, n_samples),
            'age': np.random.randint(18, 65, n_samples),
            'gender': np.random.choice([0, 1], n_samples),
            
            # Advanced features
            'hrv': np.random.uniform(20, 60, n_samples),  # Heart Rate Variability
            'skin_temp': np.random.uniform(36, 37, n_samples),  # Skin Temperature
            'movement_restlessness': np.random.uniform(0, 1, n_samples),
            'screen_time': np.random.uniform(2, 12, n_samples),  # Hours of screen time
            'outdoor_time': np.random.uniform(0, 4, n_samples),  # Hours outdoors
            'social_media_usage': np.random.uniform(0, 8, n_samples),
            'meditation_frequency': np.random.uniform(0, 7, n_samples),
            'water_intake': np.random.uniform(1, 4, n_samples),  # Liters per day
            'meal_regularity': np.random.uniform(0, 1, n_samples),  # How regular meals are
            'noise_level': np.random.uniform(30, 80, n_samples),  # Environmental noise
        }
        
        df = pd.DataFrame(data)
        
        # Create more sophisticated stress scoring
        stress_factors = (
            (df['heart_rate'] > 85).astype(int) * 2 +
            (df['sleep_hours'] < 6).astype(int) * 3 +
            (df['activity_level'] < 0.3).astype(int) * 2 +
            (df['mood_score'] < 4).astype(int) * 3 +
            (df['work_stress'] > 7).astype(int) * 2 +
            (df['social_interaction'] < 3).astype(int) * 1 +
            (df['caffeine_intake'] > 3).astype(int) * 1 +
            (df['hrv'] < 30).astype(int) * 2 +
            (df['screen_time'] > 8).astype(int) * 1 +
            (df['outdoor_time'] < 0.5).astype(int) * 1 +
            (df['social_media_usage'] > 4).astype(int) * 1 +
            (df['meditation_frequency'] < 1).astype(int) * 1 +
            (df['water_intake'] < 2).astype(int) * 1 +
            (df['meal_regularity'] < 0.5).astype(int) * 1 +
            (df['noise_level'] > 60).astype(int) * 1
        )
        
        # Create stress levels with more nuanced distribution
        df['stress_level'] = pd.cut(stress_factors, 
                                  bins=[-1, 5, 10, 15, 25], 
                                  labels=[0, 1, 2, 3]).astype(int)
        
        # Add some realistic correlations
        df['heart_rate'] = df['heart_rate'] + (df['stress_level'] * 5)
        df['sleep_hours'] = df['sleep_hours'] - (df['stress_level'] * 0.5)
        df['mood_score'] = df['mood_score'] - (df['stress_level'] * 1.5)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for training
        """
        feature_columns = [
            'heart_rate', 'sleep_hours', 'activity_level', 'mood_score',
            'work_stress', 'social_interaction', 'caffeine_intake',
            'exercise_frequency', 'age', 'gender', 'hrv', 'skin_temp',
            'movement_restlessness', 'screen_time', 'outdoor_time',
            'social_media_usage', 'meditation_frequency', 'water_intake',
            'meal_regularity', 'noise_level'
        ]
        
        X = df[feature_columns].values
        y = df['stress_level'].values
        
        return X, y
    
    def create_ensemble_model(self):
        """
        Create advanced ensemble model
        """
        # Individual models
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=1000, random_state=42)
        
        # Ensemble model
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('svm', svm),
                ('gb', gb),
                ('mlp', mlp)
            ],
            voting='soft'
        )
    
    def train(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Train the advanced stress detection model
        """
        if df is None:
            df = self.create_advanced_dataset()
        
        X, y = self.prepare_features(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train ensemble model
        self.create_ensemble_model()
        self.ensemble_model.fit(X_train_scaled, y_train)
        
        # Evaluate individual models
        individual_scores = {}
        for name, model in self.ensemble_model.named_estimators_.items():
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            individual_scores[name] = score
        
        # Evaluate ensemble
        y_pred = self.ensemble_model.predict(X_test_scaled)
        ensemble_accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.ensemble_model, X_train_scaled, y_train, cv=5)
        
        # Feature importance (from Random Forest)
        rf_model = self.ensemble_model.named_estimators_['rf']
        self.feature_importance = rf_model.feature_importances_
        
        self.is_trained = True
        self.model_performance = {
            'ensemble_accuracy': ensemble_accuracy,
            'individual_scores': individual_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return self.model_performance
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict stress level for given features
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert features to array
        feature_order = [
            'heart_rate', 'sleep_hours', 'activity_level', 'mood_score',
            'work_stress', 'social_interaction', 'caffeine_intake',
            'exercise_frequency', 'age', 'gender', 'hrv', 'skin_temp',
            'movement_restlessness', 'screen_time', 'outdoor_time',
            'social_media_usage', 'meditation_frequency', 'water_intake',
            'meal_regularity', 'noise_level'
        ]
        
        X = np.array([features.get(f, 0) for f in feature_order]).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        stress_level = self.ensemble_model.predict(X_scaled)[0]
        stress_probability = self.ensemble_model.predict_proba(X_scaled)[0]
        
        stress_labels = {0: "Very Low", 1: "Low", 2: "Medium", 3: "High"}
        
        # Get individual model predictions
        individual_predictions = {}
        for name, model in self.ensemble_model.named_estimators_.items():
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0]
            individual_predictions[name] = {
                'prediction': pred,
                'confidence': float(max(prob))
            }
        
        return {
            'stress_level': int(stress_level),
            'stress_label': stress_labels[stress_level],
            'confidence': float(max(stress_probability)),
            'probabilities': {
                'very_low': float(stress_probability[0]),
                'low': float(stress_probability[1]),
                'medium': float(stress_probability[2]),
                'high': float(stress_probability[3])
            },
            'individual_predictions': individual_predictions,
            'model_agreement': len(set([p['prediction'] for p in individual_predictions.values()])) == 1
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        feature_names = [
            'heart_rate', 'sleep_hours', 'activity_level', 'mood_score',
            'work_stress', 'social_interaction', 'caffeine_intake',
            'exercise_frequency', 'age', 'gender', 'hrv', 'skin_temp',
            'movement_restlessness', 'screen_time', 'outdoor_time',
            'social_media_usage', 'meditation_frequency', 'water_intake',
            'meal_regularity', 'noise_level'
        ]
        
        return dict(zip(feature_names, self.feature_importance))
    
    def save_model(self, model_path: str, scaler_path: str):
        """
        Save the trained model and scaler
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.ensemble_model, model_path)
        joblib.dump(self.scaler, scaler_path)
    
    def load_model(self, model_path: str, scaler_path: str):
        """
        Load a pre-trained model and scaler
        """
        self.ensemble_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True

# Example usage
if __name__ == "__main__":
    # Create and train the advanced model
    detector = AdvancedStressDetector()
    
    print("Training advanced stress detection model...")
    results = detector.train()
    
    print(f"Ensemble Accuracy: {results['ensemble_accuracy']:.3f}")
    print(f"Cross-validation: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
    print("\nIndividual Model Performance:")
    for name, score in results['individual_scores'].items():
        print(f"  {name}: {score:.3f}")
    
    # Test prediction
    test_features = {
        'heart_rate': 90, 'sleep_hours': 5, 'activity_level': 0.3,
        'mood_score': 4, 'work_stress': 8, 'social_interaction': 3,
        'caffeine_intake': 4, 'exercise_frequency': 1, 'age': 35, 'gender': 1,
        'hrv': 25, 'skin_temp': 36.5, 'movement_restlessness': 0.7,
        'screen_time': 10, 'outdoor_time': 0.5, 'social_media_usage': 6,
        'meditation_frequency': 0, 'water_intake': 1.5, 'meal_regularity': 0.3,
        'noise_level': 70
    }
    
    prediction = detector.predict(test_features)
    print(f"\nTest Prediction: {prediction}")
    
    # Save the model
    detector.save_model("./models/advanced_stress_model.pkl", "./models/advanced_scaler.pkl")
    print("\nAdvanced model saved successfully!")
