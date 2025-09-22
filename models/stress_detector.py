import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from typing import Dict, List, Tuple, Any

class StressDetector:
    """
    ML-based stress detection system using multiple algorithms
    """
    
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "svm":
            self.model = SVC(kernel='rbf', random_state=42)
        elif model_type == "neural_network":
            self.model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        else:
            raise ValueError("Invalid model type. Choose from: random_forest, svm, neural_network")
    
    def create_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create sample stress detection dataset
        """
        np.random.seed(42)
        
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
            'gender': np.random.choice([0, 1], n_samples)  # 0: Female, 1: Male
        }
        
        df = pd.DataFrame(data)
        
        # Create stress labels based on features
        stress_score = (
            (df['heart_rate'] > 85).astype(int) * 2 +
            (df['sleep_hours'] < 6).astype(int) * 2 +
            (df['activity_level'] < 0.3).astype(int) * 1 +
            (df['mood_score'] < 4).astype(int) * 2 +
            (df['work_stress'] > 7).astype(int) * 2 +
            (df['social_interaction'] < 3).astype(int) * 1 +
            (df['caffeine_intake'] > 3).astype(int) * 1
        )
        
        # Convert to stress levels: 0=Low, 1=Medium, 2=High
        df['stress_level'] = pd.cut(stress_score, bins=[-1, 3, 6, 10], labels=[0, 1, 2]).astype(int)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for training
        """
        feature_columns = [
            'heart_rate', 'sleep_hours', 'activity_level', 'mood_score',
            'work_stress', 'social_interaction', 'caffeine_intake',
            'exercise_frequency', 'age', 'gender'
        ]
        
        X = df[feature_columns].values
        y = df['stress_level'].values
        
        return X, y
    
    def train(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Train the stress detection model
        """
        if df is None:
            df = self.create_sample_data()
        
        X, y = self.prepare_features(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'model_type': self.model_type
        }
    
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
            'exercise_frequency', 'age', 'gender'
        ]
        
        X = np.array([features.get(f, 0) for f in feature_order]).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        stress_level = self.model.predict(X_scaled)[0]
        stress_probability = self.model.predict_proba(X_scaled)[0]
        
        stress_labels = {0: "Low", 1: "Medium", 2: "High"}
        
        return {
            'stress_level': stress_level,
            'stress_label': stress_labels[stress_level],
            'confidence': float(max(stress_probability)),
            'probabilities': {
                'low': float(stress_probability[0]),
                'medium': float(stress_probability[1]),
                'high': float(stress_probability[2])
            }
        }
    
    def save_model(self, model_path: str, scaler_path: str):
        """
        Save the trained model and scaler
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
    
    def load_model(self, model_path: str, scaler_path: str):
        """
        Load a pre-trained model and scaler
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True

# Example usage and training
if __name__ == "__main__":
    # Create and train the model
    detector = StressDetector("random_forest")
    
    # Generate sample data and train
    print("Training stress detection model...")
    results = detector.train()
    print(f"Model accuracy: {results['accuracy']:.3f}")
    print(f"Classification Report:\n{results['classification_report']}")
    
    # Save the model
    detector.save_model("./models/stress_model.pkl", "./models/scaler.pkl")
    print("Model saved successfully!")
    
    # Test prediction
    test_features = {
        'heart_rate': 90,
        'sleep_hours': 5,
        'activity_level': 0.2,
        'mood_score': 3,
        'work_stress': 8,
        'social_interaction': 2,
        'caffeine_intake': 4,
        'exercise_frequency': 1,
        'age': 30,
        'gender': 1
    }
    
    prediction = detector.predict(test_features)
    print(f"\nTest Prediction: {prediction}")
