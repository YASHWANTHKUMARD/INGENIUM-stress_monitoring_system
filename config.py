import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./stress_monitoring.db")
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model Configuration
    MODEL_PATH = os.getenv("MODEL_PATH", "./models/stress_model.pkl")
    SCALER_PATH = os.getenv("SCALER_PATH", "./models/scaler.pkl")
    
    # Stress Detection Thresholds
    HEART_RATE_THRESHOLD = int(os.getenv("HEART_RATE_THRESHOLD", 100))
    SLEEP_HOURS_THRESHOLD = int(os.getenv("SLEEP_HOURS_THRESHOLD", 7))
    ACTIVITY_LEVEL_THRESHOLD = float(os.getenv("ACTIVITY_LEVEL_THRESHOLD", 0.5))
    
    # Recommendation Categories
    RECOMMENDATION_CATEGORIES = [
        "books", "music", "activities", "meditation", 
        "exercise", "social", "hobbies", "therapy"
    ]
