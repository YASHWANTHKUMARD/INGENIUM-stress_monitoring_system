#!/usr/bin/env python3
"""
Main application runner for the stress monitoring system
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config

def main():
    """Run the stress monitoring application"""
    
    print("🧠 Starting Stress Monitoring System...")
    print("=" * 50)
    
    # Check if models exist, if not train them
    if not os.path.exists(Config.MODEL_PATH) or not os.path.exists(Config.SCALER_PATH):
        print("No trained models found. Training new models...")
        from train_model import train_model
        train_model(save_model=True)
        print("Model training completed!")
    
    # Check OpenAI API key
    if not Config.OPENAI_API_KEY:
        print("⚠️  Warning: OpenAI API key not found.")
        print("   The recommendation engine will use fallback recommendations.")
        print("   To enable AI-powered recommendations, set your OPENAI_API_KEY in the environment.")
    
    print(f"🚀 Starting server on {Config.API_HOST}:{Config.API_PORT}")
    print(f"📊 Dashboard: http://{Config.API_HOST}:{Config.API_PORT}")
    print(f"📚 API Docs: http://{Config.API_HOST}:{Config.API_PORT}/docs")
    print("=" * 50)
    
    # Run the application
    uvicorn.run(
        "api.main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.DEBUG,
        log_level="info"
    )

if __name__ == "__main__":
    main()
