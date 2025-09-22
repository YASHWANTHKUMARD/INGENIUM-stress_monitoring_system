from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import io
from typing import Optional
from fastapi.encoders import jsonable_encoder
import asyncio
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.stress_detector import StressDetector
from models.recommendation_engine import RecommendationEngine
from config import Config
import requests
import numpy as np

app = FastAPI(
    title="Stress Monitoring System API",
    description="AI/ML-powered stress monitoring with personalized recommendations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
stress_detector = None
recommendation_engine = None

# Simple in-memory connection manager for WebSocket broadcasting
class ConnectionManager:
    def __init__(self):
        self._active: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self._active.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self._active:
                self._active.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        data = jsonable_encoder(_sanitize_numpy(message))
        async with self._lock:
            websockets = list(self._active)
        for ws in websockets:
            try:
                await ws.send_json(data)
            except Exception:
                # Drop dead connections silently
                await self.disconnect(ws)

manager = ConnectionManager()

# Pydantic models for API
class StressData(BaseModel):
    heart_rate: float
    sleep_hours: float
    activity_level: float
    mood_score: float
    work_stress: float
    social_interaction: float
    caffeine_intake: float
    exercise_frequency: float
    age: int
    gender: int  # 0: Female, 1: Male

class UserPreferences(BaseModel):
    interests: Optional[List[str]] = []
    available_time: Optional[str] = "1-2 hours daily"
    preferred_activities: Optional[List[str]] = []
    music_preferences: Optional[List[str]] = []
    book_genres: Optional[List[str]] = []

class StressPrediction(BaseModel):
    stress_level: int
    stress_label: str
    confidence: float
    probabilities: Dict[str, float]

class StreamSample(BaseModel):
    heart_rate: float
    sleep_hours: float
    activity_level: float
    mood_score: float
    work_stress: float
    social_interaction: float
    caffeine_intake: float
    exercise_frequency: float
    age: int
    gender: int
    timestamp: Optional[str] = None  # ISO8601 (client-supplied optional)

class Recommendations(BaseModel):
    immediate_actions: List[str]
    books: List[str]
    music: List[str]
    activities: List[str]
    meditation: List[str]
    social: List[str]
    long_term: List[str]

class CoachChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None

class CoachChatResponse(BaseModel):
    reply: str
    sos: bool = False
    disclaimer: str = "This assistant provides general wellness information and is not a substitute for professional medical advice. If you are in crisis, seek immediate help."

# Dependency to get stress detector
def get_stress_detector():
    global stress_detector
    if stress_detector is None:
        stress_detector = StressDetector("random_forest")
        # Try to load existing model, otherwise train a new one
        try:
            stress_detector.load_model(Config.MODEL_PATH, Config.SCALER_PATH)
        except:
            print("No existing model found, training new model...")
            stress_detector.train()
            stress_detector.save_model(Config.MODEL_PATH, Config.SCALER_PATH)
    return stress_detector

# Dependency to get recommendation engine
def get_recommendation_engine():
    global recommendation_engine
    if recommendation_engine is None:
        try:
            recommendation_engine = RecommendationEngine()
        except ValueError as e:
            print(f"Warning: {e}")
            recommendation_engine = None
    return recommendation_engine

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("Initializing stress monitoring system...")
    # Initialize stress detector
    get_stress_detector()
    # Initialize recommendation engine
    get_recommendation_engine()
    print("System initialized successfully!")

def _sanitize_numpy(obj: Any) -> Any:
    """Recursively convert NumPy types to plain Python for JSON encoding."""
    try:
        if isinstance(obj, dict):
            return {k: _sanitize_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_numpy(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_sanitize_numpy(x) for x in obj)
        # Scalars
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj
    except Exception:
        return obj

def _triage_needs_sos(text: str) -> bool:
    if not text:
        return False
    keywords = [
        "suicide", "kill myself", "self harm", "overdose", "can't breathe",
        "chest pain", "heart attack", "stroke", "fainting", "unconscious"
    ]
    lowered = text.lower()
    return any(k in lowered for k in keywords)

@app.post("/api/coach/chat", response_model=CoachChatResponse)
async def coach_chat(req: CoachChatRequest):
    """Proxy chat to local Ollama with minimal safety triage."""
    try:
        sos = _triage_needs_sos(req.message)
        if sos:
            reply = (
                "This sounds urgent. Please seek immediate help: call local emergency services or visit the nearest ER. "
                "Try slow breathing while you get help: inhale 4s, hold 4s, exhale 6s."
            )
            return CoachChatResponse(reply=reply, sos=True)

        ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        url = f"{ollama_host.rstrip('/')}/api/generate"
        system_prompt = (
            "You are a supportive wellness assistant. Provide calm, practical, evidence-informed guidance for stress relief. "
            "Do not diagnose. Suggest simple actions (breathing, grounding, hydration, short walk, social support). "
            "Answer concisely: use at most 3 short bullet points and a single-line caution when urgent care is needed."
        )
        # Build a concise prompt with optional short history
        context = "\n".join(
            [f"{m.get('role','user')}: {m.get('content','')}" for m in (req.history or [])][-6:]
        )
        prompt = f"System: {system_prompt}\n{context}\nUser: {req.message}\nAssistant:"
        payload = {
            "model": os.getenv("OLLAMA_MODEL", "llama3"),
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3}
        }
        r = requests.post(url, json=payload, timeout=60)
        if not r.ok:
            raise HTTPException(status_code=502, detail="Ollama not reachable or returned error. Ensure it is running and OLLAMA_HOST/OLLAMA_MODEL are set.")
        data = r.json()
        # Be tolerant to different response shapes
        reply = (
            data.get("response")
            or data.get("message")
            or (data.get("choices", [{}])[0] or {}).get("message", {}).get("content")
            or ""
        )
        if isinstance(reply, dict):
            # Some backends may nest content
            reply = reply.get("content", "")
        if not reply:
            raise HTTPException(status_code=502, detail="Empty response from Ollama.")
        return CoachChatResponse(reply=reply, sos=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coach chat failed: {str(e)}")

@app.post("/api/stream/ingest")
async def ingest_stream_sample(sample: StreamSample, detector: StressDetector = Depends(get_stress_detector)):
    """Ingest a single wearable/phone sample, run prediction, and broadcast to subscribers."""
    try:
        payload = sample.dict()
        # Ensure timestamp
        if not payload.get("timestamp"):
            payload["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Build features expected by the model
        features = {k: payload[k] for k in [
            "heart_rate","sleep_hours","activity_level","mood_score","work_stress",
            "social_interaction","caffeine_intake","exercise_frequency","age","gender"
        ]}

        pred = _sanitize_numpy(detector.predict(features))
        event = _sanitize_numpy({
            "type": "prediction",
            "timestamp": payload["timestamp"],
            "input": features,
            "output": pred,
        })
        # Broadcast asynchronously (fire-and-forget)
        asyncio.create_task(manager.broadcast(event))
        return event
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Keep the connection alive; server pushes predictions on ingest
        while True:
            # Optionally receive pings or client filters; for now, just await to detect disconnects
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard"""
    # Read the advanced interactive frontend HTML
    try:
        with open("frontend/advanced_dashboard.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        try:
            with open("frontend/index.html", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            # Fallback to simple page if frontend file not found
            return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stress Monitoring System</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #333; text-align: center; margin-bottom: 30px; }
                .api-info { background: #e8f4f8; padding: 20px; border-radius: 5px; margin: 20px 0; }
                .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }
                .method { font-weight: bold; color: #007bff; }
                .description { margin-top: 5px; color: #666; }
                .btn { background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 5px; }
                .btn:hover { background: #0056b3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧠 Stress Monitoring System</h1>
                <p style="text-align: center; color: #666; font-size: 18px;">
                    AI/ML-powered stress detection with personalized recommendations
                </p>
                
                <div class="api-info">
                    <h2>API Endpoints</h2>
                    
                    <div class="endpoint">
                        <div class="method">POST /api/stress/predict</div>
                        <div class="description">Predict stress level based on user data</div>
                    </div>
                    
                    <div class="endpoint">
                        <div class="method">GET /api/recommendations</div>
                        <div class="description">Get personalized recommendations for stress management</div>
                    </div>
                    
                    <div class="endpoint">
                        <div class="method">GET /api/emergency</div>
                        <div class="description">Get immediate emergency stress relief recommendations</div>
                    </div>
                    
                    <div class="endpoint">
                        <div class="method">GET /api/weekly-plan</div>
                        <div class="description">Get a weekly stress management plan</div>
                    </div>
                    
                    <div class="endpoint">
                        <div class="method">GET /api/health</div>
                        <div class="description">Check system health and model status</div>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 30px;">
                    <a href="/docs" class="btn">📚 API Documentation</a>
                    <a href="/api/health" class="btn">🔍 System Health</a>
                </div>
            </div>
        </body>
        </html>
        """

@app.post("/api/stress/predict", response_model=StressPrediction)
async def predict_stress(
    stress_data: StressData,
    detector: StressDetector = Depends(get_stress_detector)
):
    """Predict stress level based on user data"""
    try:
        features = stress_data.dict()
        prediction = detector.predict(features)
        return StressPrediction(**prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/recommendations", response_model=Recommendations)
async def get_recommendations(
    stress_data: StressData,
    user_preferences: UserPreferences = None,
    detector: StressDetector = Depends(get_stress_detector),
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Get personalized recommendations based on stress level"""
    try:
        # First predict stress level
        features = stress_data.dict()
        stress_prediction = detector.predict(features)
        
        # Generate recommendations
        if engine:
            recommendations = engine.generate_recommendations(
                stress_prediction, 
                user_preferences.dict() if user_preferences else None
            )
        else:
            # Fallback to basic recommendations
            recommendations = {
                'immediate_actions': ["Take deep breaths", "Step away from stressor"],
                'books': ["The Power of Now", "Atomic Habits"],
                'music': ["Calming instrumental music", "Nature sounds"],
                'activities': ["Gentle exercise", "Meditation"],
                'meditation': ["10-minute breathing exercise", "Body scan"],
                'social': ["Talk to a friend", "Join support group"],
                'long_term': ["Regular exercise", "Stress management training"]
            }
        
        return Recommendations(**recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")

@app.get("/api/emergency")
async def get_emergency_recommendations(
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Get immediate emergency stress relief recommendations"""
    try:
        if engine:
            recommendations = engine.get_emergency_recommendations()
        else:
            recommendations = [
                "Stop and take 5 deep breaths",
                "Use the 5-4-3-2-1 grounding technique",
                "Step away from the current situation",
                "Call a trusted friend or family member"
            ]
        return {"emergency_recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get emergency recommendations: {str(e)}")

@app.get("/api/weekly-plan")
async def get_weekly_plan(
    stress_level: int = 0,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Get a weekly stress management plan"""
    try:
        if engine:
            plan = engine.get_weekly_plan(stress_level)
        else:
            plan = {
                'monday': ["Morning meditation", "Regular exercise"],
                'tuesday': ["Work-life balance check", "Social activity"],
                'wednesday': ["Mid-week stress check", "Nature walk"],
                'thursday': ["Professional development", "Social connection"],
                'friday': ["Week reflection", "Fun activity"],
                'saturday': ["Family time", "Outdoor activity"],
                'sunday': ["Rest and recovery", "Next week planning"]
            }
        return {"weekly_plan": plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get weekly plan: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Check system health and model status"""
    try:
        detector_status = "Ready" if stress_detector and stress_detector.is_trained else "Not Ready"
        engine_status = "Ready" if recommendation_engine else "Not Ready (API key required)"
        
        return {
            "status": "Healthy",
            "models": {
                "stress_detector": detector_status,
                "recommendation_engine": engine_status
            },
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "Unhealthy",
            "error": str(e),
            "version": "1.0.0"
        }

@app.get("/api/sample-data")
async def get_sample_data():
    """Get sample stress data for testing"""
    return {
        "sample_stress_data": {
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
        },
        "sample_preferences": {
            "interests": ["reading", "music", "nature"],
            "available_time": "1-2 hours daily",
            "preferred_activities": ["walking", "meditation", "reading"],
            "music_preferences": ["classical", "ambient", "jazz"],
            "book_genres": ["self-help", "psychology", "fiction"]
        }
    }

@app.get("/api/dataset")
async def get_dataset(n: int = 1000):
    """Return a generated sample dataset (JSON) with metadata"""
    try:
        detector = get_stress_detector()
        import pandas as pd  # local import to avoid startup cost if unused
        n = max(10, min(n, 10000))
        df = detector.create_sample_data(n)
        preview = df.head(50).to_dict(orient="records")
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "preview": preview
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build dataset: {str(e)}")

@app.get("/download/dataset.csv")
async def download_dataset_csv(n: int = 1000):
    """Download a generated sample dataset as CSV"""
    try:
        detector = get_stress_detector()
        n = max(10, min(n, 10000))
        df = detector.create_sample_data(n)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=stress_dataset_{n}.csv"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate CSV: {str(e)}")

@app.get("/api/dataset/filter")
async def filter_dataset(
    n: int = 1000,
    stress_level: Optional[int] = None,
    min_sleep: Optional[float] = None,
    max_sleep: Optional[float] = None,
    min_hr: Optional[float] = None,
    max_hr: Optional[float] = None,
    min_work_stress: Optional[float] = None,
    max_work_stress: Optional[float] = None,
):
    """Filter the generated dataset and return a compact summary + preview"""
    try:
        detector = get_stress_detector()
        import pandas as pd
        n = max(10, min(n, 10000))
        df = detector.create_sample_data(n)

        # Apply filters
        if stress_level is not None:
            df = df[df["stress_level"] == int(stress_level)]
        if min_sleep is not None:
            df = df[df["sleep_hours"] >= float(min_sleep)]
        if max_sleep is not None:
            df = df[df["sleep_hours"] <= float(max_sleep)]
        if min_hr is not None:
            df = df[df["heart_rate"] >= float(min_hr)]
        if max_hr is not None:
            df = df[df["heart_rate"] <= float(max_hr)]
        if min_work_stress is not None:
            df = df[df["work_stress"] >= float(min_work_stress)]
        if max_work_stress is not None:
            df = df[df["work_stress"] <= float(max_work_stress)]

        # Build summary
        summary = df.describe(include="all").to_dict()
        preview = df.head(50).to_dict(orient="records")
        return {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "summary": summary,
            "preview": preview,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to filter dataset: {str(e)}")

@app.get("/api/analytics/trends")
async def get_analytics_trends():
    """Get stress trend analytics"""
    return {
        "weekly_trend": [
            {"day": "Monday", "stress_level": 2, "mood": 6, "activity": 0.7},
            {"day": "Tuesday", "stress_level": 3, "mood": 5, "activity": 0.6},
            {"day": "Wednesday", "stress_level": 1, "mood": 8, "activity": 0.8},
            {"day": "Thursday", "stress_level": 4, "mood": 4, "activity": 0.5},
            {"day": "Friday", "stress_level": 2, "mood": 7, "activity": 0.6},
            {"day": "Saturday", "stress_level": 1, "mood": 9, "activity": 0.9},
            {"day": "Sunday", "stress_level": 1, "mood": 8, "activity": 0.7}
        ],
        "monthly_insights": {
            "avg_stress": 2.1,
            "improvement": 15.3,
            "best_day": "Saturday",
            "challenging_day": "Thursday"
        },
        "correlations": {
            "sleep_vs_stress": -0.78,
            "activity_vs_mood": 0.65,
            "work_stress_vs_overall": 0.82
        }
    }

@app.get("/api/ai-coach/insights")
async def get_ai_coach_insights():
    """Get AI coach personalized insights"""
    return {
        "personalized_message": "Based on your recent patterns, I notice you tend to feel more stressed on Thursdays. Let's work on some Thursday-specific strategies!",
        "recommendations": [
            "Try a 5-minute breathing exercise every Thursday morning",
            "Schedule lighter tasks for Thursday afternoons",
            "Plan a relaxing activity for Thursday evenings"
        ],
        "progress_tips": [
            "Your sleep quality has improved 20% this week!",
            "Consider adding 10 more minutes of daily activity",
            "Your stress management techniques are working well"
        ],
        "next_goals": [
            "Maintain your current sleep schedule",
            "Try one new relaxation technique this week",
            "Connect with a friend or family member"
        ]
    }

@app.get("/api/gamification/achievements")
async def get_achievements():
    """Get user achievements and progress"""
    return {
        "achievements": [
            {
                "id": "first_check",
                "title": "First Steps",
                "description": "Completed your first stress assessment",
                "icon": "🎯",
                "unlocked": True,
                "date": "2024-01-15"
            },
            {
                "id": "week_streak",
                "title": "Consistency Champion",
                "description": "Used the app for 7 consecutive days",
                "icon": "🔥",
                "unlocked": True,
                "date": "2024-01-22"
            },
            {
                "id": "stress_master",
                "title": "Stress Master",
                "description": "Maintained low stress for 5 days straight",
                "icon": "🧘",
                "unlocked": False,
                "progress": 3
            },
            {
                "id": "community_helper",
                "title": "Community Helper",
                "description": "Helped 5 community members",
                "icon": "🤝",
                "unlocked": False,
                "progress": 2
            }
        ],
        "current_streak": 12,
        "total_points": 1250,
        "level": "Wellness Warrior",
        "next_level_points": 500
    }

@app.get("/api/wearable/sync")
async def sync_wearable_data():
    """Simulate wearable device data sync"""
    import random
    return {
        "heart_rate": {
            "current": random.randint(60, 100),
            "avg_resting": random.randint(55, 75),
            "variability": random.uniform(20, 50)
        },
        "sleep": {
            "duration": random.uniform(6, 9),
            "quality": random.uniform(70, 95),
            "deep_sleep": random.uniform(15, 25),
            "rem_sleep": random.uniform(20, 30)
        },
        "activity": {
            "steps": random.randint(5000, 15000),
            "calories": random.randint(1800, 2500),
            "active_minutes": random.randint(30, 120)
        },
        "stress_indicators": {
            "hrv": random.uniform(30, 60),
            "skin_temp": random.uniform(36, 37),
            "movement_restlessness": random.uniform(0, 1)
        }
    }

@app.post("/api/community/share")
async def share_achievement(achievement_data: dict):
    """Share achievement with community"""
    return {
        "success": True,
        "message": "Achievement shared successfully!",
        "community_response": {
            "likes": 15,
            "comments": 3,
            "encouragement": "Great job! Keep up the amazing work! 💪"
        }
    }

@app.get("/api/meditation/sessions")
async def get_meditation_sessions():
    """Get available meditation sessions"""
    return {
        "sessions": [
            {
                "id": "breathing_basics",
                "title": "5-Minute Breathing Basics",
                "duration": 5,
                "difficulty": "Beginner",
                "description": "Perfect for beginners to start their meditation journey",
                "audio_url": "/api/meditation/audio/breathing_basics"
            },
            {
                "id": "stress_relief",
                "title": "10-Minute Stress Relief",
                "duration": 10,
                "difficulty": "Intermediate",
                "description": "Designed specifically for stress reduction",
                "audio_url": "/api/meditation/audio/stress_relief"
            },
            {
                "id": "body_scan",
                "title": "15-Minute Body Scan",
                "duration": 15,
                "difficulty": "Intermediate",
                "description": "Progressive relaxation technique for deep calm",
                "audio_url": "/api/meditation/audio/body_scan"
            }
        ]
    }

@app.get("/api/ai-coach/voice-interaction")
async def get_voice_interaction():
    """Get AI coach voice interaction capabilities"""
    return {
        "voice_enabled": True,
        "available_commands": [
            "Tell me about my stress level",
            "Give me a breathing exercise",
            "What should I do right now?",
            "Play calming music",
            "Schedule a meditation session"
        ],
        "current_mood": "calm",
        "suggested_response": "I sense you might be feeling a bit overwhelmed. Would you like to try a quick breathing exercise together?"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.DEBUG
    )
