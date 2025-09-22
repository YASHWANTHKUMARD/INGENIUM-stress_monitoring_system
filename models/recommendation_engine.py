import openai
from typing import Dict, List, Any
import json
import random
from config import Config

class RecommendationEngine:
    """
    GenAI-powered recommendation engine for personalized stress management
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def generate_recommendations(self, stress_data: Dict[str, Any], user_preferences: Dict[str, Any] = None) -> Dict[str, List[str]]:
        """
        Generate personalized recommendations based on stress level and user data
        """
        stress_level = stress_data.get('stress_level', 0)
        stress_label = stress_data.get('stress_label', 'Low')
        
        # Base recommendations by stress level
        base_recommendations = self._get_base_recommendations(stress_level)
        
        # Generate AI-powered personalized recommendations
        ai_recommendations = self._generate_ai_recommendations(stress_data, user_preferences)
        
        # Combine base and AI recommendations
        recommendations = {
            'immediate_actions': base_recommendations['immediate_actions'],
            'books': ai_recommendations.get('books', base_recommendations['books']),
            'music': ai_recommendations.get('music', base_recommendations['music']),
            'activities': ai_recommendations.get('activities', base_recommendations['activities']),
            'meditation': ai_recommendations.get('meditation', base_recommendations['meditation']),
            'social': ai_recommendations.get('social', base_recommendations['social']),
            'long_term': ai_recommendations.get('long_term', base_recommendations['long_term'])
        }
        
        return recommendations
    
    def _get_base_recommendations(self, stress_level: int) -> Dict[str, List[str]]:
        """
        Get base recommendations based on stress level
        """
        if stress_level == 0:  # Low stress
            return {
                'immediate_actions': [
                    "Continue your current routine",
                    "Maintain healthy habits",
                    "Consider preventive stress management"
                ],
                'books': [
                    "The Power of Now by Eckhart Tolle",
                    "Atomic Habits by James Clear",
                    "The 7 Habits of Highly Effective People by Stephen Covey"
                ],
                'music': [
                    "Classical music for focus",
                    "Nature sounds for relaxation",
                    "Upbeat instrumental music"
                ],
                'activities': [
                    "Regular exercise routine",
                    "Creative hobbies",
                    "Social activities with friends"
                ],
                'meditation': [
                    "10-minute daily meditation",
                    "Mindfulness exercises",
                    "Breathing techniques"
                ],
                'social': [
                    "Regular social interactions",
                    "Join hobby groups",
                    "Family time activities"
                ],
                'long_term': [
                    "Maintain work-life balance",
                    "Regular health checkups",
                    "Personal development goals"
                ]
            }
        
        elif stress_level == 1:  # Medium stress
            return {
                'immediate_actions': [
                    "Take a 5-minute break",
                    "Practice deep breathing",
                    "Step away from stressful situation"
                ],
                'books': [
                    "The Stress-Proof Brain by Melanie Greenberg",
                    "Why Zebras Don't Get Ulcers by Robert Sapolsky",
                    "The Relaxation Response by Herbert Benson"
                ],
                'music': [
                    "Calming instrumental music",
                    "Binaural beats for relaxation",
                    "Soft acoustic music"
                ],
                'activities': [
                    "Gentle yoga or stretching",
                    "Walking in nature",
                    "Art therapy or journaling"
                ],
                'meditation': [
                    "15-minute guided meditation",
                    "Progressive muscle relaxation",
                    "Body scan meditation"
                ],
                'social': [
                    "Talk to a trusted friend",
                    "Join a support group",
                    "Schedule quality time with family"
                ],
                'long_term': [
                    "Identify stress triggers",
                    "Develop coping strategies",
                    "Consider professional help if needed"
                ]
            }
        
        else:  # High stress
            return {
                'immediate_actions': [
                    "Stop and take deep breaths",
                    "Remove yourself from the situation",
                    "Use grounding techniques"
                ],
                'books': [
                    "The Body Keeps the Score by Bessel van der Kolk",
                    "When the Body Says No by Gabor Maté",
                    "Burnout by Emily Nagoski"
                ],
                'music': [
                    "Solfeggio frequencies",
                    "White noise or rain sounds",
                    "Very slow, calming music"
                ],
                'activities': [
                    "Gentle stretching or yoga",
                    "Coloring or art therapy",
                    "Walking in a peaceful environment"
                ],
                'meditation': [
                    "20-minute guided meditation",
                    "Loving-kindness meditation",
                    "Emergency stress relief techniques"
                ],
                'social': [
                    "Reach out to support system immediately",
                    "Consider professional counseling",
                    "Join stress management groups"
                ],
                'long_term': [
                    "Seek professional help",
                    "Consider lifestyle changes",
                    "Develop comprehensive stress management plan"
                ]
            }
    
    def _generate_ai_recommendations(self, stress_data: Dict[str, Any], user_preferences: Dict[str, Any] = None) -> Dict[str, List[str]]:
        """
        Generate AI-powered personalized recommendations
        """
        try:
            # Prepare context for AI
            context = self._prepare_ai_context(stress_data, user_preferences)
            
            # Generate recommendations using OpenAI
            prompt = f"""
            Based on the following stress monitoring data and user preferences, provide personalized recommendations for stress management:
            
            Stress Data: {json.dumps(stress_data, indent=2)}
            User Preferences: {json.dumps(user_preferences or {}, indent=2)}
            
            Please provide specific, actionable recommendations in the following categories:
            1. Books (3-5 specific book titles with brief explanations)
            2. Music (3-5 specific music recommendations with genres/artists)
            3. Activities (3-5 specific activities tailored to their situation)
            4. Meditation (3-5 specific meditation techniques or apps)
            5. Social (3-5 specific social activities or support options)
            6. Long-term (3-5 specific long-term strategies)
            
            Format the response as a JSON object with these categories as keys and arrays of recommendations as values.
            Make recommendations specific, practical, and tailored to their stress level and preferences.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a stress management expert providing personalized recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            # Parse AI response
            ai_response = response.choices[0].message.content
            try:
                recommendations = json.loads(ai_response)
                return recommendations
            except json.JSONDecodeError:
                # Fallback to base recommendations if AI response is not valid JSON
                return {}
                
        except Exception as e:
            print(f"Error generating AI recommendations: {e}")
            return {}
    
    def _prepare_ai_context(self, stress_data: Dict[str, Any], user_preferences: Dict[str, Any] = None) -> str:
        """
        Prepare context for AI recommendation generation
        """
        context_parts = []
        
        # Stress level context
        stress_level = stress_data.get('stress_label', 'Unknown')
        confidence = stress_data.get('confidence', 0)
        context_parts.append(f"Current stress level: {stress_level} (confidence: {confidence:.2f})")
        
        # Feature context
        if 'probabilities' in stress_data:
            probs = stress_data['probabilities']
            context_parts.append(f"Stress probability distribution: Low={probs.get('low', 0):.2f}, Medium={probs.get('medium', 0):.2f}, High={probs.get('high', 0):.2f}")
        
        # User preferences context
        if user_preferences:
            if 'interests' in user_preferences:
                context_parts.append(f"User interests: {', '.join(user_preferences['interests'])}")
            if 'available_time' in user_preferences:
                context_parts.append(f"Available time: {user_preferences['available_time']}")
            if 'preferred_activities' in user_preferences:
                context_parts.append(f"Preferred activities: {', '.join(user_preferences['preferred_activities'])}")
        
        return "; ".join(context_parts)
    
    def get_emergency_recommendations(self) -> List[str]:
        """
        Get immediate emergency stress relief recommendations
        """
        return [
            "Stop and take 5 deep breaths",
            "Use the 5-4-3-2-1 grounding technique",
            "Step away from the current situation",
            "Call a trusted friend or family member",
            "Use progressive muscle relaxation",
            "Listen to calming music",
            "Go for a short walk",
            "Practice box breathing (4-4-4-4 pattern)"
        ]
    
    def get_weekly_plan(self, stress_level: int) -> Dict[str, List[str]]:
        """
        Generate a weekly stress management plan
        """
        if stress_level == 0:
            return {
                'monday': ["Morning meditation", "Regular exercise", "Healthy meal planning"],
                'tuesday': ["Work-life balance check", "Social activity", "Hobby time"],
                'wednesday': ["Mid-week stress check", "Nature walk", "Gratitude journaling"],
                'thursday': ["Professional development", "Social connection", "Relaxation time"],
                'friday': ["Week reflection", "Fun activity", "Weekend planning"],
                'saturday': ["Family time", "Outdoor activity", "Creative pursuit"],
                'sunday': ["Rest and recovery", "Next week planning", "Self-care routine"]
            }
        elif stress_level == 1:
            return {
                'monday': ["Extended meditation", "Gentle exercise", "Stress journaling"],
                'tuesday': ["Breathing exercises", "Support group", "Relaxation techniques"],
                'wednesday': ["Stress assessment", "Nature therapy", "Mindfulness practice"],
                'thursday': ["Professional support", "Social support", "Calming activities"],
                'friday': ["Stress relief techniques", "Fun but calm activity", "Weekend rest planning"],
                'saturday': ["Family support time", "Gentle outdoor activity", "Art therapy"],
                'sunday': ["Deep rest", "Stress management planning", "Self-compassion practice"]
            }
        else:  # High stress
            return {
                'monday': ["Emergency stress relief", "Professional help", "Rest and recovery"],
                'tuesday': ["Crisis support", "Gentle movement", "Grounding techniques"],
                'wednesday': ["Professional counseling", "Support system", "Minimal stress activities"],
                'thursday': ["Continued support", "Calm environment", "Stress reduction"],
                'friday': ["Gentle recovery", "Support network", "Rest and relaxation"],
                'saturday': ["Family support", "Very gentle activity", "Recovery focus"],
                'sunday': ["Complete rest", "Support planning", "Recovery assessment"]
            }

# Example usage
if __name__ == "__main__":
    # Example stress data
    stress_data = {
        'stress_level': 1,
        'stress_label': 'Medium',
        'confidence': 0.85,
        'probabilities': {'low': 0.15, 'medium': 0.85, 'high': 0.0}
    }
    
    # Example user preferences
    user_preferences = {
        'interests': ['reading', 'music', 'nature'],
        'available_time': '1-2 hours daily',
        'preferred_activities': ['walking', 'meditation', 'reading']
    }
    
    # Initialize recommendation engine (requires API key)
    try:
        engine = RecommendationEngine()
        recommendations = engine.generate_recommendations(stress_data, user_preferences)
        
        print("Personalized Recommendations:")
        for category, items in recommendations.items():
            print(f"\n{category.upper()}:")
            for item in items:
                print(f"  - {item}")
                
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your OpenAI API key in the environment variables.")
