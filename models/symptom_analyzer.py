"""
AI Doctor Symptom Analyzer - Analyzes symptoms and provides recommendations
"""
from typing import Dict, List, Optional, Tuple
import re

class SymptomAnalyzer:
    """Analyzes symptoms and provides medical recommendations"""
    
    def __init__(self):
        self.symptom_patterns = {
            "fever": {
                "keywords": ["fever", "temperature", "hot", "burning", "chills", "sweating"],
                "follow_ups": [
                    "What did you eat in the last 24 hours?",
                    "Do you have a headache or body aches?",
                    "How long have you had this fever?",
                    "Do you have a cough or sore throat?",
                    "Have you been exposed to anyone who was sick?"
                ],
                "causes": [
                    "Viral infection (common cold, flu)",
                    "Bacterial infection",
                    "Food poisoning from contaminated food",
                    "Dehydration",
                    "Overexertion or heat exhaustion"
                ],
                "dos": [
                    "Rest and stay hydrated - drink plenty of water, electrolyte solutions",
                    "Take paracetamol/acetaminophen as directed (if no allergies)",
                    "Use a cool compress on forehead and body",
                    "Wear light, breathable clothing",
                    "Monitor temperature regularly",
                    "Get adequate sleep to help your body fight infection"
                ],
                "donts": [
                    "Don't bundle up in heavy blankets - this can raise temperature further",
                    "Avoid alcohol and caffeinated drinks - they cause dehydration",
                    "Don't take aspirin if you're under 18 (risk of Reye's syndrome)",
                    "Avoid strenuous activities until fever subsides",
                    "Don't ignore persistent high fever (>103°F/39.4°C) - seek medical help"
                ],
                "specialty": "General Medicine, Internal Medicine"
            },
            "stomach": {
                "keywords": ["stomach", "stomachache", "abdominal", "belly", "nausea", "vomiting", "diarrhea", "indigestion", "bloating", "cramps"],
                "follow_ups": [
                    "What did you eat recently?",
                    "When did the pain start?",
                    "Is the pain sharp, dull, or cramping?",
                    "Do you have nausea or vomiting?",
                    "Have you noticed any changes in bowel movements?",
                    "Are you experiencing bloating or gas?"
                ],
                "causes": [
                    "Food poisoning or food intolerance",
                    "Gastroenteritis (stomach flu)",
                    "Acid reflux or GERD",
                    "Irritable Bowel Syndrome (IBS)",
                    "Overeating or eating too quickly",
                    "Stress or anxiety affecting digestion",
                    "Lactose intolerance or other food allergies"
                ],
                "dos": [
                    "Stay hydrated - sip water, clear broths, or electrolyte solutions",
                    "Eat bland foods like rice, bananas, toast (BRAT diet)",
                    "Rest and avoid lying down immediately after eating",
                    "Apply a warm compress to your abdomen",
                    "Eat smaller, more frequent meals",
                    "Keep a food diary to identify triggers"
                ],
                "donts": [
                    "Avoid spicy, fatty, or fried foods",
                    "Don't consume dairy products if lactose intolerant",
                    "Avoid alcohol, caffeine, and carbonated drinks",
                    "Don't take NSAIDs (ibuprofen, aspirin) on empty stomach",
                    "Avoid lying down right after eating",
                    "Don't ignore severe or persistent pain - seek medical attention"
                ],
                "specialty": "Gastroenterology, General Medicine"
            },
            "headache": {
                "keywords": ["headache", "head", "migraine", "tension", "pain"],
                "follow_ups": [
                    "Where is the pain located? (forehead, temples, back of head)",
                    "How long have you had this headache?",
                    "Do you have any vision changes or sensitivity to light?",
                    "Have you been under stress recently?",
                    "Are you drinking enough water?"
                ],
                "causes": [
                    "Tension headache from stress or poor posture",
                    "Dehydration",
                    "Migraine triggered by food, hormones, or stress",
                    "Sinus congestion or infection",
                    "Eye strain from screens",
                    "Lack of sleep"
                ],
                "dos": [
                    "Rest in a dark, quiet room",
                    "Stay hydrated - drink water regularly",
                    "Apply a cold or warm compress to forehead or neck",
                    "Practice relaxation techniques or gentle neck stretches",
                    "Take over-the-counter pain relievers if appropriate (paracetamol, ibuprofen)",
                    "Get adequate sleep"
                ],
                "donts": [
                    "Don't skip meals - low blood sugar can trigger headaches",
                    "Avoid excessive caffeine or sudden caffeine withdrawal",
                    "Don't stare at screens for prolonged periods",
                    "Avoid loud noises and bright lights",
                    "Don't ignore sudden severe headaches or headaches with vision changes - seek immediate medical help"
                ],
                "specialty": "Neurology, General Medicine"
            },
            "cough": {
                "keywords": ["cough", "coughing", "throat", "chest", "phlegm", "mucus"],
                "follow_ups": [
                    "Is it a dry cough or do you have phlegm?",
                    "How long have you been coughing?",
                    "Do you have a fever or body aches?",
                    "Are you experiencing shortness of breath?",
                    "Do you smoke or have you been exposed to smoke?"
                ],
                "causes": [
                    "Common cold or upper respiratory infection",
                    "Allergies or post-nasal drip",
                    "Bronchitis",
                    "Asthma",
                    "Acid reflux (GERD)",
                    "Smoking or exposure to irritants"
                ],
                "dos": [
                    "Stay hydrated - warm liquids like tea with honey can soothe throat",
                    "Use a humidifier or steam inhalation",
                    "Gargle with warm salt water",
                    "Rest your voice",
                    "Elevate your head while sleeping",
                    "Avoid irritants like smoke and strong perfumes"
                ],
                "donts": [
                    "Don't smoke or expose yourself to secondhand smoke",
                    "Avoid dairy products if they increase mucus production",
                    "Don't suppress productive coughs completely - they help clear airways",
                    "Avoid cold drinks and ice cream",
                    "Don't ignore persistent cough lasting more than 3 weeks - see a doctor"
                ],
                "specialty": "Pulmonology, General Medicine"
            },
            "cold": {
                "keywords": ["cold", "runny nose", "sneezing", "congestion", "nasal", "stuffy"],
                "follow_ups": [
                    "Do you have a fever?",
                    "Are you experiencing body aches?",
                    "How long have you had these symptoms?",
                    "Do you have a sore throat or cough?",
                    "Have you been around anyone with similar symptoms?"
                ],
                "causes": [
                    "Viral infection (rhinovirus, coronavirus)",
                    "Seasonal allergies",
                    "Weather changes",
                    "Weakened immune system",
                    "Exposure to infected individuals"
                ],
                "dos": [
                    "Rest and get plenty of sleep",
                    "Stay hydrated - drink warm fluids",
                    "Use saline nasal sprays or rinses",
                    "Gargle with warm salt water for sore throat",
                    "Wash hands frequently to prevent spread",
                    "Use a humidifier to ease congestion"
                ],
                "donts": [
                    "Don't share utensils or personal items",
                    "Avoid touching your face frequently",
                    "Don't take antibiotics unless prescribed (colds are viral)",
                    "Avoid close contact with others to prevent spreading",
                    "Don't ignore symptoms that worsen or persist beyond 10 days"
                ],
                "specialty": "General Medicine, ENT"
            }
        }
    
    def detect_symptom(self, user_input: str) -> Optional[str]:
        """Detect the main symptom from user input"""
        user_lower = user_input.lower()
        for symptom, data in self.symptom_patterns.items():
            for keyword in data["keywords"]:
                if keyword in user_lower:
                    return symptom
        return None
    
    def get_follow_up_question(self, symptom: str, conversation_history: List[Dict]) -> Optional[str]:
        """Get the next follow-up question based on conversation history"""
        if symptom not in self.symptom_patterns:
            return None
        
        asked_questions = [msg.get("question", "") for msg in conversation_history if msg.get("role") == "assistant"]
        follow_ups = self.symptom_patterns[symptom]["follow_ups"]
        
        for question in follow_ups:
            if question not in asked_questions:
                return question
        
        return None
    
    def analyze_symptoms(self, symptom: str, conversation_history: List[Dict]) -> Dict:
        """Analyze symptoms and provide comprehensive recommendations"""
        if symptom not in self.symptom_patterns:
            return {
                "symptom": symptom,
                "causes": ["Unable to determine specific cause. Please consult a healthcare professional."],
                "dos": ["Seek medical advice for proper diagnosis"],
                "donts": ["Don't self-diagnose serious conditions"],
                "specialty": "General Medicine"
            }
        
        data = self.symptom_patterns[symptom]
        
        # Extract answers from conversation history
        user_answers = {}
        for msg in conversation_history:
            if msg.get("role") == "user":
                user_answers[msg.get("question", "")] = msg.get("content", "")
        
        # Customize analysis based on answers
        analysis = {
            "symptom": symptom,
            "causes": data["causes"],
            "dos": data["dos"],
            "donts": data["donts"],
            "specialty": data["specialty"]
        }
        
        # Add specific insights based on food-related answers
        if "food" in str(user_answers).lower() or "eat" in str(user_answers).lower():
            analysis["causes"].insert(0, "Possible food-related issue - consider food diary and elimination diet")
        
        return analysis
    
    def chat(self, message: str, conversation_history: List[Dict] = None) -> Dict:
        """Main chat interface for symptom analysis"""
        if conversation_history is None:
            conversation_history = []
        
        # Detect symptom from current message
        symptom = self.detect_symptom(message)
        
        # If no symptom detected and this is the first message, ask what the problem is
        if not symptom and len(conversation_history) == 0:
            return {
                "role": "assistant",
                "content": "Hello! I'm your AI health assistant. What seems to be the problem? (e.g., fever, stomach pain, headache, cough, etc.)",
                "question": "What seems to be the problem?",
                "needs_followup": True,
                "analysis": None
            }
        
        # If symptom detected but no follow-ups asked yet
        if symptom and len([m for m in conversation_history if m.get("role") == "assistant"]) == 0:
            follow_up = self.get_follow_up_question(symptom, conversation_history)
            if follow_up:
                return {
                    "role": "assistant",
                    "content": f"I understand you're experiencing {symptom} symptoms. {follow_up}",
                    "question": follow_up,
                    "needs_followup": True,
                    "symptom": symptom,
                    "analysis": None
                }
        
        # Check if we have enough information for analysis
        assistant_messages = [m for m in conversation_history if m.get("role") == "assistant"]
        
        # If we have asked at least 2-3 questions, provide analysis
        if len(assistant_messages) >= 2:
            if symptom:
                analysis = self.analyze_symptoms(symptom, conversation_history)
                return {
                    "role": "assistant",
                    "content": f"Based on your symptoms, I've completed my analysis. Please see the recommendations below.",
                    "question": None,
                    "needs_followup": False,
                    "symptom": symptom,
                    "analysis": analysis
                }
            else:
                # Try to detect symptom from conversation
                for msg in conversation_history:
                    detected = self.detect_symptom(msg.get("content", ""))
                    if detected:
                        symptom = detected
                        break
                
                if symptom:
                    analysis = self.analyze_symptoms(symptom, conversation_history)
                    return {
                        "role": "assistant",
                        "content": f"Based on our conversation, I've analyzed your {symptom} symptoms. Please see the recommendations below.",
                        "question": None,
                        "needs_followup": False,
                        "symptom": symptom,
                        "analysis": analysis
                    }
        
        # Otherwise, ask next follow-up question
        if symptom:
            follow_up = self.get_follow_up_question(symptom, conversation_history)
            if follow_up:
                return {
                    "role": "assistant",
                    "content": follow_up,
                    "question": follow_up,
                    "needs_followup": True,
                    "symptom": symptom,
                    "analysis": None
                }
        
        # Fallback
        return {
            "role": "assistant",
            "content": "Could you provide more details about your symptoms? (e.g., fever, stomach pain, headache)",
            "question": "Could you provide more details?",
            "needs_followup": True,
            "analysis": None
        }

