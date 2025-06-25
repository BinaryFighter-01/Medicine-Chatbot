import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template_string, jsonify
from flask_cors import CORS
import requests
import re
import json
from datetime import datetime
import nltk
from textblob import TextBlob
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class AdvancedMedicineChatbot:
    def __init__(self):
        self.load_dataset()
        self.setup_vectorizer()
        self.setup_medical_keywords()
        self.conversation_history = []
        
    def load_dataset(self):
        """Load and preprocess the medicine dataset"""
        try:
            self.data = pd.read_csv('Medicine_Details_Final.csv')
            logger.info(f"Loaded {len(self.data)} medicines from dataset")
            
            # Clean and preprocess data
            self.data = self.data.fillna('Not available')
            
            # Create enhanced combined text for better matching
            self.data['combined'] = (
                self.data['Medicine Name'] + ' ' + 
                self.data['Uses'] + ' ' + 
                self.data['Composition'] + ' ' +
                self.data['Side_effects'] + ' ' +
                self.data['Manufacturer']
            )
            
            # Create searchable keywords
            self.data['keywords'] = self.data['combined'].str.lower()
            
        except FileNotFoundError:
            logger.error("Medicine_Details_Final.csv not found!")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def setup_vectorizer(self):
        """Initialize and fit the TF-IDF vectorizer"""
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['combined'])
        logger.info("TF-IDF vectorizer initialized")
    
    def setup_medical_keywords(self):
        """Setup medical terminology and common queries"""
        self.medical_keywords = {
            'symptoms': ['headache', 'fever', 'pain', 'cough', 'cold', 'flu', 'nausea', 'vomiting', 'diarrhea', 'constipation'],
            'conditions': ['diabetes', 'hypertension', 'asthma', 'arthritis', 'depression', 'anxiety', 'infection'],
            'body_parts': ['heart', 'liver', 'kidney', 'stomach', 'brain', 'lung', 'skin', 'eye', 'ear'],
            'drug_types': ['antibiotic', 'painkiller', 'antacid', 'vitamin', 'supplement', 'tablet', 'capsule', 'syrup']
        }
        
        self.query_patterns = {
            'side_effects': ['side effect', 'adverse', 'reaction', 'harmful', 'danger'],
            'dosage': ['dose', 'dosage', 'how much', 'quantity', 'amount'],
            'usage': ['use', 'used for', 'treat', 'cure', 'help'],
            'interaction': ['interact', 'combination', 'together', 'mix'],
            'storage': ['store', 'storage', 'keep', 'preserve'],
            'composition': ['ingredient', 'composition', 'contain', 'made of']
        }
    
    def preprocess_query(self, query):
        """Enhanced query preprocessing"""
        query = query.lower().strip()
        
        # Remove common question words
        query = re.sub(r'\b(what|how|when|where|why|can|could|should|would|is|are|do|does)\b', '', query)
        
        # Handle common misspellings and variations
        corrections = {
            'paracetamol': 'acetaminophen',
            'asprin': 'aspirin',
            'ibuprofin': 'ibuprofen'
        }
        
        for wrong, correct in corrections.items():
            query = query.replace(wrong, correct)
        
        return query.strip()
    
    def analyze_query_intent(self, query):
        """Determine the intent of the user's query"""
        query_lower = query.lower()
        
        for intent, keywords in self.query_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        
        return 'general'
    
    def get_medicine_matches(self, query, top_k=3):
        """Get top medicine matches using TF-IDF similarity"""
        try:
            processed_query = self.preprocess_query(query)
            user_vector = self.vectorizer.transform([processed_query])
            similarities = cosine_similarity(user_vector, self.tfidf_matrix)
            
            # Get top matches
            top_indices = similarities[0].argsort()[-top_k:][::-1]
            matches = []
            
            for idx in top_indices:
                if similarities[0][idx] > 0.1:  # Minimum similarity threshold
                    matches.append({
                        'index': idx,
                        'similarity': similarities[0][idx],
                        'medicine': self.data.iloc[idx]
                    })
            
            return matches
        except Exception as e:
            logger.error(f"Error in matching: {e}")
            return []
    
    def get_openfda_details(self, medicine_name):
        """Fetch additional details from OpenFDA API"""
        try:
            # Clean medicine name for API query
            clean_name = medicine_name.split()[0].lower()
            url = f"https://api.fda.gov/drug/label.json?search={clean_name}&limit=1"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                drug_data = response.json()
                if 'results' in drug_data and drug_data['results']:
                    result = drug_data['results'][0]
                    
                    fda_info = {}
                    fda_info['indications'] = result.get('indications_and_usage', ['Not available'])[0] if result.get('indications_and_usage') else 'Not available'
                    fda_info['warnings'] = result.get('warnings', ['Not available'])[0] if result.get('warnings') else 'Not available'
                    fda_info['dosage'] = result.get('dosage_and_administration', ['Not available'])[0] if result.get('dosage_and_administration') else 'Not available'
                    fda_info['contraindications'] = result.get('contraindications', ['Not available'])[0] if result.get('contraindications') else 'Not available'
                    
                    return fda_info
            
            return None
        except Exception as e:
            logger.error(f"FDA API error: {e}")
            return None
    
    def format_medicine_info(self, medicine_data, intent='general', fda_data=None):
        """Format medicine information based on query intent"""
        medicine = medicine_data['medicine']
        
        response = f"## 💊 {medicine['Medicine Name']}\n\n"
        
        if intent == 'composition' or intent == 'general':
            response += f"**🧪 Composition:** {medicine['Composition']}\n\n"
        
        if intent == 'usage' or intent == 'general':
            response += f"**🎯 Uses:** {medicine['Uses']}\n\n"
        
        if intent == 'side_effects' or intent == 'general':
            response += f"**⚠️ Side Effects:** {medicine['Side_effects']}\n\n"
        
        if intent == 'storage' or intent == 'general':
            response += f"**📦 Storage:** {medicine['Storage Condition']} at {medicine['Storage Temperature (°C)']}°C, {medicine['Storage Humidity (%)']}% humidity\n\n"
        
        response += f"**🏭 Manufacturer:** {medicine['Manufacturer']}\n\n"
        
        # Add FDA information if available
        if fda_data:
            response += "### 🏛️ FDA Information:\n"
            if fda_data['indications'] != 'Not available':
                response += f"**Indications:** {fda_data['indications'][:200]}...\n\n"
            if fda_data['warnings'] != 'Not available':
                response += f"**⚠️ FDA Warnings:** {fda_data['warnings'][:200]}...\n\n"
        
        # Add confidence score
        confidence = int(medicine_data['similarity'] * 100)
        response += f"*Confidence: {confidence}%*"
        
        return response
    
    def get_general_medical_advice(self, query):
        """Provide general medical advice for common queries"""
        advice_db = {
            'headache': "For headaches, you can try:\n• Rest in a quiet, dark room\n• Apply cold or warm compress\n• Stay hydrated\n• Consider over-the-counter pain relievers like acetaminophen or ibuprofen\n• If severe or persistent, consult a doctor",
            'fever': "For fever management:\n• Stay hydrated with plenty of fluids\n• Rest and avoid strenuous activities\n• Use fever reducers like acetaminophen or ibuprofen\n• Dress lightly and keep room cool\n• Seek medical attention if fever exceeds 103°F (39.4°C)",
            'cold': "For common cold:\n• Get plenty of rest\n• Drink warm liquids\n• Use saline nasal drops\n• Consider throat lozenges\n• Humidify the air\n• Most colds resolve in 7-10 days",
            'cough': "For cough relief:\n• Stay hydrated\n• Use honey (for adults)\n• Try warm salt water gargle\n• Use humidifier\n• Avoid irritants like smoke\n• See doctor if cough persists over 2 weeks"
        }
        
        query_lower = query.lower()
        for condition, advice in advice_db.items():
            if condition in query_lower:
                return f"## General Advice for {condition.title()}\n\n{advice}\n\n**⚠️ Disclaimer:** This is general information only. Always consult healthcare professionals for proper diagnosis and treatment."
        
        return None
    
    def generate_response(self, user_input):
        """Generate comprehensive response to user query"""
        try:
            # Store conversation
            self.conversation_history.append({
                'timestamp': datetime.now(),
                'user_input': user_input,
                'type': 'user'
            })
            
            # Analyze query intent
            intent = self.analyze_query_intent(user_input)
            
            # Check for general medical advice first
            general_advice = self.get_general_medical_advice(user_input)
            if general_advice:
                response = general_advice
            else:
                # Get medicine matches
                matches = self.get_medicine_matches(user_input, top_k=2)
                
                if not matches:
                    response = self.get_no_match_response(user_input)
                else:
                    # Get the best match
                    best_match = matches[0]
                    
                    # Fetch FDA data for the best match
                    fda_data = self.get_openfda_details(best_match['medicine']['Medicine Name'])
                    
                    # Format response
                    response = self.format_medicine_info(best_match, intent, fda_data)
                    
                    # Add alternative suggestions if available
                    if len(matches) > 1:
                        response += f"\n\n### 🔍 You might also be interested in:\n"
                        for match in matches[1:]:
                            response += f"• **{match['medicine']['Medicine Name']}** - {match['medicine']['Uses'][:100]}...\n"
            
            # Store bot response
            self.conversation_history.append({
                'timestamp': datetime.now(),
                'response': response,
                'type': 'bot'
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try rephrasing your question or contact support if the issue persists."
    
    def get_no_match_response(self, query):
        """Generate helpful response when no matches are found"""
        suggestions = [
            "Try using the generic name of the medicine",
            "Check the spelling of the medicine name",
            "Describe your symptoms instead of the medicine name",
            "Ask about the condition you want to treat"
        ]
        
        response = "## 🤔 I couldn't find a specific match for your query.\n\n"
        response += "**Here are some suggestions:**\n"
        for suggestion in suggestions:
            response += f"• {suggestion}\n"
        
        response += "\n**You can ask me about:**\n"
        response += "• Specific medicine names (e.g., 'Tell me about Aspirin')\n"
        response += "• Symptoms (e.g., 'Medicine for headache')\n"
        response += "• Conditions (e.g., 'Treatment for diabetes')\n"
        response += "• Side effects (e.g., 'Side effects of ibuprofen')\n"
        
        return response

# Initialize the chatbot
chatbot = AdvancedMedicineChatbot()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Enhanced HTML template with modern UI
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Medicine Chatbot</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .chat-container {
            width: 100%;
            max-width: 900px;
            height: 700px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 20px;
            text-align: center;
            position: relative;
        }
        
        .chat-header h1 {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .chat-header p {
            font-size: 14px;
            opacity: 0.9;
        }
        
        .status-indicator {
            position: absolute;
            top: 20px;
            right: 20px;
            width: 12px;
            height: 12px;
            background: #2ecc71;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .chat-box {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }
        
        .chat-box::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-box::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        
        .chat-box::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }
        
        .message {
            margin: 15px 0;
            padding: 15px 20px;
            border-radius: 18px;
            max-width: 85%;
            word-wrap: break-word;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .user-message {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 5px;
        }
        
        .bot-message {
            background: white;
            color: #333;
            margin-right: auto;
            border: 1px solid #e1e8ed;
            border-bottom-left-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        
        .bot-message h2 {
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 18px;
        }
        
        .bot-message h3 {
            color: #3498db;
            margin: 15px 0 8px 0;
            font-size: 16px;
        }
        
        .bot-message strong {
            color: #2c3e50;
        }
        
        .welcome-message {
            text-align: center;
            background: linear-gradient(135deg, #3498db, #2ecc71);
            color: white;
            margin: 0 auto 20px auto;
            font-size: 16px;
        }
        
        .typing-indicator {
            display: none;
            background: white;
            border: 1px solid #e1e8ed;
            margin-right: auto;
            padding: 15px 20px;
            border-radius: 18px;
            border-bottom-left-radius: 5px;
        }
        
        .typing-dots {
            display: flex;
            gap: 4px;
        }
        
        .typing-dots span {
            width: 8px;
            height: 8px;
            background: #3498db;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }
        
        .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
        
        .chat-input-area {
            padding: 20px;
            background: white;
            border-top: 1px solid #e1e8ed;
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        .user-input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e1e8ed;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: all 0.3s ease;
        }
        
        .user-input:focus {
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }
        
        .send-btn {
            padding: 15px 25px;
            background: linear-gradient(135deg, #3498db, #2ecc71);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .send-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
        }
        
        .send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .quick-suggestions {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        
        .suggestion-btn {
            padding: 8px 15px;
            background: rgba(52, 152, 219, 0.1);
            color: #3498db;
            border: 1px solid #3498db;
            border-radius: 20px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .suggestion-btn:hover {
            background: #3498db;
            color: white;
        }
        
        @media (max-width: 768px) {
            .chat-container {
                height: 100vh;
                border-radius: 0;
                margin: 0;
            }
            
            .message {
                max-width: 95%;
            }
            
            .chat-input-area {
                flex-direction: column;
                gap: 10px;
            }
            
            .user-input {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <div class="status-indicator"></div>
            <h1><i class="fas fa-pills"></i> Advanced Medicine Chatbot</h1>
            <p>Your intelligent healthcare companion</p>
        </div>
        
        <div class="chat-box" id="chatBox">
            <div class="message welcome-message">
                <i class="fas fa-robot"></i> Hello! I'm your advanced medicine assistant. 
                Ask me about medicines, symptoms, side effects, or general health advice.
            </div>
            
            <div class="quick-suggestions">
                <button class="suggestion-btn" onclick="sendSuggestion('What is aspirin used for?')">
                    <i class="fas fa-pills"></i> Aspirin uses
                </button>
                <button class="suggestion-btn" onclick="sendSuggestion('Medicine for headache')">
                    <i class="fas fa-head-side-virus"></i> Headache relief
                </button>
                <button class="suggestion-btn" onclick="sendSuggestion('Side effects of ibuprofen')">
                    <i class="fas fa-exclamation-triangle"></i> Side effects
                </button>
                <button class="suggestion-btn" onclick="sendSuggestion('How to store medicines?')">
                    <i class="fas fa-box"></i> Storage tips
                </button>
            </div>
            
            <div class="typing-indicator" id="typingIndicator">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
        
        <div class="chat-input-area">
            <input type="text" class="user-input" id="userInput" 
                   placeholder="Ask about medicines, symptoms, or health advice..." 
                   onkeypress="if(event.key === 'Enter') sendMessage();">
            <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                <i class="fas fa-paper-plane"></i> Send
            </button>
        </div>
    </div>

    <script>
        let isProcessing = false;
        
        function sendMessage() {
            if (isProcessing) return;
            
            let input = document.getElementById('userInput').value.trim();
            if (input === '') return;
            
            addUserMessage(input);
            showTypingIndicator();
            
            isProcessing = true;
            document.getElementById('sendBtn').disabled = true;
            document.getElementById('userInput').value = '';
            
            fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: input })
            })
            .then(response => {
                if (!response.ok) throw new Error('Network response not ok');
                return response.json();
            })
            .then(data => {
                hideTypingIndicator();
                addBotMessage(data.response);
            })
            .catch(error => {
                hideTypingIndicator();
                addBotMessage('Sorry, I encountered an error. Please try again.');
                console.error('Error:', error);
            })
            .finally(() => {
                isProcessing = false;
                document.getElementById('sendBtn').disabled = false;
                document.getElementById('userInput').focus();
            });
        }
        
        function sendSuggestion(text) {
            document.getElementById('userInput').value = text;
            sendMessage();
        }
        
        function addUserMessage(message) {
            const chatBox = document.getElementById('chatBox');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message user-message';
            messageDiv.innerHTML = `<i class="fas fa-user"></i> ${message}`;
            chatBox.appendChild(messageDiv);
            scrollToBottom();
        }
        
        function addBotMessage(message) {
            const chatBox = document.getElementById('chatBox');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message bot-message';
            
            // Convert markdown-like formatting to HTML
            let formattedMessage = message
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/## (.*?)$/gm, '<h2>$1</h2>')
                .replace(/### (.*?)$/gm, '<h3>$1</h3>')
                .replace(/\n/g, '<br>');
            
            messageDiv.innerHTML = `<i class="fas fa-robot"></i> ${formattedMessage}`;
            chatBox.appendChild(messageDiv);
            scrollToBottom();
        }
        
        function showTypingIndicator() {
            document.getElementById('typingIndicator').style.display = 'block';
            scrollToBottom();
        }
        
        function hideTypingIndicator() {
            document.getElementById('typingIndicator').style.display = 'none';
        }
        
        function scrollToBottom() {
            const chatBox = document.getElementById('chatBox');
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        // Focus on input when page loads
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('userInput').focus();
        });
        
        // Handle Enter key
        document.getElementById('userInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !isProcessing) {
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_template)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message', '')
        if not user_input:
            return jsonify({'response': 'Please enter a valid question.'})
        
        response = chatbot.generate_response(user_input)
        return jsonify({'response': response})
    
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return jsonify({'response': 'I apologize, but I encountered an error. Please try again.'}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'medicines_loaded': len(chatbot.data),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info("Starting Advanced Medicine Chatbot...")
    app.run(debug=True, host='0.0.0.0', port=5000)