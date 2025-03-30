import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template_string
import requests

# Load the dataset with error handling
try:
    data = pd.read_csv('Medicine_Details_Final.csv')
except FileNotFoundError:
    print("Error: Medicine_Details_Final.csv not found!")
    exit(1)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Combine relevant columns for similarity matching
data['combined'] = data['Medicine Name'] + ' ' + data['Uses'] + ' ' + data['Composition']

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['combined'])

# Function to fetch details from OpenFDA API
def get_openfda_details(medicine_name):
    url = f"https://api.fda.gov/drug/label.json?search={medicine_name}&limit=1"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            drug_data = response.json()
            if 'results' in drug_data and drug_data['results']:
                result = drug_data['results'][0]
                indications = result.get('indications_and_usage', ['Not available'])[0]
                adverse_effects = result.get('adverse_reactions', ['Not available'])[0]
                return f"**FDA Indications:** {indications}\n**FDA Adverse Reactions:** {adverse_effects}"
            return "No detailed FDA data found for this medicine."
        return "API request failed. Using local data only."
    except requests.RequestException:
        return "Error connecting to FDA API. Using local data only."

# Function to get medicine info based on index
def get_medicine_info(index):
    medicine = data.iloc[index]
    return (
        f"**Medicine:** {medicine['Medicine Name']}\n"
        f"**Composition:** {medicine['Composition']}\n"
        f"**Uses:** {medicine['Uses']}\n"
        f"**Side Effects:** {medicine['Side_effects']}\n"
        f"**Manufacturer:** {medicine['Manufacturer']}\n"
        f"**Storage:** {medicine['Storage Condition']} at {medicine['Storage Temperature (°C)']}°C, "
        f"{medicine['Storage Humidity (%)']}% humidity"
    )

# Function to get response based on user input (TF-IDF only)
def get_response(user_input):
    try:
        user_input = user_input.lower()
        user_vector = vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, tfidf_matrix)
        best_match_index = similarities.argmax()
        similarity_score = similarities[0, best_match_index]

        if similarity_score > 0.2:  # Threshold for a decent match
            api_response = get_openfda_details(data.iloc[best_match_index]['Medicine Name'].split()[0])
            return f"{get_medicine_info(best_match_index)}\n\n{api_response}"
        return "Sorry, I couldn’t find a match. Try asking about a specific medicine or condition!"
    except Exception as e:
        print(f"Error in get_response: {e}")
        return "Sorry, something went wrong. Please try again!"

# Initialize Flask app
app = Flask(__name__)

# Fixed HTML template
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medicine Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
            display: flex;
            justify-content: center;
        }
        .chat-container {
            width: 800px;
            height: 600px;
            background-color: #ffffff;
            border: 1px solid #ccc;
            border-radius: 5px;
            display: flex;
            flex-direction: column;
        }
        .chat-header {
            background-color: #333;
            color: #fff;
            padding: 10px;
            text-align: center;
            font-size: 20px;
            border-bottom: 1px solid #999;
        }
        .chat-box {
            flex-grow: 1;
            padding: 15px;
            overflow-y: auto;
            background-color: #fff;
        }
        .chat-input-area {
            padding: 15px;
            border-top: 1px solid #ccc;
            display: flex;
            background-color: #f9f9f9;
        }
        .user-input {
            flex-grow: 1; /* Fixed: Allows button to stay visible */
            padding: 10px;
            border: 1px solid #999;
            border-radius: 5px;
            font-size: 16px;
            margin-right: 10px;
        }
        .send-btn {
            padding: 10px 20px;
            background-color: #333;
            color: #fff;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
        }
        .send-btn:hover {
            background-color: #555;
        }
        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            max-width: 90%;
        }
        .user-message {
            background-color: #d3e0ea;
            margin-left: auto;
            text-align: right;
        }
        .bot-message {
            background-color: #e6e6e6;
            margin-right: auto;
            white-space: pre-wrap;
        }
        .welcome-message {
            text-align: center;
            color: #555;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">Medicine Chatbot</div>
        <div class="chat-box" id="chatBox">
            <div class="message bot-message welcome-message">
                Hello! Ask me about any medicine or condition.
            </div>
        </div>
        <div class="chat-input-area">
            <input type="text" class="user-input" id="userInput" placeholder="Type your question..." onkeypress="if(event.key === 'Enter') sendMessage();">
            <button class="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        console.log("Script loaded"); // Debug: Confirm script runs
        function sendMessage() {
            console.log("Send button clicked"); // Debug: Confirm click
            let input = document.getElementById('userInput').value;
            console.log("Input: " + input); // Debug: Log input

            if (input.trim() === '') {
                console.log("Empty input, aborted");
                return;
            }

            let chatBox = document.getElementById('chatBox');
            chatBox.innerHTML += '<div class="message user-message">You: ' + input + '</div>';
            console.log("User message added"); // Debug

            fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: input })
            })
            .then(response => {
                console.log("Fetch status: " + response.status); // Debug
                if (!response.ok) throw new Error('Network response not ok');
                return response.json();
            })
            .then(data => {
                console.log("Bot response: " + data.response); // Debug
                chatBox.innerHTML += '<div class="message bot-message">Bot: ' + data.response + '</div>';
                chatBox.scrollTop = chatBox.scrollHeight;
            })
            .catch(error => {
                console.error('Fetch error:', error); // Debug
                chatBox.innerHTML += '<div class="message bot-message">Error: ' + error.message + '</div>';
            });

            document.getElementById('userInput').value = '';
            console.log("Input cleared"); // Debug
        }
    </script>
</body>
</html>
"""

# Flask routes
@app.route('/')
def home():
    return render_template_string(html_template)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    print(f"Received input: {user_input}")  # Debug
    response = get_response(user_input)
    print(f"Response: {response}")  # Debug
    return {'response': response}

# Run the app
if __name__ == '__main__':
    app.run(debug=True)