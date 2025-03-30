# Medicine Chatbot

A Flask-based web chatbot that provides detailed information about medicines using a local dataset and the OpenFDA API. The chatbot uses TF-IDF similarity matching to interpret user queries and responds with structured information about medicine uses, side effects, storage, and more.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Medicine Chatbot is a web application designed to assist users in retrieving information about medicines. Users can ask questions like "What is Augmentin used for?" or "Side effects of Azithral?" via a simple chat interface. The bot processes these queries using TF-IDF vectorization to match them against a local CSV dataset (`Medicine_Details_Final.csv`) and fetches additional data from the OpenFDA API when available. Responses are structured with bold headings and bullet points for clarity, mimicking the style of advanced AI assistants like ChatGPT or Grok.

This project is ideal for learning about web development with Flask, natural language processing with TF-IDF, and API integration.

## Features

- **Interactive Chat Interface**: Simple, classic UI with a chat box, input field, and "Send" button.
- **Medicine Information**: Provides details from a local dataset, including:
  - Medicine name
  - Composition
  - Uses
  - Side effects
  - Manufacturer
  - Storage conditions
- **OpenFDA Integration**: Fetches real-time FDA data (indications and adverse reactions) for matched medicines.
- **TF-IDF Matching**: Uses term frequency-inverse document frequency to find the best medicine match based on user input.
- **Structured Responses**: Formats answers with bold headings and bullet points for readability.
- **Error Handling**: Gracefully handles missing data, API failures, and invalid queries.

## Technologies

- **Python 3.13**: Core programming language.
- **Flask**: Lightweight web framework for serving the app.
- **Pandas**: Data manipulation for the CSV dataset.
- **Scikit-learn**: TF-IDF vectorization and cosine similarity.
- **Requests**: HTTP requests to the OpenFDA API.
- **HTML/CSS/JavaScript**: Frontend for the chat interface.

## Installation

### Prerequisites
- Python 3.13 installed ([Download](https://www.python.org/downloads/)).
- Git installed ([Download](https://git-scm.com/downloads)).
- A text editor (e.g., VS Code).

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/medicine-chatbot.git
   cd medicine-chatbot
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install pandas scikit-learn flask requests
   ```

4. **Prepare the Dataset**:
   - Ensure `Medicine_Details_Final.csv` is in the project root.
   - The CSV must have these columns:
     - `Medicine Name`
     - `Composition`
     - `Uses`
     - `Side_effects`
     - `Manufacturer`
     - `Storage Condition`
     - `Storage Temperature (°C)`
     - `Storage Humidity (%)`
   - Example CSV is available in the repo as `Medicine_Details_Final.csv`.

5. **Run the Application**:
   ```bash
   python chatbot.py
   ```

6. **Access the Chatbot**:
   - Open a browser and go to `http://127.0.0.1:5000/`.

## Usage

1. **Start the Server**:
   - Run `python chatbot.py` in your terminal.
   - You’ll see output like:
     ```
     * Serving Flask app 'chatbot'
     * Debug mode: on
     * Running on http://127.0.0.1:5000
     ```

2. **Interact with the Chatbot**:
   - Open `http://127.0.0.1:5000/` in your browser.
   - Type a question (e.g., "What is Augmentin used for?") in the input box.
   - Click "Send" or press Enter.
   - The bot responds with structured info (e.g., uses, FDA data).

## File Structure

```
medicine-chatbot/
├── chatbot.py                # Main application script
├── Medicine_Details_Final.csv # Medicine dataset
├── README.md                 # This file
└── requirements.txt          # List of Python dependencies
```

## How It Works

### Backend
- **CSV Loading**: `pandas.read_csv()` loads `Medicine_Details_Final.csv` into a DataFrame.
- **TF-IDF Matching**: `TfidfVectorizer` creates a matrix of term frequencies from the dataset.
- **API Integration**: Queries OpenFDA API with medicine names.
- **Response Generation**: Combines local data and API results into structured responses.

### Frontend
- **HTML/CSS**: Chat UI with a chat box, input field, and button.
- **JavaScript**: Handles sending user input and displaying bot responses.

## Troubleshooting

- **Server Not Starting**:
  - Run `pip install -r requirements.txt`.
  - Change `app.run(port=5001)` if port 5000 is in use.

- **No Response**:
  - Check API connectivity.
  - Ensure dataset is correctly formatted.

## Contributing

1. Fork the repository.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

