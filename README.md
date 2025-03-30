Below is the `README.md` content formatted as a proper Markdown file for your GitHub repository. You can copy this directly into a `README.md` file in your project directory.

---

```markdown
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

3. **Example Queries**:
   - "What is Augmentin used for?"
     ```
     **Uses:** Treatment of Bacterial infections
     **FDA Indications:** AUGMENTIN is a combination penicillin-class...
     **FDA Adverse Reactions:** Diarrhea, nausea, vomiting...
     ```
   - "Side effects of Azithral?"
     ```
     **Side Effects:** Nausea Abdominal pain Diarrhea
     **FDA Indications:** Azithromycin is a macrolide antibacterial...
     **FDA Adverse Reactions:** Diarrhea, nausea, abdominal pain...
     ```
   - "How to store Avastin?"
     ```
     **Storage:** refrigerated at 5°C, 50% humidity
     **FDA Indications:** Not available
     **FDA Adverse Reactions:** Not available
     ```

## File Structure

```
medicine-chatbot/
├── chatbot.py                # Main application script
├── Medicine_Details_Final.csv # Medicine dataset
├── README.md                 # This file
└── requirements.txt          # List of Python dependencies
```

- **`chatbot.py`**: Contains Flask app, TF-IDF logic, API integration, and HTML/JavaScript for the UI.
- **`Medicine_Details_Final.csv`**: CSV file with medicine data (must be provided or created).
- **`requirements.txt`**: For easy dependency installation:
  ```
  pandas
  scikit-learn
  flask
  requests
  ```

## How It Works

### Backend
1. **CSV Loading**:
   - `pandas.read_csv()` loads `Medicine_Details_Final.csv` into a DataFrame.
   - Columns are combined into a single text field (`combined`) for TF-IDF.

2. **TF-IDF Matching**:
   - `TfidfVectorizer` creates a matrix of term frequencies from the dataset.
   - User input is transformed into a vector and compared to the matrix using `cosine_similarity`.
   - The highest similarity score identifies the best-matching medicine.

3. **API Integration**:
   - `get_openfda_details()` queries the OpenFDA API with the medicine name.
   - Returns FDA indications and adverse reactions or an error message.

4. **Response Generation**:
   - `get_response()` combines local data (`get_medicine_info()`) and API data into a structured string.
   - Uses a similarity threshold (0.2) to ensure relevance.

5. **Flask Server**:
   - `/` route serves the HTML page.
   - `/chat` route handles POST requests, processes input, and returns JSON.

### Frontend
1. **HTML/CSS**:
   - A chat container with a header, scrollable chat box, and input area.
   - Styled with a simple, classic look (gray bot messages, blue user messages).

2. **JavaScript**:
   - `sendMessage()`:
     - Captures user input.
     - Adds it to the chat box.
     - Sends a POST request to `/chat`.
     - Displays the bot’s response with formatting (line breaks, bold text).

## Dataset

The chatbot relies on `Medicine_Details_Final.csv`. Example format:

| Medicine Name          | Composition                    | Uses                        | Side_effects                  | Manufacturer                  | Storage Condition | Storage Temperature (°C) | Storage Humidity (%) |
|-----------------------|--------------------------------|-----------------------------|------------------------------|------------------------------|-------------------|--------------------------|---------------------|
| Augmentin 625 Duo     | Amoxycillin (500mg) + Clav... | Treatment of Bacterial inf...| Vomiting Nausea Diarrhea...  | Glaxo SmithKline Pharmace... | dry               | 20                       | 40                  |
| Azithral 500          | Azithromycin (500mg)          | Treatment of Bacterial inf...| Nausea Abdominal pain Dia... | Alembic Pharmaceuticals L... | cool              | 25                       | 50                  |

- **Source**: Create your own or use a sample from the repo.
- **Requirements**: Must match the column names exactly as above.

## Troubleshooting

- **"Send" Button Doesn’t Work**:
  - **Check Console**: Open browser Inspect > Console for JavaScript errors (e.g., "fetch error").
  - **Check Terminal**: Ensure Flask logs POST requests (`Received input:`).
  - **Fix**: Verify `.user-input { flex-grow: 1; }` in CSS.

- **No Response**:
  - **API Failure**: OpenFDA might be down; check `get_openfda_details()` output.
  - **CSV Issue**: Ensure `Medicine_Details_Final.csv` exists and has correct columns.

- **Server Not Starting**:
  - **Dependencies**: Run `pip install -r requirements.txt`.
  - **Port Conflict**: Change `app.run(port=5001)` if 5000 is in use.

- **Debugging**:
  - Add `console.log` in `sendMessage()` or `print` in Python to trace execution.

## Contributing

1. Fork the repository.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Suggestions:
- Add intent detection with NLP (e.g., spaCy, NLTK).
- Improve UI with modern styling (e.g., Bootstrap).
- Cache API responses for performance.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
```

---

### Instructions
1. **Create the File**:
   - Open a text editor (e.g., VS Code).
   - Copy the above content.
   - Save it as `README.md` in your project root (`C:\Users\Anil Abhange\Downloads\Chatbot\`).

2. **Push to GitHub**:
   - Initialize a Git repo if not already done:
     ```bash
     git init
     git add README.md chatbot.py Medicine_Details_Final.csv
     git commit -m "Initial commit with README and chatbot"
     git remote add origin https://github.com/yourusername/medicine-chatbot.git
     git push -u origin main
     ```
   - Replace `yourusername` with your GitHub username.

3. **Optional Additions**:
   - Add a `requirements.txt` file with:
     ```
     pandas
     scikit-learn
     flask
     requests
     ```
   - Include `Medicine_Details_Final.csv` or a sample in the repo.
