# Complaint Classification System (NLP Mini-Project)

A beginner-friendly academic prototype that classifies complaint text into one of eight categories using TF-IDF and Logistic Regression, with a Streamlit web interface.

## Features
- Classifies a complaint into one of 8 categories.
- Suggests a relevant department.
- Displays model scores by category.
- Includes sample examples and a downloadable sample CSV dataset.
- Shows a test-set accuracy, classification report, and confusion matrix.

## Requirements
- Python 3.10 or newer recommended
- Internet connection for the first-time package installation

## Run on Windows
1. Extract the ZIP file.
2. Open the extracted `complaint_classification_system` folder.
3. Click the folder address bar in File Explorer, type `cmd`, and press Enter.
4. (Optional but recommended) Create a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

5. Install packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

6. Start the app:

   ```bash
   python -m streamlit run app.py
   ```

7. Streamlit will show a local URL, usually `http://localhost:8501`. Open it in your browser.

## Run in VS Code
Open the folder in VS Code, choose Terminal > New Terminal, then run the install and start commands above.

## How it works
1. The sample dataset stores each complaint alongside its category.
2. Basic cleaning converts text to lowercase and removes punctuation.
3. `TfidfVectorizer` converts words and word pairs into numerical features.
4. Logistic Regression learns patterns that associate those features with complaint categories.
5. A held-out test set is used to calculate evaluation metrics.
6. Streamlit displays the prediction and evaluation information in a browser.

## Important academic note
The included dataset is small and manually written for demonstration. Metrics may vary with code or package versions and are not representative of real-world performance. A production system needs a larger, diverse, accurately labelled dataset and careful evaluation. Do not upload personal or confidential complaint details.

## Project structure
- `app.py` — full Streamlit application and sample dataset
- `requirements.txt` — required Python packages
- `README.md` — setup instructions
