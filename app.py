import time
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_cors import CORS
import openai
import spacy
import os

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'your_secret_key'  # Change this to a secret key for your application
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['USERS_FILE'] = 'users.txt'  # File to store username and password credentials
app.config['REGISTERED_FILE'] = 'registered.txt'  # File to store registered user details

# Create the uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Check if the users file exists, and create it if not
if not os.path.exists(app.config['USERS_FILE']):
    with open(app.config['USERS_FILE'], 'w'):
        pass

# Check if the registered file exists, and create it if not
if not os.path.exists(app.config['REGISTERED_FILE']):
    with open(app.config['REGISTERED_FILE'], 'w'):
        pass

# Set your OpenAI API key
openai.api_key = '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'

# Load the spaCy English model
nlp = spacy.load("en_core_web_md")

# Placeholder for storing the document content
document_content = ""

def load_user_credentials():
    users = {}
    with open(app.config['USERS_FILE'], 'r') as file:
        lines = file.readlines()
        for line in lines:
            username, password = line.strip().split(':')
            users[username] = password
    return users

def check_user_credentials(username, password, users):
    return username in users and users[username] == password

def add_user_credentials(username, password):
    with open(app.config['USERS_FILE'], 'a') as file:
        file.write(f"{username}:{password}\n")

def add_registered_user_details(username, name, email, phone):
    with open(app.config['REGISTERED_FILE'], 'a') as file:
        file.write(f"Username: {username}\nName: {name}\nEmail: {email}\nPhone: {phone}\n\n")

# Initialize the users file with default credentials
if os.path.getsize(app.config['USERS_FILE']) == 0:
    add_user_credentials('Tilak', 'SK')
    add_user_credentials('user2', 'pass2')
    add_user_credentials('user3', 'pass3')

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    # Load user credentials
    users = load_user_credentials()

    # Check credentials
    if check_user_credentials(username, password, users):
        flash('Login successful', 'success')
        return redirect(url_for('upload_page'))
    else:
        flash('Login failed. Check your username and password.', 'danger')
        return redirect(url_for('index'))

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register_user():
    username = request.form['Username']
    password = request.form['password']
    name = request.form['Name']
    email = request.form['email']
    phone = request.form['phone']

    # Load user credentials
    users = load_user_credentials()

    # Check if the username already exists
    if username in users:
        flash('Username already exists. Choose a different username.', 'danger')
        return redirect(url_for('register'))

    # Add user credentials
    add_user_credentials(username, password)
    add_registered_user_details(username, name, email, phone)
    flash('Registration successful. You can now log in.', 'success')

    # Redirect to the login page after successful registration
    return redirect(url_for('index'))

@app.route('/upload-page')
def upload_page():
    return render_template('upload.html')

@app.route('/upload-document', methods=['POST'])
def upload_document():
    global document_content

    try:
        file = request.files['file']

        # Read the uploaded text file content
        document_content = file.read().decode('utf-8')

        print("Document Content:", document_content)  # Debugging line

        # Introduce a delay of 1 second (import time if not already imported)
        time.sleep(1)

        return jsonify({'response': 'Document uploaded successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/ask-chatgpt', methods=['POST'])
def ask_chatgpt():
    global document_content

    try:
        query = request.form['query']

        # Check if the query is related to the document content
        if is_related_to_document(query):
            # Combine user query with document context
            prompt = f"User: {query}\nDoc: {document_content}"
        else:
            # Use only user query for unrelated questions
            prompt = f"User: {query}"

        # Use ChatGPT API for user query and document context
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )

        chatgpt_response = response.choices[0].text.strip()

        # If the response is empty, attempt to find an exact match in the document
        if not chatgpt_response:
            exact_match_answer = get_exact_match_answer(query)
            if exact_match_answer:
                return jsonify({'response': exact_match_answer})

        return jsonify({'response': chatgpt_response})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/ask')
def ask():
    return render_template('asking.html')

def is_related_to_document(query):
    # Use spaCy for NLP processing
    doc = nlp(document_content)
    query_doc = nlp(query)

    # Find sentences in the document that are semantically similar to the user's query
    similarity_scores = [query_doc.similarity(sent) for sent in doc.sents]
    most_similar_score = max(similarity_scores)

    # Define a threshold for relevance
    relevance_threshold = 0.7

    return most_similar_score > relevance_threshold

def get_exact_match_answer(query):
    # Tokenize the document and user query
    doc_tokens = nlp(document_content.lower())
    query_tokens = nlp(query.lower())

    # Check if any of the query tokens are present in the document
    for token in query_tokens:
        if token.text.lower() in [t.text.lower() for t in doc_tokens]:
            return extract_relevant_info(document_content, token.text)

    return None

def extract_relevant_info(document_content, query):
    # Tokenize the document and user query
    doc_tokens = nlp(document_content.lower())
    query_tokens = nlp(query.lower())

    # Calculate similarity based on token overlap
    similarity_scores = [token.similarity(query_tokens) for token in doc_tokens]

    # Get the index of the most similar token
    most_similar_index = similarity_scores.index(max(similarity_scores))

    # Get the corresponding sentence
    most_similar_sentence = list(doc_tokens.sents)[most_similar_index]

    return most_similar_sentence.text

@app.route('/search-in-document', methods=['POST'])
def search_in_document():
    global document_content

    try:
        # Use ChatGPT API for document context
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Doc: {document_content}",
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )

        chatgpt_response = response.choices[0].text.strip()

        return jsonify({'response': chatgpt_response})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)