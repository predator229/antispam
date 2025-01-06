import numpy as np
import pandas as pd
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import nltk.data
import warnings
import ssl
import nltk
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import subprocess
import socket
from threading import Thread, Lock
import time
from googletrans import Translator


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize necessary tools
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
label_encoder = LabelEncoder()
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
warnings.filterwarnings("ignore")
random_forest_model = RandomForestRegressor()

# Load dataset
data = pd.read_csv("sms_datasets/sms_dataset.csv")
new_datas = data[['label', 'message']]

# Configuration constants
one_process_time = 459.9088180065155
script_name = 'analyser.py'
data_types = ['speed-check', 'profile-analysis', 'history', 'full-discursion-check']
data_formats = {
    'speed-check': {
        'token': 'token',
        'type': 'speed-check',
        'content': {
            'message': 'content message',
        },
    },
    'history': {
        'token': 'token',
        'type': 'history',
        'content': {
            'email': 'your@email.com',
            'verification-code': 'code-sent-per-email',
        },
    },
    'profile-analysis': {
        'token': 'token',
        'type': 'profile-analysis',
        'content': {
            'url': 'https://facebook.com/something',
        },
    },
    'full-discursion-check': {
        'token': 'token',
        'type': 'profile-analysis',
        'content': {
            'file': 'contentfile',
        },
    }
}
new_datas = pd.read_csv("sms_datasets/sms_dataset.csv")[['label', 'message']]

MAX_ANALYZER_PROCESSES = 5
MAX_TICKETS = 10

def remove_punctuation(msg):
    """Remove punctuation and apply lemmatization and stopword removal."""
    msg = re.sub(r'[^a-zA-Z0-9\s]', '', msg)
    trans = str.maketrans('', '', string.punctuation)
    msg = msg.translate(trans)
    words = [lemmatizer.lemmatize(word) for word in msg.split() if word not in stop_words]
    return " ".join(words)

def calculate_wait_time():
    """Calculate the estimated wait time based on the number of running analyzers and available tickets."""
    running_analyzers_count = get_running_analyzers_count()
    wait_time = (running_analyzers_count * one_process_time) + ((running_analyzers_count - ticket_system.available_tickets) * one_process_time)
    return wait_time if wait_time > 0 else 30

def visualize_dataset(new_datas):
    """Generate visualizations for the dataset."""
    labels = new_datas['label'].value_counts().index
    plt.figure(figsize=(20, 22))

    stats = {}
    groups_msg = {}
    colors = ['green', 'red', 'yellow']

    for lab, color in zip(labels, colors):
        stats[lab] = new_datas[new_datas['label'] == lab][['num_characters', 'num_words', 'num_sentences']].describe()
        sns.histplot(new_datas[new_datas['label'] == lab]['num_characters'], color=color, label=lab, kde=True)
        groups_msg[lab] = new_datas[new_datas.label == lab].message

    # Create visualizations directory if needed
    os.makedirs('visualizations/datasets', exist_ok=True)

    plt.legend(title='Labels')
    plt.title('Character Count Distribution by Label')
    plt.xlabel('Number of Characters')
    plt.ylabel('Frequency')
    plt.savefig('visualizations/datasets/char_count_by_label.png')
    plt.close()

    for lab, wc in groups_msg.items():
        wordcloud = WordCloud(width=1500, height=900, max_words=2500).generate(' '.join(wc))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {lab} Messages')
        plt.savefig(f'visualizations/datasets/wordcloud_{lab}.png')
        plt.close()

def visualize_and_compare_with_new_messages(received_messages, model, new_datas):
    """Visualize and compare new messages with the existing dataset."""
    transformed_messages = [remove_punctuation(msg.lower()) for msg in received_messages]
    new_x_text = tfidf_vectorizer.transform(transformed_messages).toarray()

    # Predict with text-based model
    predictions_text = model.predict(new_x_text)

    # Statistical characteristics
    new_data_lengths = pd.DataFrame({
        'num_characters': [len(msg) for msg in received_messages],
        'num_words': [len(msg.split()) for msg in received_messages],
        'num_sentences': [len(sent_tokenize(msg)) for msg in received_messages]
    })

    # Model based on statistical features
    x_stat = new_datas[['num_characters', 'num_words', 'num_sentences']]
    y_stat = new_datas['label']
    model_stat = RandomForestRegressor(random_state=0)
    model_stat.fit(x_stat, y_stat)
    predictions_stat = model_stat.predict(new_data_lengths)

    # Combine predictions from both models
    final_predictions = (predictions_text + predictions_stat) / 2
    final_predictions = [round(pred) for pred in final_predictions]

    label_map = {i: label for i, label in enumerate(label_encoder.classes_)}

    results = [{'type': 'General', 'title': '', 'datas': ['Analysis of new messages:']}]

    for i, msg in enumerate(received_messages):
        label = label_map[final_predictions[i]]

        result_info = [
            f"Message: {msg}",
            f"Prediction: {label}",
            f"Number of characters: {len(msg)}",
            f"Number of words: {len(msg.split())}",
            f"Number of sentences: {len(sent_tokenize(msg))}"
        ]

        results.append({'type': 'Result', 'title': '', 'datas': result_info})

    # Visualize new messages in a Word Cloud
    plt.figure(figsize=(20, 22))
    wc = WordCloud(width=1500, height=900, max_words=2500).generate(' '.join(transformed_messages))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Used Words in the Messages')
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'visualizations/wordcloud_new_messages_{current_date}.png')
    results.append({'type': 'image', 'title': 'Most Used Words in the Messages', 'datas': [f'visualizations/wordcloud_new_messages_{current_date}.png']})

    # Compare word frequencies
    existing_data_words = ' '.join(new_datas['message'])
    new_messages_words = ' '.join(transformed_messages)

    existing_wc = WordCloud(width=1500, height=900, max_words=2500).generate(existing_data_words)
    new_wc = WordCloud(width=1500, height=900, max_words=2500).generate(new_messages_words)

    plt.figure(figsize=(20, 22))
    plt.subplot(1, 2, 1)
    plt.imshow(existing_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Existing Messages')

    plt.subplot(1, 2, 2)
    plt.imshow(new_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of New Messages')

    plt.savefig(f'visualizations/comparison_wordcloud_{current_date}.png')
    results.append({'type': 'image', 'title': 'Comparison of Most Used Words in the Messages and Dataset', 'datas': [f'visualizations/comparison_wordcloud_{current_date}.png']})

    # Descriptive statistics
    new_data_lengths['label'] = ['spam' if pred == 1 else 'ham' if pred == 0 else 'smishing' for pred in final_predictions]

    # Add histograms for new messages
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(new_data_lengths['num_characters'], bins=20, kde=True, color='blue')
    plt.title('Distribution of Character Counts')

    plt.subplot(1, 3, 2)
    sns.histplot(new_data_lengths['num_words'], bins=20, kde=True, color='green')
    plt.title('Distribution of Word Counts')

    plt.subplot(1, 3, 3)
    sns.histplot(new_data_lengths['num_sentences'], bins=10, kde=True, color='red')
    plt.title('Distribution of Sentence Counts')

    plt.tight_layout()
    plt.savefig(f'visualizations/new_messages_histogram_{current_date}.png')
    results.append({'type': 'image', 'title': 'Distributions of the Messages Statistics', 'datas': [f'visualizations/new_messages_histogram_{current_date}.png']})

    return results


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    mae = mean_absolute_error(val_y, model.predict(val_X))
    return mae

def transformDataset(new_datas):
    new_datas['message'] = new_datas['message'].str.lower()  # Conversion de tous les caractères en minuscule
    new_datas['message'] = new_datas['message'].apply(remove_punctuation)  # Retrait de toutes les ponctuations
    new_datas['label'] = new_datas['label'].str.lower()

    new_datas['num_characters'] = new_datas['message'].apply(len)  # Calcul du nombre de caractères
    new_datas['num_words'] = new_datas['message'].apply(lambda x: len(nltk.word_tokenize(x)))  # Calcul du nombre de mots
    new_datas['num_sentences'] = new_datas['message'].apply(lambda x: len(nltk.sent_tokenize(x)))  # Calcul du nombre de phrases
    return new_datas

def obtainXY(new_datas):
    new_datas['label'] = label_encoder.fit_transform(new_datas['label'])
    X = tfidf_vectorizer.fit_transform(new_datas['message']).toarray()
    y = new_datas['label']
    return X, y

def getbestTreeSizeFromTrain(X, y):
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    candidate_max_leaf_nodes = [5, 10, 20, 30, 40, 25, 50, 55, 40, 200, 100, 250, 500]
    scores = {node_leaf: get_mae(node_leaf, train_X, val_X, train_y, val_y) for node_leaf in candidate_max_leaf_nodes}
    best_tree_size = min(scores, key=scores.get)
    return best_tree_size

class TicketSystem:
    def __init__(self):
        self.lock = Lock()
        self.available_tickets = MAX_TICKETS
    def acquire_ticket(self):
        with self.lock:
            if self.available_tickets > 0:
                self.available_tickets -= 1
                return True
            else:
                return False
    def release_ticket(self):
        with self.lock:
            self.available_tickets += 1
ticket_system = TicketSystem()

class RequestHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        json_data = json.loads(post_data.decode())
        if "type" in json_data and json_data["type"] in data_types:
            format_valid = True
            problems = []
            for key in data_types[json_data["type"]]["content"].items():
                if key not in json_data["content"] or not isinstance(json_data["content"][key], str):
                    format_valid = False
                    problems.append(key)
                    break
            format_valid = True

            if format_valid:
                if json_data["type"] in data_formats:
                    if ticket_system.acquire_ticket():
                        if get_running_analyzers_count() < MAX_ANALYZER_PROCESSES:
                            start_time = time.time()
                            if json_data['type'] == 'speed-check':
                                response = run_analyzer([translate_text(json_data['content']['message'])])
                                oneprocesstime = time.time() - start_time
                                response_data = {"status": "success", "message": "Analysis completed", "response": response}
                                ticket_system.release_ticket()
                        else:
                            estimated_wait_time = calculate_wait_time()
                            response_data = {"status": "pending" , 'waitingtime': estimated_wait_time, "message": f"Please wait. Your request is pending. The estimated waiting time {estimated_wait_time} secondes."}
                    else:
                        response_data = {"status": "sendback", 'waitingtime':oneprocesstime, "message": "No free tiquets availables! We trying to get process your request ! ... "}
                else:
                    response_data = {"status": "error", "message": "Invalid request"}
            else:
                problems = [str(problem) for problem in problems]  # Convertir tous les éléments en chaînes de caractères
                response_data = {"status": "error", "message": "Datas format not valide "+" ".join(problems)}
        else:
            response_data = {"status": "error", "message": "Datas type not valide"}

        response_json = json.dumps(response_data)
        self.wfile.write(response_json.encode())

def get_running_analyzers_count():
    countScriptRunning = 0
    ps_output = subprocess.check_output(["ps", "-ef"])
    ps_lines = ps_output.decode("utf-8").split("\n")
    for line in ps_lines:
        if script_name in line:
            countScriptRunning+=1
    return countScriptRunning

def get_time_process_in_progress():
    return 20

def run_analyzer(request_data):
    return vizualizationAndComparationwithNewMessages(request_data, model,new_datas)

def translate_text(text, dest_language='en'):
    try:
        translator = Translator()        
        translated = translator.translate(text, dest=dest_language)
        return translated.text
    except Exception as e:
        return f"Translation error: {e}"

# def main():
# Vectorisation et transformation du dataset
new_datas = transformDataset(new_datas)
if not os.path.exists('visualizations'):
    visualize_and_compare_with_new_messages(new_datas)

X, y = obtainXY(new_datas)
best_tree_size = getbestTreeSizeFromTrain(X, y)

# Prédiction des nouveaux messages
model = RandomForestRegressor(max_leaf_nodes=best_tree_size, random_state=0)
model.fit(X, y)

hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
server_address = (local_ip, 12345)

print(server_address)

httpd = HTTPServer(server_address, RequestHandler)
print("==================================================================")
print(" ===> Server is listerning on port 12345...")
httpd.serve_forever()

# if __name__ == "__main__":
#     # Configure SSL context for nltk downloads
#     main()
