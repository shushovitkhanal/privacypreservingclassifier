import time
from contextlib import contextmanager
import pandas as pd
import matplotlib.pyplot as plt
import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from Client import Client
from Server import Server


@contextmanager
def timer():
    start = time.time()
    yield
    end = time.time()
    print(f"Execution Time: {end - start}\n")


class Classifier:
    def __init__(self, spam_folder, ham_folder):
        self.spam_folder = spam_folder
        self.ham_folder = ham_folder

    def custom_tokenizer(self, text):
        tokens = nltk.word_tokenize(text)

        tokens_lower = []
        for token in tokens:
            tokens_lower.append(token.lower())

        stop_words = set(stopwords.words('english'))
        punctuation = set(string.punctuation)

        filtered_elements = []
        for element in tokens_lower:
            if element not in stop_words and element not in punctuation and not element.isdigit():
                filtered_elements.append(element)
        return filtered_elements

    def vectorizeData(self, spam_path, ham_path):

        spam_files = os.listdir(spam_path)
        ham_files = os.listdir(ham_path)

        corpus = []
        y = []
        for filenames in spam_files:
            sentence = ""
            cur_file = open(spam_path + "/" + filenames, errors='ignore')
            sentence = cur_file.read()
            corpus.append(sentence)
            y.append(1)

        for filenames in ham_files:
            sentence = ""
            cur_file = open(ham_path + "/" + filenames, errors='ignore')
            sentence = cur_file.read()
            corpus.append(sentence)
            y.append(0)

        vectorizer = TfidfVectorizer(tokenizer=self.custom_tokenizer, min_df=0.1)
        X = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()

        X, y = shuffle(X, y, random_state=0)

        print(y)
        tfidf = pd.DataFrame(X.toarray(), columns=feature_names)
        print(tfidf)
        return X, y, feature_names

    def shuffle_data(self, X, y):
        X, y = shuffle(X, y, random_state=0)
        return X, y

    def run(self):
        spam_folder = self.spam_folder
        ham_folder = self.ham_folder
        X, y, feature_names = self.vectorizeData(spam_folder, ham_folder)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        print("Server: Generating paillier keypair")
        server = Server()
        server.create_paillier_keypair()

        print("Server: Training the machine learning model")
        with timer() as t:
            server.fit(X_train, y_train)
            y_pred = server.predict(X_test)
        print('Normal Accuracy without encryption: {score}'.format(score=accuracy_score(y_test, y_pred)))
        conf_matrix = confusion_matrix(y_test, y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['ham', 'spam'])
        cm_display.plot()
        plt.show()

        with timer() as t:
            print("Server: Encrypting classifier")
            encrypted_weights, encrypted_intercept = server.encrypt_params()

        with timer() as t:
            print("Client: scoring with encrypted classifier")
            client = Client(server.get_public_key())
            encrypted_scores = client.encrypted_predict(encrypted_weights, encrypted_intercept, X_test)

        print("Server: Decrypting client's scores")
        with timer() as t:
            scores = server.decrypt_score(encrypted_scores)
        print('Accuracy with encryption: {score}'.format(score=accuracy_score(y_test, scores)))
        conf_matrix = confusion_matrix(y_test, scores)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['ham', 'spam'])
        cm_display.plot()
        plt.show()
