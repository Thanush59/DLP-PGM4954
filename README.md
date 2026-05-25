3.
# Import Libraries
import numpy as np
import pandas as pd
import nltk
import re

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# Download Required NLTK Data
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords

# ---------------- LOAD DATASET ---------------- #

# Load IMDB Dataset
df = pd.read_csv("IMDB Dataset.csv")

print("Dataset Loaded Successfully\n")
print(df.head())

# Take only first 500 reviews for faster training
df = df.head(500)

# ---------------- TEXT PREPROCESSING ---------------- #

stop_words = set(stopwords.words('english'))

def preprocess(text):

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords
    words = [
        word for word in words
        if word not in stop_words
    ]

    return words

# Apply preprocessing
corpus = df["review"].apply(preprocess)

print("\nSample Tokenized Review:\n")
print(corpus.iloc[0])

# ---------------- CBOW MODEL ---------------- #

# sg = 0  ---> CBOW Model
cbow_model = Word2Vec(
    sentences=corpus,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=0
)

print("\nCBOW Model Trained Successfully")

# ---------------- SKIP-GRAM MODEL ---------------- #

# sg = 1 ---> Skip-Gram Model
skipgram_model = Word2Vec(
    sentences=corpus,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1
)

print("\nSkip-Gram Model Trained Successfully")

# ---------------- WORD SIMILARITIES ---------------- #

words_to_check = ["movie", "good", "bad"]

print("\n===== CBOW Similar Words =====\n")

for word in words_to_check:

    if word in cbow_model.wv:

        print(f"Top similar words for '{word}' :")

        similar_words = cbow_model.wv.most_similar(word, topn=5)

        for similar_word, score in similar_words:
            print(similar_word, ":", round(score, 3))

        print()

print("\n===== SKIP-GRAM Similar Words =====\n")

for word in words_to_check:

    if word in skipgram_model.wv:

        print(f"Top similar words for '{word}' :")

        similar_words = skipgram_model.wv.most_similar(word, topn=5)

        for similar_word, score in similar_words:
            print(similar_word, ":", round(score, 3))

        print()

# ---------------- WORD VECTOR VISUALIZATION ---------------- #

# Select words for visualization
visualize_words = [
    "movie",
    "good",
    "bad",
    "love",
    "story",
    "actor",
    "film",
    "best",
    "worst",
    "comedy"
]

# Store vectors
word_vectors = []

valid_words = []

for word in visualize_words:

    if word in cbow_model.wv:

        word_vectors.append(cbow_model.wv[word])

        valid_words.append(word)

# Convert into numpy array
word_vectors = np.array(word_vectors)

# Apply PCA
pca = PCA(n_components=2)

result = pca.fit_transform(word_vectors)

# ---------------- PLOT WORD VECTORS ---------------- #

plt.figure(figsize=(10, 8))

for i, word in enumerate(valid_words):

    plt.scatter(result[i, 0], result[i, 1])

    plt.text(
        result[i, 0] + 0.01,
        result[i, 1] + 0.01,
        word
    )

plt.title("Word2Vec Word Embedding Visualization using PCA")

plt.xlabel("PCA Component 1")

plt.ylabel("PCA Component 2")

plt.grid()

plt.show()


4.
# Next Word Prediction using Neural Language Model (LSTM)

# Import Libraries
import numpy as np
import pandas as pd
import re

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

from sklearn.model_selection import train_test_split

# ---------------- LOAD DATASET ---------------- #

# Load text dataset
df = pd.read_csv("scifi.csv")

print("Dataset Loaded Successfully\n")

print(df.head())

# ---------------- TEXT PREPROCESSING ---------------- #

# Combine all text into one paragraph
text = " ".join(df.iloc[:, 0].astype(str))

# Convert to lowercase
text = text.lower()

# Remove special characters
text = re.sub(r'[^a-zA-Z\s]', '', text)

print("\nSample Cleaned Text:\n")

print(text[:500])

# ---------------- TOKENIZATION ---------------- #

tokenizer = Tokenizer()

tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

print("\nTotal Vocabulary Size :", total_words)

# ---------------- CREATE INPUT SEQUENCES ---------------- #

input_sequences = []

token_list = tokenizer.texts_to_sequences([text])[0]

for i in range(1, len(token_list)):

    n_gram_sequence = token_list[:i + 1]

    input_sequences.append(n_gram_sequence)

print("\nTotal Input Sequences :", len(input_sequences))

# ---------------- PAD SEQUENCES ---------------- #

max_sequence_len = max([len(seq) for seq in input_sequences])

input_sequences = np.array(
    pad_sequences(
        input_sequences,
        maxlen=max_sequence_len,
        padding='pre'
    )
)

# ---------------- SPLIT INPUT AND OUTPUT ---------------- #

X = input_sequences[:, :-1]

y = input_sequences[:, -1]

# One-hot encoding output
from tensorflow.keras.utils import to_categorical

y = to_categorical(y, num_classes=total_words)

# ---------------- TRAIN TEST SPLIT ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- BUILD LSTM MODEL ---------------- #

model = Sequential()

model.add(
    Embedding(
        input_dim=total_words,
        output_dim=100,
        input_length=max_sequence_len - 1
    )
)

model.add(LSTM(150))

model.add(Dense(total_words, activation='softmax'))

# ---------------- COMPILE MODEL ---------------- #

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ---------------- TRAIN MODEL ---------------- #

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ---------------- EVALUATE MODEL ---------------- #

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy :", accuracy)

# Perplexity Calculation
perplexity = np.exp(loss)

print("Perplexity :", perplexity)

# ---------------- NEXT WORD PREDICTION ---------------- #

def predict_next_words(seed_text, next_words=3):

    print("\nInput :", seed_text)

    for i in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding='pre'
        )

        predicted = np.argmax(
            model.predict(token_list, verbose=0),
            axis=-1
        )[0]

        output_word = ""

        for word, index in tokenizer.word_index.items():

            if index == predicted:

                output_word = word

                break

        seed_text += " " + output_word

    print("Predicted Sequence :", seed_text)

# ---------------- TEST PREDICTIONS ---------------- #

predict_next_words("the movie was")

predict_next_words("once upon a")

predict_next_words("the hero")


5.


# Sentiment Analysis with Neural Model (LSTM)

# Import Libraries
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# ---------------- LOAD DATASET ---------------- #

# Load IMDB Dataset
df = pd.read_csv("IMDB Dataset.csv")

print("Dataset Loaded Successfully\n")

print(df.head())

# Take first 5000 rows for faster execution
df = df.head(5000)

# ---------------- LABEL ENCODING ---------------- #

# Convert sentiments into binary values
df["label"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

print("\nLabel Counts:\n")

print(df["label"].value_counts())

# ---------------- TEXT PREPROCESSING ---------------- #

def preprocess(text):

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text

df["clean_review"] = df["review"].apply(preprocess)

print("\nSample Cleaned Review:\n")

print(df["clean_review"].iloc[0])

# ---------------- TOKENIZATION ---------------- #

max_words = 10000

tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(df["clean_review"])

# Convert text into sequences
sequences = tokenizer.texts_to_sequences(df["clean_review"])

# ---------------- PADDING ---------------- #

max_length = 100

X = pad_sequences(
    sequences,
    maxlen=max_length
)

y = df["label"].values

print("\nShape of X :", X.shape)

print("Shape of y :", y.shape)

# ---------------- TRAIN TEST SPLIT ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- BUILD LSTM MODEL ---------------- #

model = Sequential()

# Embedding Layer
model.add(
    Embedding(
        input_dim=max_words,
        output_dim=128,
        input_length=max_length
    )
)

# LSTM Layer
model.add(LSTM(64))

# Dropout Layer
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

# ---------------- COMPILE MODEL ---------------- #

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------------- MODEL SUMMARY ---------------- #

print("\nModel Summary:\n")

model.summary()

# ---------------- TRAIN MODEL ---------------- #

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# ---------------- MODEL EVALUATION ---------------- #

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy :", accuracy)

# ---------------- PREDICTIONS ---------------- #

y_pred_prob = model.predict(X_test)

# Convert probabilities into binary values
y_pred = (y_pred_prob > 0.5).astype(int)

# ---------------- PERFORMANCE METRICS ---------------- #

print("\nF1 Score :", f1_score(y_test, y_pred))

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# ---------------- MISCLASSIFIED EXAMPLES ---------------- #

print("\nMisclassified Examples:\n")

misclassified = []

for i in range(len(y_test)):

    if y_test[i] != y_pred[i]:

        misclassified.append(i)

# Display first 5 misclassified reviews
for i in misclassified[:5]:

    print("\nReview :")

    print(df.iloc[i]["review"])

    print("Actual Label :", y_test[i])

    print("Predicted Label :", y_pred[i][0])

# ---------------- ACCURACY GRAPH ---------------- #

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(['Train', 'Validation'])

plt.show()

# ---------------- LOSS GRAPH ---------------- #

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(['Train', 'Validation'])

plt.show()
