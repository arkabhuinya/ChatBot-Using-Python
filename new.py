import nltk
nltk.download('punkt')
nltk.download('punkt_tab')   # <-- important in newer NLTK
nltk.download('wordnet')

import random
import json
import pickle
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Load intents
with open(r'C:\Users\User\Desktop\ChatBot Using Python\intents.json', encoding='utf-8') as f:
    intents = json.load(f)

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Tokenize patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize + clean words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))

# Save vocabulary + classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Training data
training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1 if word in wordPatterns else 0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

# Make sure data is consistent
random.shuffle(training)
training = np.array(training, dtype=object)

# Split into X (inputs) and Y (labels)
trainX = np.array(list(training[:, :len(words)]), dtype=np.float32)
trainY = np.array(list(training[:, len(words):]), dtype=np.float32)

# Build model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Compile
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train
hist = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)

# Save model
model.save('chatbot_model.h5')
print("✅ Training complete. Model saved as chatbot_model.h5")
