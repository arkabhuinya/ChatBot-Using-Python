import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load files
intents = json.loads(open(r'C:\Users\User\Desktop\ChatBot Using Python\intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the user input"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    """Return bag of words array"""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """Predict intent of user input"""
    bow_data = bow(sentence, words)
    res = model.predict(np.array([bow_data]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    """Get bot response based on intent"""
    if len(ints) == 0:
        return "Sorry, I didn’t quite get that. Can you rephrase?"

    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

    return "I'm still learning. Please try asking something else."

def chatbot_response(msg):
    """Generate response for user message"""
    ints = predict_class(msg)
    res = get_response(ints, intents)
    return res

# Simple CLI loop
if __name__ == "__main__":
    print("🤖 Chatbot is ready! (type 'quit' to exit)")
    while True:
        message = input("You: ")
        if message.lower() == "quit":
            print("Bot: Goodbye! 👋")
            break
        response = chatbot_response(message)
        print("Bot:", response)
