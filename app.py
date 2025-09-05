import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import streamlit as st

lemmatizer = WordNetLemmatizer()

# Load files
intents = json.loads(open(r'C:\Users\User\Desktop\ChatBot Using Python\intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
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
    if len(ints) == 0:
        return "Sorry, I didn’t quite get that. Can you rephrase?"
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm still learning. Please try asking something else."

def chatbot_response(msg):
    ints = predict_class(msg)
    return get_response(ints, intents)

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

st.title("🤖 AI Chatbot")
st.write("Ask me anything!")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    reply = chatbot_response(user_input)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", reply))

# Display chat history
for role, text in st.session_state.history:
    if role == "You":
        st.markdown(f"**🧑 {role}:** {text}")
    else:
        st.markdown(f"**🤖 {role}:** {text}")
