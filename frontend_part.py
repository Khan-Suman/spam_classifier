# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 11:57:09 2025

@author: suman
"""

import nltk
import pickle
import string
import numpy as np
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the saved model and vectorizer
tfidf = pickle.load(open("D:/sms_spam_classifier/vectorizer.pkl", 'rb'))
model = pickle.load(open("D:/sms_spam_classifier/model.pkl", 'rb'))

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    
    return " ".join(y)

def spam_prediction(input_text):
    transformed_text = transform_text(input_text)
    vector_input = tfidf.transform([transformed_text])
    prediction = model.predict(vector_input)[0]
    
    return "Spam" if prediction == 1 else "Not Spam"

def main():
    st.title("SMS/Email Spam Classifier")
    
    input_sms = st.text_area("Enter the message")
    
    classification = ""
    if st.button("Classify Message"):
        classification = spam_prediction(input_sms)
    
    st.success(classification)

if __name__ == '__main__':
    main()
