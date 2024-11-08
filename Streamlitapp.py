import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('Next_Word_LSTM.h5')

with open('tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    # Convert the input text to a sequence of tokens
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # Ensure token list is no longer than the max sequence length
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    
    # Pad sequences to ensure consistent input length
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    
    # Use the model to predict the next word's index
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=-1)[0]
    
    # Map the predicted index back to the word
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    
    return None  # In case the word is not found (which shouldn't happen)

st.title('Next Word Prediction with LSTM')
input_text = st.text_input('Enter the sequence of words','To be or not to be')
if st.button('Predict Next Word'):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f'Next Word:{next_word}')
