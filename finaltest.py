from flask import Flask,request,jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Concatenate
from nltk.tokenize import word_tokenize

app=Flask(__name__)
CORS(app)
# Load the trained model and tokenizers
model = load_model("model.h5")
input_tokenizer = Tokenizer()
target_tokenizer = Tokenizer()
with open("input_tokenizer.pkl", "rb") as file:
    input_tokenizer = pickle.load(file)
# Load the target tokenizer from file
with open("target_tokenizer.pkl", "rb") as file:
    target_tokenizer = pickle.load(file)
train=pd.read_csv("train.csv")
#tweet column is input 
input_data=train["tweet"]
target_data=train.iloc[:,4:].values
target_label=train.iloc[:,4:].columns.tolist()
target_label_val=["I can't tell","Negative","Neutral","Positive","Tweet not related to weather condition","current (same day) weather",
                  "future (forecast)","I can't tell","past weather",
"clouds","cold","dry","hot","humid","hurricane","I can't tell","ice","other","rain","snow","storms","sun","tornado","wind"]

max_input_length = 14 
max_target_length=16 
latent_dim=500
label_value = dict((i, v) for i, v in zip(target_label, target_label_val))

encoder_model = Model(model.input[0], model.layers[6].output)
decoder_input_state_h = Input(shape=(latent_dim,), name='decoder_input_h')
decoder_input_state_c = Input(shape=(latent_dim,), name='decoder_input_c')
decoder_hidden_input_state = Input(shape=(max_input_length, latent_dim), name='decoder_hidden_input_state')

# Get the all the layers from the model
decoder_inputs = model.input[1]
decoder_embed_layer = model.layers[5]
decoder_lstm = model.layers[7]
decoder_attn = model.layers[8]
decoder_dense_layer = model.layers[10]
decoder_emb = decoder_embed_layer(decoder_inputs)
decoder_lstm_outputs, state_h, state_c = decoder_lstm(decoder_emb, initial_state=[decoder_input_state_h, decoder_input_state_c])

decoder_attn_output = decoder_attn([decoder_lstm_outputs, decoder_hidden_input_state])
decoder_merged = Concatenate(axis=-1, name='concat_layer1')([decoder_lstm_outputs, decoder_attn_output])

decoder_outputs = decoder_dense_layer(decoder_merged)
decoder_model = Model([decoder_inputs, decoder_hidden_input_state, decoder_input_state_h, decoder_input_state_c],[decoder_outputs, state_h, state_c])

reverse_target_wordidx = target_tokenizer.index_word
reverse_input_wordidx = input_tokenizer.index_word
tar_word_index = target_tokenizer.word_index
reverse_target_wordidx[0]=' '
 
def decode_sequence(input_sequence):
    encoder_out, encoder_h, encoder_c= encoder_model.predict(input_sequence)
    target_sequence = np.zeros((1, 1))
    target_sequence[0, 0] = tar_word_index['sos']
    stoping_condn = False
    decoded_sentence = ""
    while not stoping_condn: 
      output_words, dec_h, dec_c= decoder_model.predict([target_sequence] + [encoder_out,encoder_h, encoder_c])
      word_index = np.argmax(output_words[0, -1, :])
      text_word = reverse_target_wordidx[word_index]
      decoded_sentence += text_word +" "
      if text_word == "eos" or len(decoded_sentence) > max_target_length:
          stoping_condn = True

      target_sequence = np.zeros((1, 1))
      target_sequence[0, 0] = word_index
      encoder_h, encoder_c = dec_h, dec_c
    return decoded_sentence

@app.route('/predict',methods=['POST'])
def predict_hashtag():
    tweet=request.json.get('tweet')
    input_text_data = input_tokenizer.texts_to_sequences([tweet])
    input_text_data = pad_sequences(input_text_data, maxlen=max_input_length, padding='post')
    tag = decode_sequence(input_text_data.reshape(1, input_text_data.shape[1])).replace('eos', '')
    hashtags=["#" + label_value[i] for i in word_tokenize(tag)]
    return jsonify({'message':'Success','payload':hashtags})

if __name__=='__main__':
    app.run(debug=True)

