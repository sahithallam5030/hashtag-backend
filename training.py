
import numpy as np
import pandas as pd
import pickle
import nltk
from tensorflow.keras.models import Model
from statistics import mode
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, Attention
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras import backend as K
nltk.download('wordnet')
nltk.download('punkt')
# Read the dataset file
train = pd.read_csv("train.csv")
input_data = train["tweet"]
target_data = train.iloc[:, 4:].values
# Get the column names of the target
target_label = train.iloc[:, 4:].columns.tolist()
target_label_val = ["I can't tell", "Negative", "Neutral", "Positive",
                "Tweet not related to weather condition","current (same day) weather", "future (forecast)",
               "I can't tell", "past weather",
               "clouds", "cold", "dry", "hot", "humid", "hurricane",
               "I can't tell", "ice", "other", "rain", "snow", "storms",
               "sun", "tornado", "wind"]

input_textdata = []
target_textdata = []
input_words = []
target_words = []
contractions = pickle.load(open("contractions.pkl", "rb"))['contractions']

# Clean the tweets
def clean(tweet):
    tweet = tweet.replace(":", "").lower()
    words = list(filter(lambda w: (w.isalpha()), tweet.split(" ")))
    words = [contractions[w] if w in contractions else w for w in words]
    return words

# Iterate over input data
for tweet in input_data:
    inpt_words = clean(tweet)
    input_textdata += [' '.join(inpt_words)]
    input_words += inpt_words

# Iterate over target data
for label in target_data:
    sentiment = target_label[np.argmax(label[:5])]
    when = target_label[np.argmax(label[5:9]) + 5]
    kind = [target_label[ind] for ind, ele in enumerate(label[9:len(label)], 9) if ele >= 0.5]
    target_textdata += ["sos " + " ".join([sentiment] + [when] + kind) + " eos"]
    
# Only store unique words from the input and target word lists
input_words = sorted(list(set(input_words)))
input_word_count = len(input_words)
target_word_count = len(target_label) + 2

# Get the length of the input and the target texts which appear most frequently
max_input_length = mode([len(i) for i in input_textdata])
max_target_length = mode([len(i) for i in target_textdata])

print("number of input words : ", input_word_count)
print("number of target words : ", target_word_count)
print("maximum input length : ", max_input_length)
print("maximum target length : ", max_target_length)

# Split the input and target text into a 90:10 ratio or testing size of 10%.
x_train, x_test, y_train, y_test = train_test_split(input_textdata, target_textdata, test_size=0.1, random_state=42)

input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(x_train)
with open("input_tokenizer.pkl", "wb") as file:
    pickle.dump(input_tokenizer, file)
target_tokenizer = Tokenizer()
target_tokenizer.fit_on_texts(y_train)
with open("target_tokenizer.pkl", "wb") as file:
    pickle.dump(target_tokenizer, file)

x_train = input_tokenizer.texts_to_sequences(x_train)
y_train = target_tokenizer.texts_to_sequences(y_train)

encode_input_data = pad_sequences(x_train, maxlen=max_input_length, padding='post', dtype="float32")
decoded_data = pad_sequences(y_train, maxlen=max_target_length, padding='post', dtype="float32")
decoded_input_data = decoded_data[:, :-1]
decoded_target_data = decoded_data.reshape(len(decoded_data), max_target_length, 1)[:, 1:]

K.clear_session()

latent_dim = 500

encoder_inputs = Input(shape=(max_input_length,))
encoder_embedding = Embedding(input_word_count + 1, latent_dim)(encoder_inputs)
# Create 3 stacked LSTM layers
# 1st LSTM layer keeps only output
encoder_lstm1 = LSTM(latent_dim, return_state=True, return_sequences=True)
encoder_output1, *_ = encoder_lstm1(encoder_embedding)

# 2nd LSTM layer keeps only output
encoder_lstm2 = LSTM(latent_dim, return_state=True, return_sequences=True)
encoder_output2, *_ = encoder_lstm2(encoder_output1)

# 3rd LSTM layer keeps output as well as its states
encoder_lstm3 = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_output3, state_h3, state_c3 = encoder_lstm3(encoder_output2)

# Encoder states
encoder_states = [state_h3, state_c3]
# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embed_layer = Embedding(target_word_count + 1, latent_dim)
decoder_embedding = decoder_embed_layer(decoder_inputs)

# Initialize the LSTM layer of the decoder with the encoder's output states
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, *_ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

# Attention layer
attention = Attention()
attention_output = attention([decoder_outputs, encoder_output3])
# Merge the attention output with the decoder outputs
merge = Concatenate(axis=-1, name='concat_layer1')([decoder_outputs, attention_output])

# Fully connected Dense layer for the output
decoder_dense_layer = Dense(target_word_count + 1, activation='softmax')
decoder_outputs = decoder_dense_layer(merge)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# Train the model with input and target data from encoder and decoder
model.fit([encode_input_data, decoded_input_data], decoded_target_data,batch_size=500, epochs=9)
model.save("model.h5")
