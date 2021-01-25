from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding

import numpy as np



docs = ['너무 재밌네요', '최고예요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다', \
    '한번 더 보고 싶네요', '글쎄요', '별로예요', '생각보다 지루하네요', '연기가 어색해요', '재미없어요']

docs_class = np.array([1,1,1,1,1,0,0,0,0,0])



token = Tokenizer()
token.fit_on_texts(docs)



x = token.texts_to_sequences(docs)

padding_x = pad_sequences(x, 4)



word_size = len(token.word_index) + 1
model = Sequential()
model.add(Embedding(word_size, 8, input_length=4))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(padding_x, docs_class, epochs=20)

print("\n Accuracy: %.4f" % (model.evaluate(padding_x, docs_class)[1]))