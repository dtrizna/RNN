# About

Repository contains PoC code for various TensorFlow classes and methods, which Recurrent Neural Network training is based on.

# NLP

## A_...

### Preprocessing
```
tf.keras.preprcessing.text.Tokenizer()
tf.keras.preprocessing.sequence.pad_sequences()
```


## B_...

### Tensorflow built-in Datasets
```
tsdf.load("imdb_reviews", as_supervised=True, with_info=True)
```
### Embedding and its output visualization
```
tf.keras.layers.Embedding()
http://projector.tensorflow.org/
```

### Usage of pretrained Embeddings for english words (glove.6B.100d.txt)
```
tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length,
                              weights=embeddings_matrix], trainable=False)

```

## C_...

### Various RNN model examples
```
model5 = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMB_DIM, input_length=120),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
```

## D_...

### Sequence predictions
- Text pre-processing - building sequences and labels
```
# One Hot encode labels
from tensorflow.keras.utils import to_categorical
y = to_categorical(labels, num_classes=vocab_size+1)
```
- Predicted text generation
```
word_nr = model.predict_classes(padded_seed, verbose=0)
```
- Using larger RNN model, with 2 LSTM layers, Dropout and L2 regularizations
```
...
model.add(Bidirectional(LSTM(150, return_sequences = True)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
...
```