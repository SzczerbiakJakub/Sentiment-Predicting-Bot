import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from keras import layers




version = "1.2.4"


def labels_to_one_hot(labels, dimension=3):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results



twt_preproccesed_df = pd.read_csv(r".\datasets\PREPROCESSED_Twitter_Data.csv", encoding="utf-8")
twt_preproccesed_df = twt_preproccesed_df.sample(frac=1, random_state=42)
print(twt_preproccesed_df)
texts = twt_preproccesed_df.clean_text.values.tolist()
labels = twt_preproccesed_df.sentiment_id.values.tolist()

train_validation_range = int(len(texts) * 0.8)
train_range = int(train_validation_range * 0.8)
validation_range = train_validation_range - train_range
test_range = len(texts) - train_validation_range

train_text, train_labels = np.array(texts[:train_range], dtype=str), labels[:train_range],

validation_text, validation_labels = np.array(texts[train_range:train_validation_range], dtype=str), labels[train_range:train_validation_range],

test_text, test_labels = np.array(texts[train_validation_range:], dtype=str), labels[train_validation_range:],


    #   TRAIN AND VALIDATION DATA:

train_text_in_np = train_text
validation_text_in_np = validation_text

validation_labels_in_np = validation_labels
validation_labels_in_np = labels_to_one_hot(validation_labels_in_np)
train_labels_in_np = train_labels
train_labels_in_np = labels_to_one_hot(train_labels_in_np)

print(train_labels_in_np[-5])
print(train_labels_in_np[-4])
print(train_labels_in_np[-3])
print(train_labels_in_np[-2])
print(train_labels_in_np[-1])

print(len(train_text_in_np))
print(len(train_labels_in_np))
train_text_raw = tf.data.Dataset.from_tensor_slices(train_text_in_np)
train_labels_raw = tf.data.Dataset.from_tensor_slices(train_labels_in_np)
print(train_text_raw)
print(train_labels_raw)


print(len(validation_text_in_np))
print(len(validation_labels_in_np))
validation_text_raw = tf.data.Dataset.from_tensor_slices(validation_text_in_np)
validation_labels_raw = tf.data.Dataset.from_tensor_slices(validation_labels_in_np)
print(validation_text_raw)
print(validation_labels_raw)

    #   TEST DATA:

test_text_in_np = test_text
print(len(test_text_in_np))
test_labels_in_np = test_labels
test_labels_in_np = labels_to_one_hot(test_labels_in_np)
print(len(test_labels_in_np))

test_text_raw = tf.data.Dataset.from_tensor_slices(test_text_in_np)
test_labels_raw = tf.data.Dataset.from_tensor_slices(test_labels_in_np)
print(test_text_raw)
print(test_labels_raw)


#   LOAD VECTORIZED TEXT LAYER AND BUILD DICTIONARY:
    #   LAST VER: vocab_size == 10000
vocab_size = 10000      #   SIZE OF THIS DATASET"S VOCABULARY
max_text_size = 25      #   SIZE OF SINGLE TENSOR

    #   SET TEXTVECTORIZATION LAYER

vectorization_layer = layers.TextVectorization(
            ngrams=2,
            standardize="lower_and_strip_punctuation",
            max_tokens = vocab_size+2,
            output_mode="int",
            output_sequence_length=max_text_size
)

    #   ADAPT VOCAB INTO TRAINING DATA (AND NOT VALIDATION), DATA MIGHT BE IN NP LIST

vectorization_layer.adapt(train_text_in_np)

vectorized_layer_model = tf.keras.models.Sequential()
vectorized_layer_model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
vectorized_layer_model.add(vectorization_layer)
vectorized_layer_model.summary()

vocab = vectorization_layer.get_vocabulary()

#   SAVE TEXT VECTORIZATION MODEL TO USE IT LATER
vectorized_layer_model.save('models/text_vectorization_model_' + version + '.h5')
with open('models/vocab_' + version + '.txt', 'w', encoding='utf-8') as f:
    for token in vocab:
        f.write(token + '\n')

#   TRANSFORM DATASETS TO MODELS INPUTS

    #   TEXT DATASETS INTO MODELS INPUTS FUNCTION:
def transform_text_into_model_input(text):
    text = tf.expand_dims(text, -1)
    return tf.squeeze(vectorization_layer(text))

    #   LABELS DATASETS INTO MODELS INPUTS FUNCTION:
def transform_label_into_model_input(label):
    label = tf.expand_dims(label, -1)
    return label

    #   GET TF INPUTS:
        #   TEXT:
train_text_model_input = train_text_raw.map(transform_text_into_model_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)
validation_text_model_input = validation_text_raw.map(transform_text_into_model_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_text_model_input = test_text_raw.map(transform_text_into_model_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #   LABELS:
train_labels_model_input = train_labels_raw
validation_labels_model_input = validation_labels_raw
test_labels_model_input = test_labels_raw

#   ZIP TENSORS INTO ONE PIECES:
    #   TRAIN INPUTS:
train_model_inputs = tf.data.Dataset.zip(
                    train_text_model_input,
                    train_labels_model_input
)

#   VALIDATION INPUTS:
validation_model_inputs = tf.data.Dataset.zip(
                    validation_text_model_input,
                    validation_labels_model_input
)

#   TEST INPUTS:
test_model_inputs = tf.data.Dataset.zip(
                    test_text_model_input,
                    test_labels_model_input
)

#   BUILD AND FIT MODEL:
    #   MODEL AND FIT PARAMETERS:

batch_size = 128
epochs = 18
learning_rate = 0.0001
optimizer_fn = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
loss_fn = keras.losses.categorical_crossentropy
AUTOTUNE = tf.data.experimental.AUTOTUNE
buffer_size = train_model_inputs.cardinality().numpy()

train_model_inputs = train_model_inputs.shuffle(buffer_size=buffer_size)\
                    .batch(batch_size=batch_size, drop_remainder=True)\
                    .prefetch(AUTOTUNE)

validation_model_inputs = validation_model_inputs.shuffle(buffer_size=buffer_size)\
                    .batch(batch_size=batch_size, drop_remainder=True)\
                    .prefetch(AUTOTUNE)

test_model_inputs = test_model_inputs.shuffle(buffer_size=buffer_size)\
                    .batch(batch_size=batch_size, drop_remainder=True)\
                    .prefetch(AUTOTUNE)

    #   MODEL BUILDING FUNCTION
def create_model():
    inputs = layers.Input(shape=(max_text_size,), dtype=tf.int32)
    #   LAST VER: embedding_layer = layers.Embedding(vocab_size, 16)
    embedding_layer = layers.Embedding(vocab_size, 32)
    x = embedding_layer(inputs)
    x = layers.Flatten()(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=["accuracy"])
    return model

def create_model_2():
    inputs = layers.Input(shape=(max_text_size,), dtype=tf.int32)
    #   LAST VER: embedding_layer = layers.Embedding(vocab_size, 16)
    embedding_layer = layers.Embedding(vocab_size, 64)
    x = embedding_layer(inputs)
    x = layers.Bidirectional(layers.LSTM(32))(x) 
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=["accuracy"])
    return model

model = create_model_2()
model.summary()

    #   MODELS MUST BE FED WITH TENSORS, THIS HELPS TO CONVET DATASETS INTO TENSORS:
def _fixup_shape(text, labels):
    text.set_shape([batch_size, max_text_size])
    labels.set_shape([batch_size, 3])
    return text, labels

train_model_inputs = train_model_inputs.map(_fixup_shape)
validation_model_inputs = validation_model_inputs.map(_fixup_shape)
test_model_inputs = test_model_inputs.map(_fixup_shape)

    #   FITTING THE MODEL:
history = model.fit(train_model_inputs, verbose=1, epochs=epochs, validation_data=validation_model_inputs)

    #   PLOT RESULTS:
print(history.history.keys())
train_loss = history.history['loss']
validation_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

epochs_axis = [x for x in range(1, epochs+1)]

plt.plot(epochs_axis, train_loss, label='loss', color='blue', linestyle='-')  # Blue solid line
plt.plot(epochs_axis, validation_loss, label='val_loss', color='red', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.title('Losses plot')
plt.legend()

plt.show()

plt.clf()

plt.plot(epochs_axis, train_accuracy, label='accuracy', color='blue', linestyle='-')  # Blue solid line
plt.plot(epochs_axis, validation_accuracy, label='val_accuracy', color='red', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Accuracies')
plt.title('Accuracies plot')
plt.legend()

plt.show()


test_loss, test_accuracy = model.evaluate(test_model_inputs, verbose=1)

    # Print the test accuracy
print("Test Accuracy:", test_accuracy)

    #   SAVE THE MODEL
model.save('./models/negative_neutral_positive_prediction_' + version + '.h5')

with open('./models/readme.txt', 'a') as f:
    f.write(f'Version no. {version} -> Test Accuracy: {test_accuracy}\n')