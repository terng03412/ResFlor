from preprocess import preproc, normalization
from dataGenerator import Tokenizer
import tensorflow as tf
import numpy as np
import cv2
import string
import h5py


from dataGenerator import DataGenerator
from Models.HTR_Models import FlorResAcHTR as ResFlor
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

keras = tf.keras

print(tf.__version__)


def ctc_loss(y_true, y_pred):
    """Function for computing the CTC loss"""

    if len(y_true.shape) > 2:
        y_true = tf.squeeze(y_true)

    input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
    input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)
    label_length = tf.math.count_nonzero(
        y_true, axis=-1, keepdims=True, dtype="int64")

    loss = keras.backend.ctc_batch_cost(
        y_true, y_pred, input_length, label_length)
    loss = tf.reduce_mean(loss)

    return loss


def Callback(source, model_name):
    callbacks = [
        ModelCheckpoint(
            filepath=f"target/" + str(model_name) + "/" +
            str(source) + "_checkpoint_weights.hdf5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=True
        ),
        EarlyStopping(
            monitor="val_loss",
            min_delta=1e-8,
            patience=20,
            restore_best_weights=True,
            verbose=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            min_delta=1e-8,
            factor=0.2,
            patience=15,
            verbose=True
        ),
        CSVLogger(
            filename=f"log/" + str(model_name) + "/" +
            str(source) + "_epochs.log",
            separator=";",
            append=True
        ),
    ]
    return callbacks


SPACE = " "
SPECIAL_CHARS = "?!,."
ALPHANUMERIC = string.printable[:62]
CHARS = ALPHANUMERIC + SPECIAL_CHARS + SPACE


INPUT_SOURCE_NAME = "iam_word"
BATCH_SIZE = 16
MAX_TEXT_LENGTH = 128
CHARSET_BASE = CHARS


dtgen = DataGenerator(
    source=f"/home/kuadmin01/terng/Dataset/dataset_filter.hdf5",
    batch_size=BATCH_SIZE,
    charset=CHARSET_BASE,
    max_text_length=MAX_TEXT_LENGTH,
    predict=False
)

INPUT_SHAPE = (1024, 128, 1)
OUTPUT_SHAPE = dtgen.tokenizer.vocab_size + 1

inputs, outputs = ResFlor(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.RMSprop(
    learning_rate=5e-4), loss=ctc_loss)
model.summary()

callbacks = Callback(INPUT_SOURCE_NAME, 'Flor')


EPOCHS = 100
history = model.fit(x=dtgen.next_train_batch(),
                    epochs=EPOCHS,
                    steps_per_epoch=dtgen.steps['train'],
                    validation_data=dtgen.next_valid_batch(),
                    validation_steps=dtgen.steps['valid'],
                    callbacks=callbacks,
                    shuffle=True,
                    verbose=1
                    )

model.save(f"saved_model/Flor/{INPUT_SOURCE_NAME}_filter")


# Predict
PREDICT_IMAGE_SRC = "hello.png"
tokenizer = Tokenizer(chars=CHARSET_BASE, max_text_length=MAX_TEXT_LENGTH)
img = preproc(PREDICT_IMAGE_SRC, input_size=INPUT_SHAPE)
x_test = normalization([img])

STEPS = 1

out = model.predict(
    x=x_test,
    batch_size=None,
    verbose=False,
    steps=STEPS,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False
)

steps_done = 0
batch_size = int(np.ceil(len(out) / STEPS))
input_length = len(max(out, key=len))
predicts, probabilities = [], []

while steps_done < STEPS:
    index = steps_done * batch_size
    until = index + batch_size

    x_test = np.asarray(out[index:until])
    x_test_len = np.asarray([input_length for _ in range(len(x_test))])
    decode, log = keras.backend.ctc_decode(
        x_test,
        x_test_len,
        greedy=True,
        beam_width=10,
        top_paths=10
    )
    probabilities.extend([np.exp(x) for x in log])
    decode = [[[int(p) for p in x if p != -1] for x in y] for y in decode]
    predicts.extend(np.swapaxes(decode, 0, 1))
    # update step
    steps_done += 1
    predicts = [[tokenizer.decode(x) for x in y] for y in predicts]
    print("\n####################################")
for i, (pred, prob) in enumerate(zip(predicts, probabilities)):
    print("\nProb\t\t-\t\tPredict")
    print("=======================")
    for (pd, pb) in zip(pred, prob):
        print(f"{pb * 100:.2f}%\t\t-\t\t{pd}")
print("\n####################################")
