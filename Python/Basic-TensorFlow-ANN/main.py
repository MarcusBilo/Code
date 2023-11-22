import os
import warnings
from tensorflow import keras
from keras.layers import Dense, Dropout, LeakyReLU
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint, EarlyStopping


def create_model():
    seq_model = keras.Sequential([
        Dense(512, activation=LeakyReLU(alpha=0.1), input_shape=(784,)),
        Dropout(0.25),
        Dense(10)
    ])

    seq_model.compile(optimizer='adam',
                      loss=SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[SparseCategoricalAccuracy()])

    return seq_model


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore", module='tensorflow')

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]
    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

    model = create_model()
    model.summary()

    checkpoint_path = "training_1/cp.ckpt"

    checkpoint_cb = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,
                                    save_best_only=True, verbose=0)

    early_stop_cb = EarlyStopping(monitor='val_loss', patience=10,
                                  mode='auto', start_from_epoch=0)

    model.fit(train_images, train_labels, epochs=10,
              validation_data=(test_images, test_labels),
              callbacks=[checkpoint_cb, early_stop_cb], verbose=0)

    os.listdir(os.path.dirname(checkpoint_path))

    model = create_model()
    print("\n")
    _, acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

    model.load_weights(checkpoint_path).expect_partial()
    _, acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


if __name__ == "__main__":
    main()
