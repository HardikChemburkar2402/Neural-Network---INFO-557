"""
The main code for the recurrent and convolutional networks assignment.
See README.md for details.
"""

from typing import Tuple, List, Dict
import keras


def create_toy_rnn(input_shape: tuple, n_outputs: int) -> Tuple[keras.Sequential, Dict]:
    """Creates a GRU-based RNN for a toy sequence-to-sequence regression problem."""
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.GRU(128, return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.GRU(64, return_sequences=True))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(n_outputs, activation='linear'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0004), loss='mse')

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=8, restore_best_weights=True
    )

    return model, {
        "batch_size": 2,
        "epochs": 60,
        "callbacks": [early_stop]
    }


def create_mnist_cnn(input_shape: tuple, n_outputs: int) -> Tuple[keras.Sequential, Dict]:
    """Creates a CNN for MNIST digit classification."""
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(
        keras.layers.Conv2D(64, (3, 3), activation='relu')
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(
        keras.layers.Dense(n_outputs, activation='softmax')
    )

    model.compile(optimizer=keras.optimizers.Adam(),
                   loss='categorical_crossentropy', metrics=['accuracy'])

    return model, {"batch_size": 32}


def create_youtube_comment_rnn(vocabulary: List[str], n_outputs: int) -> Tuple[
    keras.Sequential, Dict]:
    """Creates a BiGRU-based RNN for spam classification on YouTube comments."""
    vocab_size = len(vocabulary)
    embedding_dim = 64

    model = keras.Sequential()
    model.add(keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(keras.layers.Bidirectional(keras.layers.GRU(units=64, return_sequences=True)))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(
        keras.layers.Dense(
            n_outputs, activation='softmax' if n_outputs > 1 else 'sigmoid'
        )
    )

    loss_fn = 'sparse_categorical_crossentropy' if n_outputs > 1 else 'binary_crossentropy'
    model.compile(optimizer=keras.optimizers.Adam(), loss=loss_fn, metrics=['accuracy'])

    return model, {"batch_size": 32}


def create_youtube_comment_cnn(vocabulary: List[str], n_outputs: int) -> Tuple[
    keras.Sequential, Dict]:
    """Creates a Conv1D-based CNN for spam classification on YouTube comments."""
    vocab_size = len(vocabulary)
    embedding_dim = 64

    model = keras.Sequential()
    model.add(keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(
        keras.layers.Dense(
            n_outputs, activation='softmax' if n_outputs > 1 else 'sigmoid'
        )
    )

    loss_fn = 'sparse_categorical_crossentropy' if n_outputs > 1 else 'binary_crossentropy'
    model.compile(optimizer=keras.optimizers.Adam(), loss=loss_fn, metrics=['accuracy'])

    return model, {"batch_size": 32}
