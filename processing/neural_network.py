from processing.data_import import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf


def recognize(nr, iteracja):

    # Import dataset
    train_answer, train_images, test_answer, test_images = ex(nr)
    (x_train, y_train) = (train_images, train_answer)
    (x_test, y_test) = (test_images, test_answer)

    # One Hot Encoding
    y_train_encoded = to_categorical(y_train)

    # Unrolling N-dimensional Arrays to Vectors
    x_train_reshaped = np.reshape(x_train, (700, 640)) # 700 - number of training elements (35 * 20), 640 - size of the images (32x20)
    x_test_reshaped = np.reshape(x_test, (nr, 640)) # nr - number of testing elements

    # Data Normalization
    x_mean = np.mean(x_train_reshaped)
    x_std = np.std(x_train_reshaped)

    epsilon = 1e-10

    x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
    x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)

    if iteracja == 0:
        # Creating the Model
        model = Sequential([
            Dense(600, activation='relu', input_shape=(640,)),  # input layer
            Dense(300, activation='relu'),  # three hidden layers
            Dense(150, activation='relu'),
            Dense(75, activation='relu'),
            Dense(35, activation='softmax')  # output layer
        ])

        # Compiling the Model
        model.compile(
            optimizer='sgd',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()

        # Training the model with epochs
        model.fit(x_train_norm, y_train_encoded, epochs=50)

        model.save('recognize-letters.model')

    else:
        model = tf.keras.models.load_model('recognize-letters.model')


    # Predictions on Test Set
    preds = model.predict(x_test_norm)

    # Translate numbers to signs
    pattern_transl = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                      'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # Save the predictions in list
    l = []
    start_index = 0
    for i in range(nr):
        pred = np.argmax(preds[start_index + i])
        l.append(pattern_transl[pred])

    return l
