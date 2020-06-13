from modify_picture import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def recognize(nr):
    train_answer, train_images, test_answer, test_images = ex(nr)
    (x_train, y_train) = (train_images, train_answer)
    (x_test, y_test) = (test_images, test_answer)

    y_train_encoded = to_categorical(y_train)
    # y_test_encoded = to_categorical(y_test)

    x_train_reshaped = np.reshape(x_train, (350, 640))
    x_test_reshaped = np.reshape(x_test, (nr , 640))

    x_mean = np.mean(x_train_reshaped)
    x_std = np.std(x_train_reshaped)

    epsilon = 1e-10

    x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
    x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)

    model = Sequential([
        Dense(500, activation='relu', input_shape=(640,)),
        Dense(150, activation='relu'),
        Dense(80, activation='relu'),
        Dense(40, activation='relu'),
        Dense(35, activation='softmax')
    ])

    model.compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    model.fit(x_train_norm, y_train_encoded, epochs=40)
    preds = model.predict(x_test_norm)

    # translate numbers to signs
    pattern_transl = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                      'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    l = []
    start_index = 0
    for i in range(nr):
        pred = np.argmax(preds[start_index + i])
        l.append(pattern_transl[pred])

    print("recognized: ", l)
    return l
