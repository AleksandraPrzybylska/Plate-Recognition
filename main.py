from matplotlib import pyplot as plt
from trial import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# print('Using TensorFlow version', tf.__version__)
def recognize():
    train_answer, train_images, test_answer, test_images = ex()
    (x_train, y_train) = (train_images, train_answer)
    (x_test, y_test) = (test_images, test_answer)

    # print('x_train shape:', x_train.shape)
    # print('y_train shape:', y_train.shape)
    # print('x_test shape:', x_test.shape)
    # print('y_test shape:', y_test.shape)

    # plt.imshow(x_train[0], cmap='gray')
    # plt.show()

    # y_train[0]
    # print(set(train_answer))

    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)

    # print('y_train_encoded shape:', y_train_encoded.shape)
    # print('y_test_encoded shape', y_test_encoded.shape)
    #
    # y_train_encoded[0]

    x_train_reshaped = np.reshape(x_train, (350, 640))
    x_test_reshaped = np.reshape(x_test, (9, 640))

    # print('x_train_reshaped shape:', x_train_reshaped.shape)
    # print('x_test_reshaped shape:', x_test_reshaped.shape)

    # print(set(x_train_reshaped[0]))

    x_mean = np.mean(x_train_reshaped)
    x_std = np.std(x_train_reshaped)

    epsilon = 1e-10

    x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
    x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)

    #print(set(x_train_norm[0]))

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

    # _, accuracy = model.evaluate(x_test_norm, y_test_encoded)
    # print('Test set accuracy:', accuracy * 100)

    preds = model.predict(x_test_norm)
    # print('Shape of preds:', preds.shape)

    # translate numbers to signs
    pattern_transl = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                      'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    # plot detected
    # plt.figure(figsize=(12, 12))
    l = []
    start_index = 0
    for i in range(9):
        plt.subplot(1, 9, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        pred = np.argmax(preds[start_index + i])
        # gt = y_test[start_index + i]

        col = 'g'
        # if pred != gt:
        #     col = 'r'
        l.append(pattern_transl[pred])

        #plt.xlabel('i={}, pred {}, gt {}'.format(start_index + i, pattern_transl[pred], pattern_transl[gt]), color=col)
    #     plt.xlabel('i={}, pred {}'.format(start_index + i, pattern_transl[pred]), color='b')
    #     plt.imshow(x_test[start_index + i], cmap='gray')
    # plt.show()
    print("recognized: ", l)
    return l
