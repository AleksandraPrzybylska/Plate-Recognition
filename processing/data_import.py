import numpy as np
import cv2

def ex(nr):
    # Initialize variables to store our images and photo database for neural network
    train_images =[]
    train_answer = []
    test_images =[]
    test_answer =[]

    # Number of letters
    pattern = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

    # Add train data
    for j in range(35):
        for i in range(20):
            train_answer.append(pattern[j])
            train_image = cv2.imread("dane/baza/" + str(j) + "_" + str(i) + ".jpg", 0)
            train_images.append(train_image)

    # Add numbers cut off car plate
    for j in range(nr):
        test_answer.append(pattern[j])
        test_image = cv2.imread("dane/plate_roi/roi" + str(j) + ".jpg", 0)
        test_images.append(test_image)

    # Modify images to array
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_answer = np.array(train_answer)
    test_answer = np.array(test_answer)
    return train_answer, train_images, test_answer, test_images
