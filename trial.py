import numpy as np
import cv2

def ex():
    train_images =[]
    train_answer = []
    test_images =[]
    test_answer =[]
    pattern = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
    # pattern2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    #             'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    for j in range(35):
        for i in range(10):
            train_answer.append(pattern[j])
            train_image = cv2.imread("letters/" + str(j) + "_" + str(i) + ".jpg", 0)
            train_images.append(train_image)

    # for j in range(35):
    #     for i in range(5):
    #         test_answer.append(pattern[j])
    #         test_image = cv2.imread("/home/aleksandra/Desktop/my_letters/test/" + str(j) + "_" + str(i) + ".jpg", 0)
    #         test_images.append(test_image)
    for j in range(9):
        test_answer.append(pattern[j])
        test_image = cv2.imread("Plate/roi" + str(j) + ".jpg", 0)
        test_images.append(test_image)

    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_answer = np.array(train_answer)
    test_answer = np.array(test_answer)
    return train_answer, train_images, test_answer, test_images
