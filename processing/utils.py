import imutils
from imutils import contours
from neural_network import *


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def loop_find_number(cnts, img_copy):

    my_roi = []
    for contour in cnts:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(
                contour) >= 0.0:
            if w / h >= 1.0 or h / w >= 5.0:
                continue
            else:
                if w <= 10 or h < 34 or w > 73:
                    continue
                else:

                    crop_image = img_copy[y:y + h, x:x + w]

                    if not (len(my_roi)):

                        my_roi.append(crop_image)
                        prev_w = w
                        prev_h = h
                        prev_x = x
                        prev_y = y

                    else:

                        acc_M = cv2.moments(contour)

                        if acc_M["m00"] != 0.0:
                            acc_cX = int(acc_M["m10"] / acc_M["m00"])
                            acc_cY = int(acc_M["m01"] / acc_M["m00"])
                        else:
                            acc_cX = 0
                            acc_cY = 0

                        if prev_x <= acc_cX <= prev_x + prev_w and prev_y <= acc_cY <= prev_y + prev_h:
                            continue
                        else:
                            my_roi.append(crop_image)
                            prev_w = w
                            prev_h = h
                            prev_x = x
                            prev_y = y
    return my_roi


def find_contours(image, bil1, bil2, bil3, canny1, canny2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, bil1, bil2, bil3)
    edged = cv2.Canny(bilateral, canny1, canny2)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts


def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')

    # Resize the image
    image = imutils.resize(image, width=min(500, len(image[0])))
    img_copy = np.copy(image)

    # Find contours of the image
    cnts = find_contours(image, 11, 20, 20, 50, 200)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    NumberPlateCnt = np.zeros((4, 1, 2))

    # Go through loop and find the plate on the image
    idx = 7
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            x, y, w, h = cv2.boundingRect(c)
            new_img = image[y:y + h, x:x + w]
            if new_img.shape[0] < 60 or new_img.shape[1] < 100 or new_img.shape[1] >= 500:
                NumberPlateCnt = np.zeros((4, 1, 2))
            else:
                idx += 1
            break

    #   When you find the Number Plate, straighten it and find the letters
    if np.sum(NumberPlateCnt):
        # Straighten the picture
        warped = four_point_transform(image, NumberPlateCnt.reshape(4, 2))

        # If it finds four different peaks, then straighten the image
        number = NumberPlateCnt.reshape(4, 2)
        arr = []
        check = False
        a = [number[0][0], number[0][1]]
        b = [number[1][0], number[1][1]]
        c = [number[2][0], number[2][1]]
        d = [number[3][0], number[3][1]]

        arr.append(a)
        arr.append(b)
        arr.append(c)
        arr.append(d)

        for elem in arr:
            if arr.count(elem) > 1:
                check = True
        if not check:
            new_img = warped
        else:
            new_img = new_img

        cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)

        # Find contours on the plate to identify the letters
        new_copy = np.copy(new_img)
        cnts = find_contours(new_img, 10, 28, 28, 24, 200)
        if len(cnts):
            (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
        img_copy = new_copy

    else:
        cnts = find_contours(image, 10, 28, 28, 25, 285)
        if len(cnts):
            (cnts, _) = contours.sort_contours(cnts, method="left-to-right")

    # If it finds letter, then save it, else filter the picture and find contours
    my_roi = loop_find_number(cnts, img_copy)
    if len(my_roi) == 7:
        letters = my_roi
    else:
        cnts = find_contours(image, 10, 28, 28, 200, 545)
        if len(cnts):
            (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
        new_roi = loop_find_number(cnts, img_copy)

        if len(new_roi) == 7:
            letters = new_roi
        else:
            cnts = find_contours(image, 10, 28, 28, 60, 600)
            if len(cnts):
                (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
            next_roi = loop_find_number(cnts, img_copy)

            letters = []

            for i in range(len(next_roi)):
                if next_roi[i].shape[1] == 0:
                    break
                x = next_roi[i].shape[0] / next_roi[i].shape[1]
                if x >= 4.0 or x <= 1.24:
                    continue
                else:
                    letters.append(next_roi[i])

    # Resize the letters with specific dimensions
    if len(letters):
        dim = (20, 32)
        for i in range(len(letters)):
            try:
                letters[i] = cv2.resize(letters[i], dim)
                gray = cv2.cvtColor(letters[i], cv2.COLOR_BGR2GRAY)
                thresh1 = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                cv2.imwrite("Plate/roi" + str(i) + ".jpg", thresh1)
            except Exception as e:
                error = True

    cv2.destroyAllWindows()
    nr = len(letters)
    result = recognize(nr)
    return result
    # return 'PO12345'
