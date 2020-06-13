import imutils
from imutils import contours
# from trial import *
from main import *

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def loop_find_plate(cnts, img_copy):
    # i = 0
    my_roi = []
    for contour in cnts:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(
                contour) >= 0.0:
            if w / h >= 1.0 or h/w >= 5.0:
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
                        # cv2.rectangle(img_copy, (x, y), (x + w, y + h), (70, 0, 70), 3)  # drawing rectangle
                        # cv2.imshow("img_cpy", img_copy)
                        # print("rozmiar: w:", w, "h:", h)
                        # cv2.waitKey(0)

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

                            # cv2.rectangle(img_copy, (x, y), (x + w, y + h), (70, 0, 70), 3)  # drawing rectangle
                            # cv2.imshow("img_cpy", img_copy)
                            # print("rozmiar: w:", w, "h:", h)
                            # cv2.waitKey(0)
    return my_roi

def get_contours(image, bil1, bil2, bil3, canny1, canny2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, bil1, bil2, bil3)
    edged = cv2.Canny(bilateral, canny1, canny2)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if len(cnts) == 0:
        print("zero konturów")
    return cnts

def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    # TODO: add image processing here

    image = imutils.resize(image, width=min(500, len(image[0])))
    img_copy = np.copy(image)
    # cv2.imshow("Original Image", image)
    # cv2.waitKey(0)

    cnts = get_contours(image, 11, 20, 20, 50, 200)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    NumberPlateCnt = np.zeros((4, 1, 2))

    idx = 8
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

    if np.sum(NumberPlateCnt):
        warped = four_point_transform(image, NumberPlateCnt.reshape(4, 2))

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

        new_copy = np.copy(new_img)
        cnts = get_contours(new_img, 10, 28, 28, 24, 200)
        if len(cnts):
            (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
        img_copy = new_copy

    else:
        cnts = get_contours(image, 10, 28, 28, 25, 285)
        if len(cnts):
            (cnts, _) = contours.sort_contours(cnts, method="left-to-right")

    letters = []
    my_roi = loop_find_plate(cnts, img_copy)
    letters = my_roi
    if len(my_roi) == 7:
        letters = my_roi
    else:
        cnts = get_contours(image, 10, 28, 28, 200, 545)
        if len(cnts):
            (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
        new_roi = loop_find_plate(cnts, img_copy)

        if len(new_roi) == 7:
            letters = new_roi
        else:
            cnts = get_contours(image, 10, 28, 28, 60, 600)
            if len(cnts):
                (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
            next_roi = loop_find_plate(cnts, img_copy)

            letters = []

            for i in range(len(next_roi)):
                if next_roi[i].shape[1] == 0:
                    break
                x = next_roi[i].shape[0] / next_roi[i].shape[1]
                if x >= 4.0 or x <= 1.24:
                    continue
                else:
                    letters.append(next_roi[i])
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

    result = recognize()
    return result
    # return 'PO12345'
