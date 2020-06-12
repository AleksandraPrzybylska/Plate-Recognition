import cv2
import numpy as np
import imutils
from imutils import contours
from skimage.filters import threshold_local

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
    i = 0
    my_roi = []
    for contour in cnts:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(
                contour) >= 0.0:  # and cv2.contourArea(contour) <= 100.0 or cv2.contourArea(contour) >= 100.0:
            if w / h >= 1.0 or h/w >= 5.0:
                continue
            else:
                # if w < 15 or h < 35:  # if it finds smaller parts, just ignore them
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

def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    # TODO: add image processing here
    # Resize the image - change width to 500
    image = imutils.resize(image, width=min(500, len(image[0])))
    # image = imutils.resize(image, width=min(700, len(image[0])))
    img_copy = np.copy(image)
    # Display the original image
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)

    # RGB to Gray scale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise removal with iterative bilateral filter(removes noise while preserving edges)
    bilateral = cv2.bilateralFilter(gray, 11, 20, 20)

    # Find Edges of the grayscale image
    edged = cv2.Canny(bilateral, 50, 200)

    # Find contours based on Edges
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Create copy of original image to draw all contours
    img1 = image.copy()
    cv2.drawContours(img1, cnts, -1, (0, 255, 0), 3)
    # cv2.imshow("4- All Contours", img1)
    # cv2.waitKey(0)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    NumberPlateCnt = np.zeros((4, 1, 2))

    idx = 7
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # print ("approx = ",approx)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx  # This is our approx Number Plate Contour

            # Crop those contours and store it in Cropped Images folder
            x, y, w, h = cv2.boundingRect(c)  # This will find out co-ord for plate
            new_img = image[y:y + h, x:x + w]  # Create new image
            print(new_img.shape)
            if new_img.shape[0] < 60 or new_img.shape[1] < 100 or new_img.shape[1] >= 500:
                NumberPlateCnt = np.zeros((4, 1, 2))
                print("zle wymiary")
            else:
                idx += 1
                # cv2.imwrite("/home/aleksandra/Desktop/SW_PROJECT/ROI/wycinek_tablicy.jpg", new_img)

            break

    if np.sum(NumberPlateCnt):
        warped = four_point_transform(image, NumberPlateCnt.reshape(4, 2))
        # cv2.imshow("warped", warped)

        number = NumberPlateCnt.reshape(4,2)
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
        # cv2.imshow("Final Image With Number Plate Detected", image)
        # cv2.waitKey(0)

        # cv2.imshow("roi", new_img)
        # cv2.waitKey(0)
        # new_img = imutils.resize(new_img, width=min(500, len(image[0])))
        new_copy = np.copy(new_img)
        gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        # gray_img = cv2.bilateralFilter(gray_img, 11, 35, 35)
        gray_img = cv2.bilateralFilter(gray_img, 10,  28, 28)
        canny = cv2.Canny(gray_img, 30, 200)

        cnts = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
        img_copy = new_copy
                                
    else:
        print("not found")
        # image = imutils.resize(image, width=min(700, len(image[0])))
        gray_2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.bilateralFilter(gray_2, 10, 28, 28 )
        # gray_2 = cv2.bilateralFilter(gray_2, 11, 20, 20)
        edged_2 = cv2.Canny(gray_2, 25, 285)
        # edged_2 = cv2.Canny(gray_2, 60, 300)
        cnts = cv2.findContours(edged_2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        (cnts, _) = contours.sort_contours(cnts, method="left-to-right")

    # i = 0
    # my_roi = []
    # for contour in cnts:
    #     rect = cv2.boundingRect(contour)
    #     (x, y, w, h) = cv2.boundingRect(contour)
    #
    #     if cv2.contourArea(contour) >= 0.0:#  and cv2.contourArea(contour) <= 100.0 or cv2.contourArea(contour) >= 100.0:
    #         if w / h >= 1.0:
    #             continue
    #         else:
    #             # if w < 15 or h < 35:  # if it finds smaller parts, just ignore them
    #             if w <= 10 or h < 34 or h > 100:
    #                 continue
    #             else:
    #
    #                 crop_image = img_copy[y:y + h, x:x + w]
    #                 red = crop_image[:, :, 2]
    #                 green = crop_image[:, :, 1]
    #                 blue = crop_image[:, :, 0]
    #
    #                 if not (len(my_roi)):
    #
    #                     my_roi.append(crop_image)
    #
    #                     # cv2.rectangle(img_copy, (x, y), (x + w, y + h), (70, 0, 70), 3)  # drawing rectangle
    #                     # cv2.imshow("img_cpy", img_copy)
    #                     # print("rozmiar: w:", w, "h:", h)
    #                     # cv2.waitKey(0)
    #                     prev_w = w
    #                     prev_h = h
    #                     prev_x = x
    #                     prev_y = y
    #
    #                     cv2.rectangle(img_copy, (x, y), (x + w, y + h), (70, 0, 70), 3)  # drawing rectangle
    #                     cv2.imshow("img_cpy", img_copy)
    #                     print("rozmiar: w:", w, "h:", h)
    #                     cv2.waitKey(0)
    #
    #                 else:
    #                     acc_M = cv2.moments(contour)
    #
    #                     if acc_M["m00"] != 0.0:
    #                         acc_cX = int(acc_M["m10"] / acc_M["m00"])
    #                         acc_cY = int(acc_M["m01"] / acc_M["m00"])
    #                     else:
    #                         acc_cX = 0
    #                         acc_cY = 0
    #
    #                     if prev_x <= acc_cX <= prev_x + prev_w and prev_y <= acc_cY <= prev_y + prev_h:
    #                         continue
    #                     else:
    #
    #                         my_roi.append(crop_image)
    #                         prev_w = w
    #                         prev_h = h
    #                         prev_x = x
    #                         prev_y = y
    #
    #                         cv2.rectangle(img_copy, (x, y), (x + w, y + h), (70, 0, 70), 3)  # drawing rectangle
    #                         cv2.imshow("img_cpy", img_copy)
    #                         print("rozmiar: w:", w, "h:", h)
    #                         cv2.waitKey(0)

    # dim = (20, 32)
    # dim = (40, 70)

    my_roi = loop_find_plate(cnts, img_copy)
    # kernel = np.ones((3, 3), np.uint8)
    # for i in range(len(my_roi)):
    #     # my_roi[i] = cv2.resize(my_roi[i], dim)
    #     gray = cv2.cvtColor(my_roi[i], cv2.COLOR_BGR2GRAY)
    #     thresh1 = cv2.threshold(gray, 125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    #     my_roi[i] = thresh1

    if len(my_roi) == 7:
        for i in range(len(my_roi)):
            gray = cv2.cvtColor(my_roi[i], cv2.COLOR_BGR2GRAY)
            thresh1 = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            cv2.imwrite("/home/aleksandra/Desktop/SW_PROJECT/ROI/roi" + str(i) + ".jpg", thresh1)
    else:
        gray_3 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_3 = cv2.bilateralFilter(gray_3, 10, 28, 28)
        edged_3 = cv2.Canny(gray_3, 200, 545)
        cnts = cv2.findContours(edged_3.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        (cnts, _) = contours.sort_contours(cnts, method="le ft-to-right")
        new_roi = loop_find_plate(cnts, img_copy)

        if len(new_roi) == 7:
            for i in range(len(new_roi)):
                gray = cv2.cvtColor(new_roi[i], cv2.COLOR_BGR2GRAY)
                thresh1 = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                cv2.imwrite("/home/aleksandra/Desktop/SW_PROJECT/ROI/roi" + str(i) + ".jpg", thresh1)

        else:

            gray_4 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_4 = cv2.bilateralFilter(gray_4, 10, 28, 28)
            edged_4 = cv2.Canny(gray_4, 200, 600)
            cnts = cv2.findContours(edged_4.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            (cnts, _) = contours.sort_contours(cnts, method="le ft-to-right")
            next_roi = loop_find_plate(cnts, img_copy)

            for i in range(len(next_roi)):
                gray = cv2.cvtColor(next_roi[i], cv2.COLOR_BGR2GRAY)
                thresh1 = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                cv2.imwrite("/home/aleksandra/Desktop/SW_PROJECT/ROI/roi" + str(i) + ".jpg", thresh1)




    cv2.destroyAllWindows()

    return 'PO12345'
