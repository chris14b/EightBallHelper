import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2

w_mask = 300
h_mask = 300

def isGrey(hsvPixel):
    return hsvPixel[1] < 20

# ---------------   GET THE FELT COLOR  ----------------------
# take in the image then put all pixels in 1 array
def findTableHue(hsv):
    counts = np.zeros(256)
    for i in range(w_mask):
        for j in range(h_mask):
            if not isGrey(hsv[i][j]):
                counts[hsv[i][j][0]] += 1

    # get hue  window for the felt
    window_width = 10
    best = 0
    best_hue = 0
    for h in range(256):
        count = 0
        for w in range(window_width):
            count += counts[(h+w) % 256]
        if count > best:
            best = count
            best_hue = h

    # set those
    isFeltHue = [False] * 256
    for w in range(window_width):
        isFeltHue[(best_hue + w) % 256] = True

    return isFeltHue

# scales the image down, gets the mask based on felt hue, smooths it, fills it,
# scales it back to the original size, then returns it
def getMask(image):

    smaller = cv2.resize(image, (w_mask, h_mask))
    hsv = cv2.cvtColor(smaller, cv2.COLOR_BGR2HSV)

    isFeltHue = findTableHue(hsv)

    for i in range(w_mask):
        for j in range(h_mask):
            if isFeltHue[hsv[i][j][0]] and not isGrey(hsv[i][j]):
                hsv[i][j][2] = 255
            else:
                hsv[i][j][2] = 0

    mask = hsv[:,:,2]

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.dilate(mask, kernel,iterations = 1)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.erode(mask, kernel,iterations = 1)
    mask = cv2.medianBlur(mask, 3)

    mask = getLargestBlob(mask)

    # assumes only 1 convex blob to fill in, hence the blob detection
    for i in range(1, w_mask):
        bot = 0
        top = h_mask - 1
        while mask[i][bot] == 0 and top > bot:
            bot += 1
        while mask[i][top] == 0 and top > bot:
            top -= 1
        if mask[i][top] == 0: #nothing to fill in
            continue
        for j in range(bot, top+1):
            if mask[i-1][j] == 255:
                mask[i][j] = 255

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = cv2.medianBlur(mask, 3)
    return mask

def getMasked(image):
    mask = getMask(image)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked

# takes in a thresholded image, setting largest blob to having 255 values, 0 elsewhere
def getLargestBlob(mask):

    # find the blobs with just the 4 connectivity ie up and left check
    blobs = np.zeros((mask.shape[1], mask.shape[0]), np.uint16)
    curr_label = 1 # 0 is for no label
    label_equiv = [0]
    counts = [0]
    for i in range(w_mask):
        for j in range(h_mask):

            # ignore if not anything
            if mask[i][j] == 0:
                blobs[i][j] = 0
                continue

            #check above
            if i > 0 and blobs[i-1][j] != 0:
                blobs[i][j] = label_equiv[blobs[i-1][j]]

                # check if the label equivalences need updating
                if j > 0 and blobs[i][j-1] != blobs[i-1][j] and blobs[i][j-1] != 0:
                    if label_equiv[blobs[i][j-1]] > label_equiv[blobs[i-1][j]]:
                        label_equiv[blobs[i][j-1]] = label_equiv[blobs[i-1][j]]
                    if label_equiv[blobs[i-1][j]] > label_equiv[blobs[i][j-1]]:
                        label_equiv[blobs[i-1][j]] = label_equiv[blobs[i][j-1]]

            # nothing above, check left
            elif j > 0 and blobs[i][j-1] != 0:
                    blobs[i][j] = blobs[i][j-1]

            # otherwise we need to make anew label
            else:
                blobs[i][j] = curr_label
                label_equiv.append(curr_label)
                counts.append(0)
                curr_label += 1

            counts[blobs[i][j]] += 1


    # shuffle the counts down to the lowest label
    for i in range(1, len(counts)):
        temp = counts[i]
        counts[i] = 0
        counts[label_equiv[i]] += temp

    # # find the largest label
    largest_count = 0
    largest_blob = 1
    for i in range(1, len(counts)):
        if largest_count < counts[i]:
            largest_blob = i
            largest_count = counts[i]

    # go through the equivalences and only keep the label with the largest label
    for i in range(w_mask):
        for j in range(h_mask):
            if label_equiv[blobs[i][j]] == largest_blob:
                mask[i][j] = 255
            else:
                mask[i][j] = 0
    return mask

def getCorners(image):

    mask = getMask(image)
    masked = cv2.bitwise_and(image, image, mask=mask)

    # --------------- get edges ---------------------
    edges = cv2.Canny(mask, 50, 50)

    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    # return lines sorted by their match. take the best 4 that aren't too similar

    lines = cv2.HoughLines(edges,1,np.pi/180,50)
    min_d_rho = 10
    min_d_theta = 10
    savedLines = []

    # get the best lines that aren't too close to eachother
    for line in lines:
        line = line[0]
        if len(savedLines) == 4:
            break
        too_close = False
        for saved in savedLines:
            if abs(saved[0] - line[0]) < min_d_rho and abs(saved[1] - line[1]) < min_d_theta:
                too_close = True
        if not too_close:
            savedLines.append([line[0], line[1]])

    for line in savedLines:
        [rho, theta] = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(masked,(x1,y1),(x2,y2),(0,0,255),2)


    cv2.imshow("1", image)
    cv2.imshow("3", masked)
    cv2.waitKey()
