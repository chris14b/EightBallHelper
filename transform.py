import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2

w_mask = 400
h_mask = 300

def isGrey(hsvPixel):
    return hsvPixel[1] < 20

# ---------------   GET THE FELT COLOR  ----------------------
# take in the image then put all pixels in 1 array
def findTableHue(hsv):
    counts = np.zeros(256)
    for i in range(hsv.shape[0]):
        for j in range(hsv.shape[1]):
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

def getNoFeltMask(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    isFeltHue = findTableHue(hsv)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if isFeltHue[hsv[i][j][0]] or hsv[i][j][0] == 0:
                hsv[i][j][2] = 0
            else:
                hsv[i][j][2] = 255

    mask = hsv[:,:,2]

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.erode(mask, kernel,iterations = 1)
    mask = cv2.dilate(mask, kernel,iterations = 1)

    return mask

# scales the image down, gets the mask based on felt hue, smooths it, fills it,
# scales it back to the original size, then returns it
def getTableMask(image):

    smaller = cv2.resize(image, (h_mask, w_mask))
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

    # clean up the image post resize
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i][j] < 40:
                mask[i][j] = 0
            else:
                mask[i][j] = 255
    mask = cv2.medianBlur(mask, 3)

    return mask

def geTableTop(image):
    mask = getTableMask(image)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked

# takes in a thresholded image, setting largest blob to having 255 values, 0 elsewhere
def getLargestBlob(mask):

    # find the blobs with just the 4 connectivity ie up and left check
    blobs = np.zeros((mask.shape[0], mask.shape[1]), np.uint16)
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

# assumes rho theta form
# from https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv
def getIntersect(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return (x0, y0)

def getCorners(image):

    mask = getMask(image)
    edges = cv2.Canny(mask, 50, 50)

    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    # return lines sorted by their match. take the best 4 that aren't too similar

    lines = cv2.HoughLines(edges,1,np.pi/180,30)
    min_d_rho = 10
    min_d_theta = 0.4
    savedLines = []

    # get the best 4 lines that aren't too close to eachother
    for line in lines:
        line = line[0]
        if len(savedLines) == 4:
            break
        too_close = False
        for saved in savedLines:
            d_rho = abs(saved[0] - line[0])
            d_theta = abs(saved[1] - line[1])
            d_theta = min(d_theta, np.pi - d_theta)
            if d_rho < min_d_rho and d_theta < min_d_theta:
                too_close = True
        if not too_close:
            savedLines.append([line[0], line[1]])

    # find the most parllel lines
    best_d_theta = np.pi
    most_parallel = [False, False, False, False]
    for i in range(4):
        for j in range(i+1, 4):
            if abs(savedLines[i][1] - savedLines[j][1]) < best_d_theta:
                best_d_theta = abs(savedLines[i][1] - savedLines[j][1])
                most_parallel = [False, False, False, False]
                most_parallel[i] = True
                most_parallel[j] = True

    linesA = []
    linesB = []
    for i in range(4):
        if most_parallel[i]:
            linesA.append(savedLines[i])
        else:
            linesB.append(savedLines[i])

    corners = [
        getIntersect(linesA[0], linesB[0]),
        getIntersect(linesA[0], linesB[1]),
        getIntersect(linesA[1], linesB[1]),
        getIntersect(linesA[1], linesB[0])
    ]

    return corners


def projectiveTransform(image):

    corners = getCorners(image)

    # get longest line:
    longest = 0
    for i in range(4):
        for j in range(i+1, 4):
            length = np.sqrt(pow(corners[i][0] - corners[j][0], 2) + pow(corners[i][1] - corners[j][1], 2))
            if length > longest:
                longest = length

    oldCorners = np.array(corners, dtype = "float32")
    newCorners = np.array([ [0, 0], [longest, 0], [longest, longest/2], [0, longest/2]], dtype = "float32")

    projectionMatrix = cv2.getPerspectiveTransform(oldCorners, newCorners)
    warped = cv2.warpPerspective(image, projectionMatrix, (int(longest), int(longest/2)))

    return warped


# makes less saturated pixels appear even greyer, and more
# saturated pixels more saturated
def contrastSaturations(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    for i in range(len(hls)):
        for j in range(len(hls[i])):
            sat = sigmoid((hls[i][j][2]-150.0)/40.0)
            hls[i][j][2] = 255.0 * sat

    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

if __name__ == "__main__":
    # First command line argument will be the file name of image. If none is supplied, generate random table

    img = cv2.imread(sys.argv[1])
    tableTop = geTableTop(img)
    onlyBalls = getNoFeltMask(tableTop)

    cv2.imshow("1", img)
    cv2.imshow("2", tableTop)
    cv2.imshow("3", onlyBalls)
    cv2.waitKey()
