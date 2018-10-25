import sys
import numpy as np
import cv2y

HUE_MAX = 180    # open cv does this so it stays within a uint8
HUE_WINDOW = 10

GREY_SAT = 60

def isGrey(hsvPixel):
    return hsvPixel[1] < GREY_SAT

# ------------------------   GET THE FELT COLOR  --------------------------
# assume that the most common hue in a give window is the felt hue
def findFeltHueAutomatic(hsv, given):
    if given != 'hsv':
        trhow("error: findFeltHueAutomatic expects a hsv")

    counts = np.zeros(HUE_MAX)
    for i in range(hsv.shape[0]):
        for j in range(hsv.shape[1]):
            if not isGrey(hsv[i][j]):
                counts[hsv[i][j][0]] += 1

    # get hue  window for the felt
    best = 0
    best_hue = 0
    for h in range(HUE_MAX):
        count = 0
        for w in range(HUE_WINDOW):
            count += counts[(h+w) % HUE_MAX]
        if count > best:
            best = count
            best_hue = h

    # actual hue center is offset
    return (best_hue + int(HUE_WINDOW/2)) % HUE_MAX


# ----------------------  Felt hue array  -----------------------------
# simply makes a lookup truth table for hue values
def isFeltHueArray(hue):

    # set those
    isFeltHue = [False] * HUE_MAX
    for w in range(HUE_WINDOW):
        isFeltHue[(hue + w - int(HUE_WINDOW/2) + HUE_MAX) % HUE_MAX] = True

    return isFeltHue


# ----------------------  Find the circles/balls  -----------------------------
# assumes the size of the balls don't change too much
# also assumes that the best matches are balls...
def getCircles(img, mask, radius, given):

    if given != 'bgr':
        throw("Error: getCircles only accepts bgr atm ")

    # gets the canny edges of the images
    canny = cv2.Canny(img, 100, 80)
    canny = cv2.bitwise_and(canny, canny, mask=mask)

    # cv2.imshow("pre", mask)

    kernel = np.ones((3,3),np.uint8)
    smoothed = cv2.dilate(canny, kernel,iterations = 1)
    smoothed = cv2.medianBlur(smoothed, 3)

    avg = np.add(canny/2, smoothed/2)
    avg = np.array(avg, dtype=np.uint8)

    # cv2.imshow("post", avg)

    circles = cv2.HoughCircles(avg, cv2.HOUGH_GRADIENT, 1, 1.5*radius,
                            param1=50,param2=6,minRadius=radius-1,maxRadius=radius+1)

    circles = np.uint16(np.around(circles))

    return circles[0]


def midPoint(tuple1, tuple2):
    x = int((tuple1[0] + tuple2[0])/2)
    y = int((tuple1[1] + tuple2[1])/2)
    return (x, y)


# ----------------------  Find the pockets  -----------------------------
# figure make an edge mask, then find 6 largest circles?
def getPockets(img, corners, radius):

    # # make a mask vaguely where corners are, and half way between them
    mask = np.zeros((img.shape[0],img.shape[1], 1),np.uint8)
    cv2.circle(mask, corners[0], radius*2, (255), -1)
    cv2.circle(mask, corners[1], radius*2, (255), -1)
    cv2.circle(mask, corners[2], radius*2, (255), -1)
    cv2.circle(mask, corners[3], radius*2, (255), -1)
    cv2.circle(mask, midPoint(corners[0], corners[1]), radius*2, (255), -1)
    cv2.circle(mask, midPoint(corners[1], corners[2]), radius*2, (255), -1)
    cv2.circle(mask, midPoint(corners[2], corners[3]), radius*2, (255), -1)
    cv2.circle(mask, midPoint(corners[3], corners[0]), radius*2, (255), -1)

    # gets the canny edges of the images
    canny = cv2.Canny(img, 100, 80)
    canny = cv2.bitwise_and(canny, canny, mask=mask)

    kernel = np.ones((3,3),np.uint8)
    smoothed = cv2.dilate(canny, kernel,iterations = 1)
    smoothed = cv2.medianBlur(smoothed, 3)

    avg = np.add(canny/2, smoothed/2)
    avg = np.array(avg, dtype=np.uint8)

    # cv2.imshow("post", avg)

    circles = cv2.HoughCircles(avg, cv2.HOUGH_GRADIENT, 1, 10*radius,
                            param1=50,param2=4,minRadius=radius-1,maxRadius=radius+1)

    circles = np.uint16(np.around(circles))

    return circles[0][:6]

# -------------------- Automatic ball radius finder  -----------------------------
# assumes the size of the balls don't change too much
# also assumes there are a few balls to get the median of
def findBallRadiusAutomatic(img, mask, given):

    if given != 'bgr':
        throw("Error: getCircles only accepts bgr atm ")

    # gets the canny edges of the images
    canny =  cv2.Canny(img, 100, 80)
    canny = cv2.bitwise_and(canny, canny, mask=mask)

    kernel = np.ones((3,3),np.uint8)
    smoothed = cv2.dilate(canny, kernel,iterations = 1)
    smoothed = cv2.medianBlur(smoothed, 3)

    avg = np.add(smoothed/2, canny/2)
    avg = np.array(avg, dtype=np.uint8)

    # run the algorithm with many radii to try find the best match
    circles = cv2.HoughCircles(avg, cv2.HOUGH_GRADIENT, 1, 2,
                                 param1=50,param2=10,minRadius=5,maxRadius=10)

    #run it again get the median radius of the top 5 circles to use next
    radius = int(np.median(circles[0,:5,2])) + 1 # err on the side of larger

    return radius

# ----------------------- get a filled in Table blob  --------------------------
# scales the image down, gets the mask based on felt hue, smooths it, fills it,
# scales it back to the original size, then returns it
def getTableMask(image, hue, given):

    if given != 'hsv':
        throw("Error: getTableMask only accepts hsv atm ")

    hsv = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
    isFeltHue = isFeltHueArray(hue)

    for i in range(hsv.shape[0]):
        for j in range(hsv.shape[1]):
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
    for i in range(mask.shape[0]):
        bot = 0
        top = mask.shape[1] - 1
        while mask[i][bot] == 0 and top > bot:
            bot += 1
        while mask[i][top] == 0 and top > bot:
            top -= 1
        if mask[i][top] == 0: #nothing to fill in
            continue
        for j in range(bot, top+1):
            mask[i][j] = 255
    for j in range(mask.shape[1]):
        left = 0
        right = mask.shape[0]  - 1
        while mask[left][j] == 0 and right > left:
            left += 1
        while mask[right][j] == 0 and right > left:
            right -= 1
        if mask[left][j] == 0: #nothing to fill in
            continue
        for i in range(left, right+1):
            mask[i][j] = 255

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # clean up the image post resize
    thresh = mask > 40
    mask[thresh] = [255]
    # mask = cv2.medianBlur(mask, 3)

    return mask

# -------------------- Blob detection and largest blob  -----------------------
# takes in a thresholded image, setting largest blob to having 255 values, 0 elsewhere
def getLargestBlob(mask):

    # find the blobs with just the 4 connectivity ie up and left check
    blobs = np.zeros((mask.shape[0], mask.shape[1]), np.uint16)
    curr_label = 1 # 0 is for no label
    label_equiv = [0]
    counts = [0]
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):

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
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if label_equiv[blobs[i][j]] == largest_blob:
                mask[i][j] = 255
            else:
                mask[i][j] = 0
    return mask

# ------------------------- Hough line intersection  ----------------------------
# taken from https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv
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
    intersect = [x0, y0]
    return (x0, y0)

# ------------------------ Get the table corners  ----------------------------
# TODO: change so that this only takes in a mask, or split up into find hough lines
# then pass this to a corners part
def getCorners(mask, given):

    if given != 'mask':
        throw("getCorners takes a mask")

    # get the edges, and make them thicker so theres more votes along the best axii
    edges = cv2.Canny(mask, 50, 20)
    edges = cv2.dilate(edges, np.ones((3,3),np.uint8))

    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    # return lines sorted by their match. take the best 4 that aren't too similar

    lines = cv2.HoughLines(edges,1,np.pi/360,30)
    min_d_rho = 40
    min_d_theta = 0.3
    savedLines = []

    # get the best 4 lines that aren't too close to eachother
    for line in lines:
        line = line[0]
        if len(savedLines) == 4:
            break
        too_close = False
        for saved in savedLines:
            d_rho = abs(abs(saved[0]) - abs(line[0]))
            d_theta = abs(saved[1] - line[1])
            d_theta = min(d_theta, np.pi - d_theta)
            if d_rho < min_d_rho and d_theta < min_d_theta:
                too_close = True
        if not too_close:
            savedLines.append([line[0], line[1]])

    # for line in savedLines:
    #     print(line)
    #     rho = line[0]
    #     theta = line[1]
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #     pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #     cv2.line(mask, pt1, pt2, (100,100,255), 3)
    #
    # cv2.imshow('efb', mask)
    # cv2.waitKey()

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


# -------------  Get matrix project table to 2:1 Rextange  --------------------
# finds the corners, then orders them(TODO), then finds which is the long one (TODO)
# then finds matrix that affine projects it all to a 2:1 rectangle
def projectiveTransform(corners, image):

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

    return warped, projectionMatrix

# -------------  normalise sat and v values in a hsv with a mask  --------------------
# Just scales so that there is a 0 sat and a 255 sat, same for value
def normaliseSatAndVal(hsv, given):

    if given != 'hsv':
        throw("Error: getBalls only accepts hsv atm ")

    minS = np.min(hsv[:,:,1])
    maxS = np.max(hsv[:,:,1])
    minV = np.min(hsv[:,:,2])
    maxV = np.max(hsv[:,:,2])

    # x -> (x-x_min)*(255/(x_max-xmin)) -> (x-x_min)*x_scale
    np.subtract(hsv[:,:,1], minS)
    np.subtract(hsv[:,:,2], minV)
    np.multiply(hsv[:,:,1], 255.0/(maxS - minS))
    np.multiply(hsv[:,:,2], 255.0/(maxV - minV))

    return hsv


# --------------- Increase the contrast of saturation values -------------------
# makes less saturated pixels appear even greyer, and more saturated pixels more saturated
# Warning: a bit expensive
# def contrastSaturations(hsv, mask, given):
#
#     if given != 'hsv':
#         throw("Error: getBalls only accepts hsv atm ")
#
#     FORCE = 20.0 # smaller pushes the sat harder
#     for i in range(len(hsv)):
#         for j in range(len(hsv[i])):
#
#             if mask[i][j]:
#                 rescaled = (hsv[i][j][1] - 2.0*GREY_SAT)/FORCE
#                 hsv[i][j][1] = 255.0 * 1.0 / (1.0 + np.exp(-rescaled))
#
#     return hsv
