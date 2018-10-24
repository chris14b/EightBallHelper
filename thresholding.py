#!/usr/bin/python3
import sys
import numpy as np
import cv2
import transform


def getBallsAndPockets(initial):
    hsv = cv2.cvtColor(initial, cv2.COLOR_BGR2HSV)
    hsv = transform.normaliseSatAndVal(hsv, 'hsv')
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # find the table hue and table mask
    feltHue  = transform.findFeltHueAutomatic(hsv, 'hsv')
    isFeltHue = transform.isFeltHueArray(feltHue)
    tableMask  = transform.getTableMask(hsv, feltHue, 'hsv')

    # get the balls
    radius = transform.findBallRadiusAutomatic(img, tableMask, 'bgr')
    circles = transform.getCircles(img, tableMask, radius,'bgr')

    # get the corners,
    corners = transform.getCorners(tableMask, 'mask')

    # Now get pockets, and get better corner estimates now that we have pockets:
    corners = transform.getCorners(tableMask, 'mask')
    pockets = transform.getPockets(img, corners, radius*2)
    cornersMissed = 4
    for c in range(4):
        for p in range(len(pockets)):
            dx = corners[c][0] - pockets[p][0]
            dy = corners[c][1] - pockets[p][1]
            if np.sqrt(dx*dx + dy*dy) < 25:
                corners[c] = (int(pockets[p][0]), int(pockets[p][1]))
                cornersMissed -= 1
                break

    # maybe we didn't get all the corners?
    pockets = pockets[:(6-cornersMissed)]

    # rule out the pockets in the table thing
    for p in pockets:
        # draw the outer circle
        cv2.circle(tableMask, (p[0],p[1]), p[2]-2, (0), -1)

    # go again with the new mask
    circles = transform.getCircles(img, tableMask, radius,'bgr')
    balls = []

    for c in circles:

        # see how many are felt
        numFelt = 0
        for i in range(3):
            for j in range(3):
                pix = hsv[c[1]-1+i][c[0]-1+j]
                if isFeltHue[pix[0]] and pix[1] > 50:
                    numFelt += 1

        if numFelt <= 2 and tableMask[c[1]][c[0]]:
            balls.append(c)

        # could either require 2 falses in a row, or make exceptions for ones
        # far from walls or whatever. i think 2 falses is more extendible
        # if numFelt > 2 or not tableMask[c[1]][c[0]]:
        #     cv2.circle(img, (c[0], c[1]), c[2]+1, (10,10,250), 1)
        # else:
        #     cv2.circle(img, (c[0], c[1]), c[2]+1, (255,255,0), 1)
        # cv2.imshow("1", img)
        # cv2.waitKey()

    # warped, matrix = transform.projectiveTransform(corners, img)
    #
    # cv2.imshow("1", img)
    # cv2.imshow("2", initial)
    # cv2.waitKey()

    return balls, pockets


if __name__ == "__main__":
    # take in the image and do some preProcessing
    initial = cv2.imread(sys.argv[1])
    scale = int(initial.shape[1]/700.0) + 1
    initial = cv2.resize(initial, (int(initial.shape[1]/scale), int(initial.shape[0]/scale)))
    balls, pockets = getBallsAndPockets(initial)
    sys.exit()



# INVALID = (100,100,100)

# def isWhite(hsvPixel):
#     return hsvPixel[1] < 130 and hsvPixel[2] > 170
#
# def isBlack(hsvPixel):
#     return  hsvPixel[2] < 40
#
# for i in range(len(hsv)):
#     for j in range(len(hsv[i])):
#         if not tableMask[i][j]:
#             img[i][j] = INVALID         # out of bounds
#         elif isFeltHue[hsv[i][j][0]] and not transform.isGrey(hsv[i][j]):
#             img[i][j] = INVALID         # felt
#         elif isWhite(hsv[i][j]):
#             img[i][j] = (250, 250, 250)
#         elif isBlack(hsv[i][j]):
#             img[i][j] = (0,0,0)
#         elif hsv[i][j][1] < 150 and hsv[i][j][2] > 70:
#             img[i][j] = INVALID         # not interesting
