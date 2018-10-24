#!/usr/bin/python3
import sys
import numpy as np
import cv2
import transform

# take in the image and do some preProcessing
initial = cv2.imread(sys.argv[1])
initial = cv2.resize(initial, (int(initial.shape[1]/4), int(initial.shape[0]/4)))

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

def isWhite(hsvPixel):
    return hsvPixel[1] < 130 and hsvPixel[2] > 170

def isBlack(hsvPixel):
    return  hsvPixel[2] < 40

# for i in range(len(hsv)):
#     for j in range(len(hsv[i])):
#         if not tableMask[i][j]:
#             img[i][j] = (0,0,0)
#         elif isWhite(hsv[i][j]):
#             img[i][j] = (250, 250, 250)
#         elif isBlack(hsv[i][j]):
#             img[i][j] = (0,0,0)
#         elif isFeltHue[hsv[i][j][0]] or (hsv[i][j][1] < 120 and hsv[i][j][2] > 60):
#             img[i][j] = (100,100,100)


corners = transform.getCorners(tableMask, 'mask')
cv2.line(img, corners[0], corners[1], (100,100,200), 2)
cv2.line(img, corners[1], corners[2], (100,100,200), 2)
cv2.line(img, corners[2], corners[3], (100,100,200), 2)
cv2.line(img, corners[3], corners[0], (100,100,200), 2)

for i in circles:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2]+1,(255,255,0),1)


cv2.imshow("1", img)
# cv2.imshow("2", initial)
cv2.waitKey()
sys.exit()
