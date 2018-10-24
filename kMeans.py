#!/usr/bin/python3
import sys
import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture
import cv2
import transform

# take in the image then put all pixels in 1 array
initial = cv2.imread(sys.argv[1])
initial = cv2.resize(initial, (int(initial.shape[1]/4), int(initial.shape[0]/4)))
img = initial.copy()
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# find the table hue and table mask
feltHue  = transform.findFeltHueAutomatic(hsv, 'hsv')
isFeltHue = transform.isFeltHueArray(feltHue)
tableMask  = transform.getTableMask(hsv, feltHue, 'hsv')

# --- Optional and expensive ---- hsv normalisations
# hsv = transform.normaliseSatAndVal(hsv, np.ones((hsv.shape[0], hsv.shape[1])), 'hsv')
# img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

#--------------------- gaussian mixes --------------------
# gmm = GaussianMixture(n_components=6, covariance_type='full', max_iter=100, random_state=2)
# gmm.fit(colourPixels)
#
# means = gmm.means_
# preds = gmm.predict(imgPixels)

#--------------------- k means --------------------
kmeans = KMeans(n_clusters=12, algorithm="elkan")
kmeans.fit(colourPixels)

means = kmeans.cluster_centers_
preds = kmeans.predict(imgPixels)

for i in range(len(img)):
    for j in range(len(img[i])):
        if included[i][j]:
            img[i][j] = means[preds[i*len(img[i]) + j]]
        elif not transform.isBlackorWhite(hsv[i][j]):
            img[i][j] = (150,150,150)


cv2.imshow("1", projected)
cv2.imshow("3", included)
cv2.imshow("2", img)
cv2.waitKey()
