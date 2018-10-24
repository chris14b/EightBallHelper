import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import cv2
import transform

# take in the image then put all pixels in 1 array
initial = cv2.imread(sys.argv[1])
img = cv2.resize(initial, (600, 400))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

feltHue  = transform.findFeltHueAutomatic(hsv, 'hsv')
isFeltHue = transform.isFeltHueArray(feltHue)
isTable  = transform.getTableMask(hsv, feltHue, 'hsv')

hsv = transform.normaliseSatAndVal(hsv, isTable, 'hsv')
hsv = transform.contrastSaturations(hsv)

new = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# gets the canny edges of the images
canny =  cv2.Canny(img, 200, 80)
canny = cv2.bitwise_and(canny, canny, mask=isTable)

# try for different radii. This is all hacky
num = 20
min = 4
while num > 15 and min < 10:
    min += 1
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, min-1,
                            param1=100,param2=4,minRadius=min,maxRadius=min+2)
    print(circles)
    num = len(circles[0])

min -= 1
circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, min-1,
                        param1=50,param2=4,minRadius=min,maxRadius=min+2)

circles = np.uint16(np.around(circles))

median = int(np.mean(circles[:,:,2]))
# print circles
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    # cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
    cv2.imshow("1", img)
    cv2.waitKey()

cv2.imshow("1", img)
cv2.imshow("2",canny)
cv2.waitKey()
sys.exit()

colourPixels = []
for i in range(len(hsv)):
    for j in range(len(hsv[i])):
        if isTable[i][j] and not isFeltHue[hsv[i][j][0]] and hsv[i][j][1] > 180:
            colourPixels.append([img[i][j]])
            included[i][j] = (255)
        else:
            included[i][j] = (0)

colourPixels = np.array(colourPixels)
colourPixels = colourPixels.reshape(-1, 3)
imgPixels = img.reshape(-1, 3)

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
