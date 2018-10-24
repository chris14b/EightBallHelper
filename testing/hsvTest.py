import sys, cv2, transform
import numpy as np

# take in the image then put all pixels in 1 array
initial = cv2.imread(sys.argv[1])
img = cv2.resize(initial, (600, 400))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# testing hsv image
for i in range(180):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    isFeltHue = transform.isFeltHueArray(i)
    for i in range(len(hsv)):
        for j in range(len(hsv[i])):
            if isFeltHue[hsv[i][j][0]]:
                hsv[i][j][2] = 0
    img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("1", img2)
    cv2.waitKey()

sys.exit()
