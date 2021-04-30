import cv2
import numpy as np

# # Read image
img = cv2.imread("images\\training\\0000500.png")
# for i in range(len(img)):
#     for j in range(len(img[i])):
#         img[i][j][0] = 0
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = np.array([[-2, -2, -2],
                   [-2,  17,-2],
                   [-2, -2, -2]])
# kernel = np.array([[-1, -1, -1, -1, -1],
#                    [-1, -1,  3, -1, -1],
#                    [-1,  3,  8,  3, -1],
#                    [-1, -1,  3, -1, -1],
#                    [-1, -1, -1, -1, -1]])

res = cv2.filter2D(gray, -1, kernel)
# res = res + img
cv2.imshow("result", res)
cv2.waitKey(0)
cv2.destroyAllWindows()


