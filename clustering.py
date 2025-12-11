import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2

# Set the coordinates of the points
points = np.array([[504, 464], [502, 466], [498, 466], [498, 472], [496, 474], [496, 499], [498, 502], [500, 502], [506, 508], [508, 508], [510, 510], [512, 510], [514, 512], [516, 512], [520, 516], [522, 516], [524, 518], [528, 518], [530, 520], [532, 520], [536, 524], [538, 524], [540, 526], [546, 526], [548, 528], [564, 528], [566, 526], [596, 526], [600, 522], [602, 522], [604, 520], [604, 518], [606, 516], [606, 496], [598, 488], [598, 486], [596, 483], [596, 482], [590, 476], [590, 474], [588, 472], [586, 472], [582, 467], [582, 466], [580, 466], [578, 464]])

# Create a KMeans object with 4 clusters
kmeans = KMeans(n_clusters=4)

# Fit the model to the data
kmeans.fit(points)

# Assign the labels to the points
labels = kmeans.labels_

# Store each cluster into an array
clusters = np.zeros((4, len(points)))
for i in range(len(points)):
    cluster = labels[i]
    clusters[cluster][i] = points[i]

print(clusters)
# # Draw lines for each cluster
# for cluster in range(4):
#     color = (0, 255, 0)
#     for point in clusters[cluster]:
#         cv2.line(image, (point[0], point[1]), (point[0], point[1] + 20), color, 2)
#
# # Show the image
# cv2.imshow("Image", image)
# cv2.waitKey(0)