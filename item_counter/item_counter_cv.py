import cv2
import matplotlib.pyplot as plt

import cvlib as cv
# Read the image

image = cv2.imread('dogs.jpg')

# Detect objects in the image

bbox, label, conf = cv.detect_common_objects(image)

# Draw bounding boxes around the objects

output_image = cv.object_detection.draw_bbox(image, bbox, label, conf)

# Count the number of dogs

num_dogs = len([l for l in label if l == 'dog'])

# Display the output image with the boxes and count

plt.imshow(output_image)

plt.title('Detected objects')

plt.axis('off')

plt.show()

print(f'Number of dogs detected: {num_dogs}')
