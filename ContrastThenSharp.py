import cv2
import numpy as np
import matplotlib.pyplot as plt

# reading the image
image = cv2.imread('oldcar.jpeg')

# converting it into greyscale
grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# printing the GreyScale original image
plt.figure(1)
plt.title('Original Grey-scale Image')
plt.tight_layout()
plt.axis('off')
plt.imshow(abs(grey_image), plt.cm.gray)

# ENHANCEMENT STEPS
# Contrast Improvement
# generating kernel
c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe = c.apply(grey_image)

# Sharpening
# generating the kernel
kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
                             [-1,2,2,2,-1],
                             [-1,2,8,2,-1],
                             [-1,2,2,2,-1],
                             [-1,-1,-1,-1,-1]]) / 8.0
# applying kernel for sharpening
output_1 = cv2.filter2D(clahe, -1, kernel_sharpen)

# printing the values
plt.figure(2)
plt.title('Enhanced Image')
plt.tight_layout()
plt.axis('off')
plt.imshow(abs(output_1), plt.cm.gray)

# saving the result
cv2.imwrite('Oldcarfinal2.png',output_1)