import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
 
loaded_image = cv2.imread("D:\\Python Project\\python_programming\\tamrin7\\1.jpg")
loaded_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2RGB)
 
gray_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2GRAY)
 
edged_image = cv2.Canny(gray_image, threshold1=30, threshold2=100)

contours, hierarchy = cv2.findContours(edged_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(len(contours))

cv2.drawContours(loaded_image, contours, -1, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
plt.imshow(loaded_image)

result_number = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 20, param1=40, param2=30, minRadius=10, maxRadius=20)
result_number = result_number[0, :]

for circle in result_number:
    cv2.circle(loaded_image, (int(circle[0]), int(circle[1])), int(circle[2]), (250, 0, 0), 2)
    
plt.imshow(loaded_image)

x0, y0, w0, h0= cv2.boundingRect(contours[0])
detect1=cv2.rectangle(gray_image, (x0,y0),(x0+w0,y0+h0), (0,255,0),5)

x1, y1, w1, h1= cv2.boundingRect(contours[1])
detect2=cv2.rectangle(gray_image, (x1,y1),(x1+w1,y1+h1), (0,255,0),5)
plt.imshow(gray_image)


first_result = cv2.HoughCircles(detect1, cv2.HOUGH_GRADIENT, 1, 20, param1=40, param2=30, minRadius=10, maxRadius=20)
first_result = first_result[0, :]
first_dice_number = len(first_result)

second_result = cv2.HoughCircles(detect2, cv2.HOUGH_GRADIENT, 1, 20, param1=40, param2=30, minRadius=10, maxRadius=20)
second_result = second_result[0, :]
second_dice_number = len(second_result)

cv2.putText(loaded_image, f'value = {first_dice_number}', (x0, y0), cv2.FONT_HERSHEY_DUPLEX, 0.7, (50, 255, 50), 2)
cv2.putText(loaded_image, f'value = {second_dice_number}', (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (50, 255, 50), 2)

 
 
plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(loaded_image,cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(gray_image,cmap="gray")
plt.axis("off")
plt.title("GrayScale Image")
plt.subplot(1,3,3)
plt.imshow(edged_image,cmap="gray")
plt.axis("off")
plt.title("Canny Edge Detected Image")
plt.show()