import cap as cap
import numpy as np
import cv2

# Utilisation d'un algo HSV pour la détection de couleurs


# Detection de la cam
camera = cv2.VideoCapture(0)

# Start a while loop
frame = 20

while (1):
    # Récupération des frames
    _, imageFrame = camera.read()
    frame -= 1
    # Conversion RGB vers HSV
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Definition d'un kernel pour la transformation morphologique
    kernel = np.ones((5, 5), "uint8")

    # Mask et transformation morphologique de l'image pour le rouge
    low_red = np.array([164, 0, 0])
    high_red = np.array([244, 194, 194])
    red_mask = cv2.inRange(hsvFrame, low_red, high_red)
    # Dialate
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)
    red = cv2.bitwise_and(imageFrame, imageFrame, mask=red_mask)

    # Mask et transformation morphologique pour le bleu
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsvFrame, low_blue, high_blue)
    blue_mask = cv2.dilate(blue_mask, kernel,iterations=1)
    blue = cv2.bitwise_and(imageFrame, imageFrame, mask=blue_mask)

    # Mask et transformation morpholoigque pour le vert
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])
    green_mask = cv2.inRange(hsvFrame, low_green, high_green)
    green_mask = cv2.dilate(green_mask, kernel, iterations=1)
    green = cv2.bitwise_and(imageFrame, imageFrame, mask=green_mask)

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0 and frame == 0:
        frame = 20
        maxContour = max(contours, key=cv2.contourArea)
        cv2.putText(green, str(cv2.minAreaRect(maxContour)), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 255))
        print(str(cv2.minAreaRect(maxContour)))
        print("Area us  {}".format(cv2.contourArea(maxContour)))
        print("Arc length is {}".format(cv2.arcLength(maxContour, True)))






    cv2.imshow("Frame", imageFrame)
    cv2.imshow("Red", red)
    cv2.imshow("Blue", blue)
    cv2.imshow("Green", green)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
