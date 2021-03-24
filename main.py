import cap as cap
import numpy as np
import cv2

# Utilisation d'un algo HSV pour la détection de couleurs

class ColorDetector:
    # Detection de la cam
    def __init__(self, frame):
        self.camera = cv2.VideoCapture(0)
        self.frame = frame
        self.create_color_controller()

    def detect_colors(self):
        while (1):
            # Récuprer les valeurs des couleurs HSV
            h_min, s_min, v_min, h_max, s_max, v_max = self.get_tracker_pos()
            lower_bound = [h_min, s_min, v_min]
            upper_bound = [v_max, s_max, v_max]

            # Récupération des frames
            _, imageFrame = self.camera.read()
            height, width = imageFrame.shape[:2]
            self.frame -= 1
            # Conversion RGB vers HSV
            hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

            # Definition d'un kernel pour la transformation morphologique
            kernel = np.ones((5, 5), "uint8")

            # Mask et transformation morphologique de l'image pour le rouge
            low_red = np.array(lower_bound)
            high_red = np.array(upper_bound)
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
            pourcentage = 0
            if len(contours) > 0 and self.frame == 0:
                self.frame = 20
                maxContour = max(contours, key=cv2.contourArea)
                cv2.putText(green, str(cv2.contourArea(maxContour)*100/(height * width)), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))
                area = cv2.contourArea(maxContour)
                pourcentage = area*100/(height * width)
            return pourcentage, imageFrame, red, green, blue





    def visualiser(self, imageFrame, red, green, blue):
        cv2.imshow("Frame", imageFrame)
        cv2.imshow("Red", red)
        cv2.imshow("Blue", blue)
        cv2.imshow("Green", green)

    def create_color_controller(self):
        cv2.createTrackbar('Hmin', 'tracker', 0, 255, lambda : None)
        cv2.createTrackbar('Smin', 'tracker', 0, 255,  lambda : None)
        cv2.createTrackbar('Vmin', 'tracker', 0, 255, lambda : None)

        cv2.createTrackbar('Hmax', 'tracker', 0, 255, lambda : None)
        cv2.createTrackbar('Smax', 'tracker', 0, 255, lambda : None)
        cv2.createTrackbar('Vmax', 'tracker', 0, 255, lambda : None)

    def get_tracker_pos(self):
        h_min = cv2.getTrackbarPos("Hmin", "tracker")
        s_min = cv2.getTrackbarPos("Smin", "tracker")
        v_min = cv2.getTrackbarPos("Vmin", "tracker")

        h_max = cv2.getTrackbarPos("Hmax", "tracker")
        s_max = cv2.getTrackbarPos("Smax", "tracker")
        v_max = cv2.getTrackbarPos("Vmax", "tracker")

        return h_min, s_min, v_min, h_max, s_max, v_max


def main():
    colorDetector = ColorDetector(20)
    while 1:
        colorDetector.detect_colors()


if __name__ == "__main__":
    main()