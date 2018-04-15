import cv2
import matplotlib.pyplot as plt


def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_faces(f_cascade, colored_img, scaleFactor=1.1):
    # just making a copy of image passed, so that passed image is not changed
    img_copy = colored_img.copy()

    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)
    # let's detect multiscale (some images may be closer to camera than others) images

    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);

    # go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(x,y,w,h)
    return img_copy


test2 = cv2.imread('face_attr\\s2\\1.pgm')
haar_face_cascade = cv2.CascadeClassifier(
    'D:\\Anaconda3.5.0.1\\envs\\untitled\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml')
# call our function to detect faces
faces_detected_img = detect_faces(haar_face_cascade, test2, scaleFactor=1.008)

# convert image to RGB and show image
plt.imshow(convertToRGB(faces_detected_img))
plt.show()
