import numpy as np
import cv2

def convert_to_greyscale(img):
    #the algorithm to convert a RGB to greyscale described in: https://stackoverflow.com/a/51286918
    color_weights = np.array([0.07, 0.72, 0.21])
    grey_image = np.tensordot(a=img, b=color_weights, axes=(2,0))
    return grey_image

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
def extract_face(gray):
    #Taken from
    #https://www.tutorialspoint.com/how-to-detect-cat-faces-in-an-image-in-opencv-using-python#:~:text=Steps%201%20Import%20the%20required%20library.%20...%202,the%20original%20image%20using%20cv2.rectangle%20%28%29.%20More%20items

    # Detects faces in the input image
    faces = cascade.detectMultiScale(gray)
    # if atleast one face id detected
    if len(faces) > 0:
        x,y,w,h = faces[0]
        return gray[y:y+h,x:x+w]
    else:
        raise ValueError("No cat face detected")