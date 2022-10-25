import cv2

from random import randrange

#setting the window size
cv2.namedWindow('Car and Pedestrains Detector', cv2.WINDOW_NORMAL)  

#importing the video file to the code and save in some variable
video = cv2.VideoCapture('dashcam1.mp4')
# video = cv2.VideoCapture('dashcam2.mp4')
# video = cv2.VideoCapture('dashcam3.mp4')

#adding our pre-trained algorithm's tio the variables 
car_tracker_algorithm = 'car_algoritham.xml'
pedestrain_tracker_algorithm = 'pedestrain_algorithm.xml'

#adding the classifiers
car_tracker = cv2.CascadeClassifier(car_tracker_algorithm)
pedestrain_tracker = cv2.CascadeClassifier(pedestrain_tracker_algorithm)

#run the code forever until the car stops
while True:

    #read the current frames
    (read_successful, frame) = video.read()

    #safe execution of the code
    if read_successful:
        #convert the color frame to the gray frame or black and white frame
        grayscaled_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect car and the pedestrains
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrains = pedestrain_tracker.detectMultiScale(grayscaled_frame)

    #draw the rectangle for the cars and the pedestrains
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x+1,y+2), (x+w,y+h), (255,0,0) , 2)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255) , 2)

    for (x,y,w,h) in pedestrains:
        cv2.rectangle(frame, (x,y), (x+w,y+h) , (0,255,255) , 2 )

    #to show the frame in pop-up-window
    cv2.imshow('Car and Pedestrains Detector',frame)

    #to stay in the frame
    key = cv2.waitKey(1)

    #to end the loop
    if key==81 or key==113:
        break

video.release()

print("Code completed")