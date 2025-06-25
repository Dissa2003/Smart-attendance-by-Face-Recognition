import cv2

video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('data/face.xml')




while(True):
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1)
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



video.release()
cv2.destroyAllWindows()