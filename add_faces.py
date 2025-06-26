import cv2

video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('data/face.xml')


faces_data=[]


while(True):
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h,x:x+w,]
        resized_img=cv2.resize(crop_img,(50,50))
        if len(faces_data)<=100 and i%10==0:
            faces_data.append(resized_img)
        i=i+1 
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1)
    cv2.imshow('video', frame)
    if len(faces_data)==100 or cv2.waitKey(1)== ord('q'):
        break



video.release()
cv2.destroyAllWindows()