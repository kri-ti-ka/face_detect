# face-detect
face_data_collect.py
this python file uses numpy and cv2 libraries to collect face data. to achieve face recognition as to collect data, haar cascade ML model is used which helps by providing already trained data. in this file i've used "haarcascade_frontalface_alt.xml" the data collected by this file is stored in the folder "data" which further contains user input "names" following the file extension ".npy"

face_recognition.py
this file uses data collected by the previous file to recognize the face by using K Nearest neighbours also knn algorithm. knn is a popular method used in AI, machine learning and data analysis. it recognizes the face by comparing it to the files whose names end with ".npy"

face_snapchat.py
face_snapchat.py helps user try different types of filters which oddly resembles snapchat filters, without using snapchat. right now initiated with "sunglasses filter" which yet again uses the haarcascade machine learning model. in this filter we've used "haarcascade_eye.xml"  which identifies the eyes of the face(if present) along with "haarcascade_frontalface_default.xml" which is being used to check if face is present.
