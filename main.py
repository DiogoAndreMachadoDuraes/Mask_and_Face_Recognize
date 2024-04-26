import cv2 as cv
import functions
import face_recognition
import numpy as np

cam = cv.VideoCapture(0)
file_name = "haarcascade_frontalface_alt2.xml"
classifier = cv.CascadeClassifier(f"{cv.haarcascades}/{file_name}")

data_frame = functions.load_data()

X_train, X_test, y_train, y_test = functions.train_test(data_frame)
pca = functions.pca_model(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

knn = functions.knn(X_train, y_train)

person1_image = face_recognition.load_image_file(f"img/persons/person1.jpg")
person2_image = face_recognition.load_image_file(f"img/persons/person2.jpg")

person1_encoding = face_recognition.face_encodings(person1_image)[0]
person2_encoding = face_recognition.face_encodings(person2_image)[0]

known_face_encodings = [
    person1_encoding,
    person2_encoding
]
known_face_names = [
    "Person 1",
    "Person 2"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


label = {
    0: "You are wearing a mask",
    1: "You are not wearing a mask"
}

while True:
    status, frame = cam.read()

    if not status:
        break

    ch = cv.waitKey(1)

    if ch == ord('q'):
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    faces = classifier.detectMultiScale(gray)

    small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    process_this_frame = not process_this_frame

    for x,y,w,h in faces:
        gray_face = gray[y:y+h, x:x+w]

        if gray_face.shape[0] >= 200 and gray_face.shape[1] >= 200:
            gray_face = cv.resize(gray_face, (160,160))
            vector = pca.transform([gray_face.flatten()])

            pred = knn.predict(vector)[0]
            result = label[pred]

            if pred == 0:
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
                    font = cv.FONT_HERSHEY_DUPLEX
                    cv.putText(frame, name, (left , bottom - 6), font, 1.0, (255, 255, 255), 1)
                    cv.putText(frame, result, (left + 6 , top - 10), font, 0.5, (255, 255, 255), 2)
                    cv.putText(frame, f"{len(faces)} identified faces",(20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv.LINE_AA)

            elif pred == 1:
                cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
                cv.putText(frame, result, (x + 20,y + h - 250), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv.putText(frame, f"{len(faces)} identified faces",(20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv.LINE_AA)
                
    cv.imshow("Recognizer", frame)