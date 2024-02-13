import cv2
import face_recognition
import os
from datetime import datetime

# Path to the directory containing the known faces
known_faces_dir = "known_faces"


# Initialize lists to store known face encodings and names
known_face_encodings = []
known_face_names = []

# Load known faces and encodings
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".jpg"):
        face_image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
        face_encoding = face_recognition.face_encodings(face_image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])

# Initialize variables for attendance
present_students = set()

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the image from BGR color to RGB color
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Initialize list to store names of detected faces
    face_names = []

    for face_encoding in face_encodings:
        # Check if the face matches any known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
            present_students.add(name)

        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up the face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Face Recognition Attendance System', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save attendance record to a file
attendance_file = open("attendance.txt", "w")
attendance_file.write("Date: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
attendance_file.write("Present Students:\n")
for student in present_students:
    attendance_file.write(student + "\n")
attendance_file.close()

# Release webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
