import cv2
# import tensorflow as tf
# import tensorflow_hub as hub

# Buat objek kamera
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Muat model TensorFlow (misalnya, model deteksi objek)
# model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()
    
    # Konversi frame menjadi grayscale (dibutuhkan oleh classifier)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dalam frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Gambar kotak di sekitar wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Tampilkan frame dengan kotak wajah
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela tampilan
cap.release()
cv2.destroyAllWindows()