import cv2
import numpy as np
import mysql.connector
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox
import os

# ✅ MediaPipe Tasks API (Python 3.13 compatible)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================= GLOBAL STATE =================
camera_running = False
cap = None
db = None
cursor = None

last_marked_time = {}
session_marked = set()
current_identity = None   # 🔒 lock recognized identity per session

# ================= MEDIAPIPE FACE DETECTOR =================
MODEL_PATH = r"C:\Users\YASHASVINI\OneDrive\Desktop\Pro FR\blaze_face_short_range.tflite"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model file not found. Download blaze_face_short_range.tflite "
        "and place it in the same folder as this script."
    )

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(base_options=base_options)
face_detector = vision.FaceDetector.create_from_options(options)

# ================= DATABASE =================
def connect_to_database():
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Yashasvini08",
        database="attsys"
    )
    return db, db.cursor()

# ================= FACE UTILITIES =================
def get_embedding(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    detection_result = face_detector.detect(mp_image)
    if not detection_result.detections:
        return None

    bbox = detection_result.detections[0].bounding_box
    h, w, _ = image.shape

    x = max(0, bbox.origin_x)
    y = max(0, bbox.origin_y)
    bw = bbox.width
    bh = bbox.height

    face = image[y:y + bh, x:x + bw]
    if face.size == 0:
        return None

    face = cv2.resize(face, (112, 112))
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    return gray.flatten() / 255.0

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ================= LOAD STUDENTS =================
def load_students():
    encodings, ids, names = [], [], []
    db, cur = connect_to_database()

    cur.execute("SELECT id, name, facepath FROM student_details")
    for sid, name, path in cur.fetchall():
        if os.path.exists(path):
            img = cv2.imread(path)
            emb = get_embedding(img)
            if emb is not None:
                encodings.append(emb)
                ids.append(sid)
                names.append(name)

    cur.close()
    db.close()
    return encodings, ids, names

# ================= ATTENDANCE =================
def mark_attendance(student_id, student_name):
    global db, cursor

    if student_id in session_marked:
        return

    now = datetime.now()
    cooldown = timedelta(hours=1)

    if student_id in last_marked_time:
        if now - last_marked_time[student_id] < cooldown:
            return

    last_marked_time[student_id] = now
    session_marked.add(student_id)

    cursor.execute(
        "INSERT INTO attendance (student_id, student_name, attendance_time) "
        "VALUES (%s, %s, %s)",
        (student_id, student_name, now)
    )
    db.commit()

    print(f"[OK] Attendance marked for {student_name}")

# ================= CAMERA LOOP =================
def main_loop():
    global db, cursor, current_identity

    known_faces, ids, names = load_students()
    db, cursor = connect_to_database()

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        label = "No Face"
        color = (0, 0, 255)

        emb = get_embedding(frame)
        if emb is not None and known_faces:
            scores = [cosine_similarity(emb, k) for k in known_faces]
            best = np.argmax(scores)

            # 🔽 LOWERED THRESHOLD + STABLE LABEL
            if scores[best] > 0.88:
                current_identity = names[best]
                label = current_identity
                color = (0, 255, 0)
                mark_attendance(ids[best], names[best])
            else:
                if current_identity:
                    label = current_identity
                    color = (0, 255, 0)
                else:
                    label = "Unidentified"

        cv2.putText(
            frame, label, (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_camera()

    cleanup()

def cleanup():
    global db, cursor
    if cursor:
        cursor.close()
    if db:
        db.close()
    cv2.destroyAllWindows()

# ================= CAMERA CONTROL =================
def start_camera():
    global camera_running, cap, current_identity
    session_marked.clear()
    current_identity = None
    cap = cv2.VideoCapture(0)
    camera_running = True
    camera_button.config(text="Stop Camera")
    main_loop()

def stop_camera():
    global camera_running
    camera_running = False
    camera_button.config(text="Start Camera")

def toggle_camera():
    start_camera() if not camera_running else stop_camera()

# ================= STUDENT MANAGEMENT =================
def add_student():
    sid = student_id_entry.get()
    name = student_name_entry.get()
    path = face_image_path_entry.get()

    if not (sid and name and path):
        messagebox.showerror("Error", "Fill all fields")
        return

    db, cur = connect_to_database()
    cur.execute(
        "INSERT INTO student_details (id, name, facepath) VALUES (%s, %s, %s)",
        (sid, name, path)
    )
    db.commit()
    cur.close()
    db.close()

    messagebox.showinfo("Success", "Student added")

    student_id_entry.delete(0, tk.END)
    student_name_entry.delete(0, tk.END)
    face_image_path_entry.delete(0, tk.END)

# ================= GUI =================
root = tk.Tk()
root.title("Face Recognition Attendance (Python 3.13)")
root.geometry("600x400")

frame = tk.Frame(root)
frame.pack(pady=20)

camera_button = ttk.Button(frame, text="Start Camera", command=toggle_camera)
camera_button.grid(row=0, column=0, padx=10)

add_student_button = ttk.Button(frame, text="Add Student", command=add_student)
add_student_button.grid(row=0, column=1, padx=10)

exit_button = ttk.Button(
    frame,
    text="Exit",
    command=lambda: (stop_camera(), root.destroy())
)
exit_button.grid(row=0, column=2, padx=10)

student_id_entry = ttk.Entry(root)
student_id_entry.pack()

student_name_entry = ttk.Entry(root)
student_name_entry.pack()

face_image_path_entry = ttk.Entry(root)
face_image_path_entry.pack()

root.mainloop()
