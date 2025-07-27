import cv2
from insightface.app import FaceAnalysis
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the InsightFace model (ArcFace-based)
model = FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1)  # Use CPU; change to 0 for GPU if available
known_person = {}

def load_faces(folder_path):
    # Load known faces from the given folder path
    if not os.path.exists(folder_path):
        print(f"[!] Folder '{folder_path}' not found.")
        return

    for file in os.listdir(folder_path):
        name = os.path.splitext(file)[0]  # Extract the name from the file name
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Convert BGR image to RGB for InsightFace processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = model.get(img_rgb)
        if faces and len(faces) > 0:
            # Store the face embedding reshaped for comparison
            known_person[name] = faces[0].embedding.reshape(1, -1)
            print(f"Successfully added face for {name}.")
        else:
            print(f"No face found in image {name}.")

# Load known faces from directory
load_faces("people")

def find_name(embedding):
    # Find the name of the person by comparing embeddings
    embedding = embedding.reshape(1, -1)
    name_unknown = "unknown"
    similarity_threshold = 0.7  # Minimum similarity for recognition

    for name, known_embedding in known_person.items():
        sim = cosine_similarity(embedding, known_embedding)[0][0]
        if sim > similarity_threshold:
            similarity_threshold = sim
            name_unknown = name
    return name_unknown

# Open webcam stream
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame from BGR to RGB for face detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = model.get(frame_rgb)

    for face in faces:
        bbox = face.bbox.astype(int)
        name = find_name(face.embedding)
        # Draw rectangle around face and put the identified name
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("ArcFace Face Recognition", frame)

    # Press 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
