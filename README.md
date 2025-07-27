#  Real-Time Face Recognition with ArcFace & InsightFace

A **real-time face recognition** system using **InsightFace** and **ArcFace embeddings**.

> The project uses your webcam to detect and recognize faces by comparing them to a dataset of known faces.

---

##  Features

- Face detection and recognition using **ArcFace embeddings**
- Match new faces with known individuals
- Live display of names on video frames using **OpenCV**
- Works on both CPU and GPU (if available)

---

## Technologies Used

- Python 3.10 (recommended)
- OpenCV
- InsightFace
- NumPy
- scikit-learn (for cosine similarity)

---

 The `people/` folder contains images of known individuals.
- The file name (e.g., `ali.jpg`) is used as the person's name for recognition.

