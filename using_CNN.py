import cv2
import numpy as np
import csv
from datetime import datetime
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Function to create a new CSV file with headers
def create_new_csv():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'attendance_{timestamp}.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date", "Time"])
    return csv_filename

# Initialize MTCNN for face detection and FaceNet for face recognition
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load and encode multiple reference images
known_face_embeddings = []
known_face_names = []

# Assume reference images are stored in a directory called "reference_images"
reference_dir = "C:/p/attendence/image/"

for filename in os.listdir(reference_dir):
    if filename.endswith((".jpg", ".jpeg", ".png", ".webp")):
        image_path = os.path.join(reference_dir, filename)
        img = Image.open(image_path)
        print(f"Loading reference image: {filename}")

        # Detect face and get embedding
        face = mtcnn(img)
        if face is None:
            print(f"No face detected in {filename}. Skipping this image.")
            continue

        if face.dim() == 5:
            face = face.squeeze(0)  # Remove the extra dimension
        elif face.dim() == 3:
            face = face.unsqueeze(0)  # Add batch dimension if it's missing

        embedding = resnet(face).detach().cpu().squeeze()
        if embedding.dim() > 1:
            embedding = embedding.mean(dim=0)  # Take the mean if there are multiple embeddings
        known_face_embeddings.append(embedding)
        known_face_names.append(os.path.splitext(filename)[0])  # Use filename (without extension) as the person's name

print(f"Loaded {len(known_face_names)} reference images")

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Create a new CSV file for this run
csv_filename = create_new_csv()
attendance_marked = set()  # To keep track of marked attendances

recognition_history = {name: 0 for name in known_face_names}

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Convert the image from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)

    # Detect faces
    try:
        boxes, _ = mtcnn.detect(img)
    except RuntimeError:
        print("No faces detected in this frame")
        boxes = None

    recognized_names = []  # To keep track of names recognized in this frame

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            # Extract face from the image
            face = img.crop(box)
            try:
                face_tensor = mtcnn(face)
            except RuntimeError:
                print("Failed to process detected face")
                continue

            if face_tensor is not None:
                # Ensure face is 4D (batch, channels, height, width)
                if face_tensor.dim() == 5:
                    face_tensor = face_tensor.squeeze(0)  # Remove the extra dimension
                elif face_tensor.dim() == 3:
                    face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension if it's missing
                
                # Get embedding
                embedding = resnet(face_tensor).detach().cpu().squeeze()
                if embedding.dim() > 1:
                    embedding = embedding.mean(dim=0)  # Take the mean if there are multiple embeddings

                # Compare with known faces
                distances = [torch.nn.functional.pairwise_distance(embedding.unsqueeze(0), known_emb.unsqueeze(0)).item() 
                             for known_emb in known_face_embeddings]
                min_distance = min(distances)
                min_distance_index = distances.index(min_distance)
                matched_name = known_face_names[min_distance_index]

                if min_distance < 0.6:  # Threshold for face match
                    name = matched_name
                    recognized_names.append(name)
                    recognition_history[name] += 1
                    print(f"Recognized: {name}")
                    
                    # Check attendance for the recognized person
                    if recognition_history[name] >= 3:
                        if name not in attendance_marked:
                            print(f"Marking attendance for {name}")
                            now = datetime.now()
                            current_date = now.strftime("%Y-%m-%d")
                            current_time = now.strftime("%H:%M:%S")
                            
                            # Write to CSV file
                            with open(csv_filename, mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([name, current_date, current_time])
                            
                            attendance_marked.add(name)
                            print(f"Attendance marked for {name} at {current_time}")
                else:
                    print("No match found")

                # Draw a box around the face
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
    else:
        print("No faces detected in this frame")

    # Reset recognition history for names not detected in this frame
    for name in known_face_names:
        if name not in recognized_names:
            recognition_history[name] = 0

    # Display the resulting frame
    cv2.imshow("Attendance System", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()