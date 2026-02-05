import cv2
import numpy as np
from scipy.spatial.distance import cosine

# Load the videos
video1_path = "r1.mp4"
video2_path = "r2.mp4"
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

if not cap1.isOpened() or not cap2.isOpened():
    print("❌ Error: Could not open videos.")
    exit()

# Initialize YOLOv4 model
yolo_weights = "yolov4.weights"
yolo_config = "yolov4.cfg"
net = cv2.dnn.readNet(yolo_weights, yolo_config)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load and process reference image
reference_image = cv2.imread("ref2.jpg")
if reference_image is None:
    print("❌ Error: Could not load reference image.")
    exit()

def detect_persons(frame, conf_threshold=0.5):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)
    
    boxes = []
    confidences = []
    
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if class_id == 0 and confidence > conf_threshold:
                center_x, center_y, w, h = (obj[:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)
    return [boxes[i] for i in indices], [confidences[i] for i in indices]

def extract_features(frame, box):
    x, y, w, h = box
    # Ensure coordinates are within frame boundaries
    x, y = max(0, x), max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)
    
    if w <= 0 or h <= 0:
        return None
        
    person = frame[y:y+h, x:x+w]
    if person.size == 0:
        return None
    
    # Resize person to standard size
    person = cv2.resize(person, (128, 256))
    
    # Calculate multiple features
    features = []
    
    # 1. HSV Color histogram
    hsv = cv2.cvtColor(person, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
    features.extend([hist_h.flatten(), hist_s.flatten()])
    
    # 2. Upper body features (assuming upper 40% of person)
    upper_body = person[:int(person.shape[0] * 0.4), :]
    upper_hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
    hist_upper = cv2.calcHist([upper_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist_upper, hist_upper, 0, 1, cv2.NORM_MINMAX)
    features.append(hist_upper.flatten())
    
    # Combine all features
    return np.concatenate([f for f in features])

def compare_features(feat1, feat2):
    if feat1 is None or feat2 is None:
        return 0
    return 1 - cosine(feat1.flatten(), feat2.flatten())

# Get reference features from reference image
boxes_ref, _ = detect_persons(reference_image, conf_threshold=0.5)
if not boxes_ref:
    print("❌ No person detected in reference image.")
    exit()

reference_features = extract_features(reference_image, boxes_ref[0])

# Process first frame of Video 1
ret1, frame1 = cap1.read()
if not ret1:
    print("❌ Error: Could not read first frame of Video 1.")
    exit()

# Find best match in first frame
boxes1, _ = detect_persons(frame1, conf_threshold=0.5)
if not boxes1:
    print("❌ No person detected in Video 1.")
    exit()

best_match = None
best_score = 0
for box in boxes1:
    features = extract_features(frame1, box)
    score = compare_features(reference_features, features)
    if score > best_score:
        best_score = score
        best_match = box

if best_score < 0.3:  # Threshold for acceptable match
    print("⚠️ Warning: Low confidence in initial person match")

# Initialize tracker with best match
tracker1 = cv2.TrackerCSRT_create()
tracker1.init(frame1, tuple(best_match))

# Initialize variables for Video 2
tracker2 = None
last_good_box2 = None

# Process both videos
while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        break

    # Track in Video 1
    success1, box1 = tracker1.update(frame1)
    if success1:
        x, y, w, h = [int(v) for v in box1]
        current_features = extract_features(frame1, [x, y, w, h])
        match_score = compare_features(reference_features, current_features)
        
        if match_score > 0.3:  # Good match
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame1, f"Score: {match_score:.2f}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:  # Poor match, try to redetect
            boxes, _ = detect_persons(frame1)
            best_match = None
            best_score = 0
            for box in boxes:
                features = extract_features(frame1, box)
                score = compare_features(reference_features, features)
                if score > best_score:
                    best_score = score
                    best_match = box
            if best_match and best_score > 0.3:
                tracker1 = cv2.TrackerCSRT_create()
                tracker1.init(frame1, tuple(best_match))

    # Process Video 2
    if tracker2 is None:
        # Initial detection in Video 2
        boxes2, _ = detect_persons(frame2)
        best_match = None
        best_score = 0
        for box in boxes2:
            features = extract_features(frame2, box)
            score = compare_features(reference_features, features)
            if score > best_score:
                best_score = score
                best_match = box
        if best_match and best_score > 0.3:
            tracker2 = cv2.TrackerCSRT_create()
            tracker2.init(frame2, tuple(best_match))
            last_good_box2 = best_match
    else:
        success2, box2 = tracker2.update(frame2)
        if success2:
            x, y, w, h = [int(v) for v in box2]
            current_features = extract_features(frame2, [x, y, w, h])
            match_score = compare_features(reference_features, current_features)
            
            if match_score > 0.3:
                cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame2, f"Score: {match_score:.2f}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                last_good_box2 = [x, y, w, h]
            else:
                boxes2, _ = detect_persons(frame2)
                best_match = None
                best_score = 0
                for box in boxes2:
                    features = extract_features(frame2, box)
                    score = compare_features(reference_features, features)
                    if score > best_score:
                        best_score = score
                        best_match = box
                if best_match and best_score > 0.3:
                    tracker2 = cv2.TrackerCSRT_create()
                    tracker2.init(frame2, tuple(best_match))
                    last_good_box2 = best_match

    # Display videos side by side
    height = min(frame1.shape[0], frame2.shape[0])
    frame1_resized = cv2.resize(frame1, (int(frame1.shape[1] * height / frame1.shape[0]), height))
    frame2_resized = cv2.resize(frame2, (int(frame2.shape[1] * height / frame2.shape[0]), height))
    combined_frame = np.hstack((frame1_resized, frame2_resized))
    
    cv2.imshow("Video Comparison", combined_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
