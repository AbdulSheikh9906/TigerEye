"""
Robust Tiger Re-ID pipeline:
- YOLO detects tigers
- Simple centroid tracker keeps short-term identity stable across frames
- ResNet embeddings (normalized) are used as main similarity signal (avg embedding per tiger)
- ORB keypoint matching used when embeddings are ambiguous
- New tiger -> saved crop; average embedding updated when same tiger seen again
"""

import os
import time
import cv2
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

# ----------------------------
# Config / thresholds (tune these)
# ----------------------------
MODEL_PATH = "best_enlightengan_and_yolov8.pt"    # your YOLO model
SIMILARITY_THRESHOLD = 0.90            # cosine similarity to accept as same tiger
AMBIGUOUS_LOW = 0.75                   # below this => definitely new
AMBIGUOUS_HIGH = SIMILARITY_THRESHOLD  # between low & high => use ORB to decide
ORB_MATCH_THRESHOLD = 30               # if ORB good matches >= this -> same tiger
TRACK_DIST_THRESHOLD = 80              # px distance for centroid-to-track matching
MAX_MISSED_FRAMES = 10                 # remove track after this many missed frames
MIN_CROP_SIZE = 50                     # ignore tiny crops (likely false positives)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load models
# ----------------------------
print("Loading models...")
yolo = YOLO(MODEL_PATH)  # YOLO detector

resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()  # feature extractor
resnet = resnet.to(DEVICE).eval()

orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------
# Databases & trackers
# ----------------------------
os.makedirs("tiger_database", exist_ok=True)

tiger_db = {}   # tiger_id -> {"embedding": avg_vector (1D), "count": n, "samples": [path,...]}
tiger_count = 0

tracks = {}     # track_id -> {"centroid":(x,y), "tiger_id":Tiger_X, "missed":0}
next_track_id = 0

# ----------------------------
# Helper funcs
# ----------------------------
def get_embedding_from_crop(bgr_crop):
    """Return 1D normalized numpy embedding vector."""
    img = Image.fromarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = resnet(tensor).cpu().numpy().reshape(-1)
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm

def cosine_sim(a, b):
    """Assumes a and b are 1D normalized vectors or not; compute cosine similarity."""
    # ensure 1D
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def orb_good_matches(img1, img2):
    """Return number of good ORB matches between two BGR images."""
    try:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    except Exception:
        return 0
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return 0
    matches = bf.match(d1, d2)
    # matches sorted by distance (lower is better)
    good = [m for m in matches if m.distance < 60]  # distance threshold; tune if needed
    return len(good)

def identify_using_db(embedding, crop):
    """Return (tiger_id, is_new, best_score, optional_orb_matches)"""
    global tiger_count, tiger_db
    if not tiger_db:
        tiger_count += 1
        tid = f"Tiger_{tiger_count}"
        tiger_db[tid] = {"embedding": embedding.copy(), "count": 1, "samples": []}
        return tid, True, 0.0, 0

    best_score = -1
    best_id = None
    for tid, data in tiger_db.items():
        score = cosine_sim(embedding, data["embedding"])
        if score > best_score:
            best_score = score
            best_id = tid

    # clear cases:
    if best_score >= SIMILARITY_THRESHOLD:
        # same tiger: update running average embedding
        data = tiger_db[best_id]
        n = data["count"]
        new_avg = (data["embedding"] * n + embedding) / (n + 1)
        new_avg = new_avg / np.linalg.norm(new_avg)
        tiger_db[best_id]["embedding"] = new_avg
        tiger_db[best_id]["count"] += 1
        return best_id, False, best_score, 0

    if best_score < AMBIGUOUS_LOW:
        # likely a new tiger
        tiger_count += 1
        tid = f"Tiger_{tiger_count}"
        tiger_db[tid] = {"embedding": embedding.copy(), "count": 1, "samples": []}
        return tid, True, best_score, 0

    # ambiguous: use ORB matches with stored sample images (if any)
    # We'll compare crop to one sample per tiger (if exists) and pick best orb matches
    best_orb = 0
    best_orb_id = None
    for tid, data in tiger_db.items():
        # if they have a sample image saved, use it, else try to skip ORB for that id
        if data["samples"]:
            sample_path = data["samples"][0]
            sample_img = cv2.imread(sample_path)
            if sample_img is None:
                continue
            matches = orb_good_matches(crop, sample_img)
            if matches > best_orb:
                best_orb = matches
                best_orb_id = tid

    # if ORB indicates a strong match, assign accordingly
    if best_orb_id is not None and best_orb >= ORB_MATCH_THRESHOLD:
        # treat as existing tiger, update embedding & optionally add sample
        data = tiger_db[best_orb_id]
        n = data["count"]
        new_avg = (data["embedding"] * n + embedding) / (n + 1)
        new_avg = new_avg / np.linalg.norm(new_avg)
        tiger_db[best_orb_id]["embedding"] = new_avg
        tiger_db[best_orb_id]["count"] += 1
        return best_orb_id, False, best_score, best_orb

    # otherwise new
    tiger_count += 1
    tid = f"Tiger_{tiger_count}"
    tiger_db[tid] = {"embedding": embedding.copy(), "count": 1, "samples": []}
    return tid, True, best_score, best_orb

# ----------------------------
# Simple centroid tracker helpers
# ----------------------------
def bbox_centroid(box):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy)

def euclidean(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# ----------------------------
# Main loop
# ----------------------------
print("Opening webcam (press 'q' to quit)...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # use CAP_DSHOW on Windows

if not cap.isOpened():
    raise IOError("Cannot open webcam - check index/permissions/other apps")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame_idx += 1
    # YOLO detection
    results = yolo(frame)  # returns list of results
    detections = []  # list of dicts: {'box':(x1,y1,x2,y2), 'conf':, 'cls':}
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if yolo.names[cls_id] != "tiger":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            # sanitize bbox
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            if x2-x1 < MIN_CROP_SIZE or y2-y1 < MIN_CROP_SIZE:
                continue
            detections.append({'box': (x1, y1, x2, y2), 'conf': conf})

    # build list of centroids
    det_centroids = [bbox_centroid(d['box']) for d in detections]
    assigned_track_ids = set()
    used_detections = set()

    # Match detections to existing tracks by centroid distance
    for det_idx, centroid in enumerate(det_centroids):
        best_tid = None
        best_dist = 1e9
        for track_id, tdata in tracks.items():
            dist = euclidean(centroid, tdata['centroid'])
            if dist < best_dist and dist < TRACK_DIST_THRESHOLD:
                best_dist = dist
                best_tid = track_id
        if best_tid is not None:
            # assign detection -> this track
            tracks[best_tid]['centroid'] = centroid
            tracks[best_tid]['missed'] = 0
            tracks[best_tid]['last_frame'] = frame_idx
            assigned_track_ids.add(best_tid)
            used_detections.add(det_idx)
            # We won't change tiger_id here; we'll still compute crop & optionally update embedding
            det = detections[det_idx]
            x1,y1,x2,y2 = det['box']
            crop = frame[y1:y2, x1:x2]
            emb = get_embedding_from_crop(crop)
            # update corresponding tiger's avg embedding to refine profile
            tid = tracks[best_tid]['tiger_id']
            # small update: treat as seen (update avg)
            data = tiger_db.get(tid)
            if data is not None:
                n = data['count']
                new_avg = (data['embedding'] * n + emb) / (n+1)
                new_avg = new_avg / np.linalg.norm(new_avg)
                tiger_db[tid]['embedding'] = new_avg
                tiger_db[tid]['count'] += 1
            # Draw
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # For unmatched detections -> create new track, identify vs DB
    for det_idx, det in enumerate(detections):
        if det_idx in used_detections:
            continue
        x1,y1,x2,y2 = det['box']
        crop = frame[y1:y2, x1:x2]
        emb = get_embedding_from_crop(crop)
        tid, is_new, best_score, orb_matches = identify_using_db(emb, crop)
        # save sample if new or sometimes also append sample to existing
        if is_new:
            sample_path = os.path.join("tiger_database", f"{tid}.jpg")
            cv2.imwrite(sample_path, crop)
            tiger_db[tid]['samples'].append(sample_path)
            print(f"[Frame {frame_idx}] NEW tiger registered: {tid} (score={best_score:.3f}, orb={orb_matches})")
        else:
            # optionally save more samples to improve ORB matching later
            if np.random.rand() < 0.05:  # randomly append few samples
                sample_path = os.path.join("tiger_database", f"{tid}_{int(time.time())}.jpg")
                cv2.imwrite(sample_path, crop)
                tiger_db[tid]['samples'].append(sample_path)
            print(f"[Frame {frame_idx}] Matched to {tid} (score={best_score:.3f}, orb={orb_matches})")

        # create new track for this detection
        track_id = next_track_id
        next_track_id += 1
        centroid = bbox_centroid((x1,y1,x2,y2))
        tracks[track_id] = {"centroid": centroid, "tiger_id": tid, "missed": 0, "last_frame": frame_idx}
        # draw box and id
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(frame, f"{tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    # increment missed counters & remove stale tracks
    to_del = []
    for track_id, tdata in tracks.items():
        if tdata['last_frame'] != frame_idx:
            tdata['missed'] += 1
        if tdata['missed'] > MAX_MISSED_FRAMES:
            to_del.append(track_id)
    for tid in to_del:
        del tracks[tid]

    # Overlay DB summary on frame
    y = 30
    for tid, data in tiger_db.items():
        text = f"{tid}: seen={data['count']}"
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        y += 22

    cv2.imshow("Tiger Re-ID (robust)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
