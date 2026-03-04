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
# Config (tune these)
# ----------------------------
MODEL_PATH = "best_enlightengan_and_yolov8.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONF_DETECT = 0.50           # YOLO detection confidence threshold
MIN_CROP_SIZE = 60           # ignore tiny crops
MIN_AREA_RATIO = 0.001       # ignore extremely small boxes relative to frame area
MAX_ASPECT = 3.0             # ignore very tall/flat boxes

SIM_HIGH = 0.93              # if any stored embedding >= SIM_HIGH -> definitely same tiger
SIM_LOW = 0.78               # if between SIM_LOW and SIM_HIGH -> require ORB confirmation
ORB_MATCH_THRESHOLD = 30     # ORB match count to confirm ambiguous case
MIN_ORB_KEYPOINTS = 40       # require this many ORB keypoints in crop to consider it textured

SAMPLES_PER_TIGER = 6        # max stored samples (images) per tiger
EMB_DIST_ADD_THRESHOLD = 0.06  # add new sample if embedding distance from samples > this
SAVE_EVERY_N_MATCHES = 8     # fallback: save one sample every N matches if diversity low

TRACK_DIST_THRESHOLD = 80
MAX_MISSED_FRAMES = 10

# ----------------------------
# Load models & features
# ----------------------------
print("Loading models...")
yolo = YOLO(MODEL_PATH)

resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(DEVICE).eval()

orb = cv2.ORB_create(nfeatures=800)
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

# tiger_db: {tid: {"embeddings":[emb...], "samples":[path...], "count":n, "matches_since_last_save":k}}
tiger_db = {}
tiger_count = 0

tracks = {}
next_track_id = 0

# ----------------------------
# Helpers
# ----------------------------
def get_embedding_from_crop(bgr_crop):
    img = Image.fromarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = resnet(tensor).cpu().numpy().reshape(-1)
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm

def cosine_sim(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def emb_distance(a, b):
    return float(np.linalg.norm(a - b))

def orb_keypoints_count(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k, d = orb.detectAndCompute(gray, None)
    return 0 if d is None else len(k)

def orb_good_matches(img1, img2, dist_thresh=60):
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
    good = [m for m in matches if m.distance < dist_thresh]
    return len(good)

def is_valid_crop(x1,y1,x2,y2, frame_w, frame_h):
    w = x2 - x1; h = y2 - y1
    if w <= 0 or h <= 0:
        return False
    if w < MIN_CROP_SIZE or h < MIN_CROP_SIZE:
        return False
    if (w * h) < (MIN_AREA_RATIO * frame_w * frame_h):
        return False
    ar = max(w/h, h/w)
    if ar > MAX_ASPECT:
        return False
    return True

# add embedding and maybe sample to tiger record
def add_embedding_and_maybe_sample(tid, emb, crop):
    """Add embedding to tiger tid and add a sample if diversity or periodic condition met."""
    data = tiger_db[tid]
    # compute min distance to existing embeddings
    if len(data["embeddings"]) == 0:
        min_dist = float('inf')
    else:
        dists = [emb_distance(emb, e) for e in data["embeddings"]]
        min_dist = float(min(dists))

    added = False
    if (len(data["embeddings"]) < SAMPLES_PER_TIGER) and (min_dist > EMB_DIST_ADD_THRESHOLD):
        data["embeddings"].append(emb.copy())
        added = True
    else:
        # fallback periodic save if diversity not high but we want more samples
        if (len(data["samples"]) < SAMPLES_PER_TIGER) and (data.get("matches_since_last_save",0) >= SAVE_EVERY_N_MATCHES):
            # ensure sample adds some value (not exact duplicate)
            data["embeddings"].append(emb.copy())
            added = True

    # cap embeddings
    if len(data["embeddings"]) > SAMPLES_PER_TIGER:
        data["embeddings"] = data["embeddings"][-SAMPLES_PER_TIGER:]

    # save sample image when we added a new embedding and sample count not exceeded
    if added and len(data["samples"]) < SAMPLES_PER_TIGER:
        fname = os.path.join("tiger_database", f"{tid}_{int(time.time())}.jpg")
        cv2.imwrite(fname, crop)
        data["samples"].append(fname)
        data["matches_since_last_save"] = 0
        print(f"  -> Saved sample for {tid}, total samples: {len(data['samples'])}")
    else:
        # increment counter
        data["matches_since_last_save"] = data.get("matches_since_last_save",0) + 1

# Compare embedding against DB and decide new/existing
def identify_tiger_multi(emb, crop):
    global tiger_count, tiger_db
    if not tiger_db:
        tiger_count += 1
        tid = f"Tiger_{tiger_count}"
        tiger_db[tid] = {"embeddings":[emb.copy()], "samples":[], "count":1, "matches_since_last_save":0}
        # save initial sample
        fname = os.path.join("tiger_database", f"{tid}_{int(time.time())}.jpg")
        cv2.imwrite(fname, crop)
        tiger_db[tid]["samples"].append(fname)
        print(f"[NEW] Registered {tid} (initial sample)")
        return tid, True, 0.0, 0

    best_sim = -1.0
    best_tid = None
    for tid, data in tiger_db.items():
        sims = [cosine_sim(emb, e) for e in data["embeddings"]]
        s = max(sims)
        if s > best_sim:
            best_sim = s
            best_tid = tid

    # clear match
    if best_sim >= SIM_HIGH:
        # update count and maybe add sample/embedding
        tiger_db[best_tid]["count"] += 1
        add_embedding_and_maybe_sample(best_tid, emb, crop)
        return best_tid, False, best_sim, 0

    # clearly new
    if best_sim < SIM_LOW:
        tiger_count += 1
        tid = f"Tiger_{tiger_count}"
        tiger_db[tid] = {"embeddings":[emb.copy()], "samples":[], "count":1, "matches_since_last_save":0}
        fname = os.path.join("tiger_database", f"{tid}_{int(time.time())}.jpg")
        cv2.imwrite(fname, crop)
        tiger_db[tid]["samples"].append(fname)
        print(f"[NEW] Registered {tid} (sim={best_sim:.3f})")
        return tid, True, best_sim, 0

    # ambiguous -> ORB across samples
    best_orb = 0
    best_orb_tid = None
    for tid, data in tiger_db.items():
        for sample_path in data["samples"]:
            sample_img = cv2.imread(sample_path)
            if sample_img is None:
                continue
            matches = orb_good_matches(crop, sample_img)
            if matches > best_orb:
                best_orb = matches
                best_orb_tid = tid

    if best_orb_tid is not None and best_orb >= ORB_MATCH_THRESHOLD:
        tiger_db[best_orb_tid]["count"] += 1
        add_embedding_and_maybe_sample(best_orb_tid, emb, crop)
        return best_orb_tid, False, best_sim, best_orb

    # otherwise new
    tiger_count += 1
    tid = f"Tiger_{tiger_count}"
    tiger_db[tid] = {"embeddings":[emb.copy()], "samples":[], "count":1, "matches_since_last_save":0}
    fname = os.path.join("tiger_database", f"{tid}_{int(time.time())}.jpg")
    cv2.imwrite(fname, crop)
    tiger_db[tid]["samples"].append(fname)
    print(f"[NEW] Registered {tid} (ambiguous sim={best_sim:.3f}, orb={best_orb})")
    return tid, True, best_sim, best_orb

# ----------------------------
# Centroid tracker helpers
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
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

frame_idx = 0
print("Ready. Showing live feed...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame_idx += 1
    h, w = frame.shape[:2]

    # YOLO detection
    results = yolo(frame, conf=CONF_DETECT)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if yolo.names[cls_id] != "tiger":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            if not is_valid_crop(x1,y1,x2,y2, w, h):
                continue
            detections.append({'box':(x1,y1,x2,y2), 'conf':conf})

    det_centroids = [bbox_centroid(d['box']) for d in detections]
    used_dets = set()
    assigned_tracks = set()

    # match detections to tracks
    for di, centroid in enumerate(det_centroids):
        best_tid = None; best_dist = 1e9
        for track_id, tdata in tracks.items():
            dist = euclidean(centroid, tdata['centroid'])
            if dist < best_dist and dist < TRACK_DIST_THRESHOLD:
                best_dist = dist; best_tid = track_id
        if best_tid is not None:
            tracks[best_tid]['centroid'] = centroid
            tracks[best_tid]['missed'] = 0
            tracks[best_tid]['last_frame'] = frame_idx
            assigned_tracks.add(best_tid)
            used_dets.add(di)
            det = detections[di]
            x1,y1,x2,y2 = det['box']
            crop = frame[y1:y2, x1:x2]
            emb = get_embedding_from_crop(crop)
            tid = tracks[best_tid]['tiger_id']
            add_embedding_and_maybe_sample(tid, emb, crop)
            tiger_db[tid]["count"] += 1
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # unmatched detections -> identify
    for di, det in enumerate(detections):
        if di in used_dets: continue
        x1,y1,x2,y2 = det['box']
        crop = frame[y1:y2, x1:x2]

        # texture check
        kp_count = orb_keypoints_count(crop)
        if kp_count < MIN_ORB_KEYPOINTS:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 1)
            cv2.putText(frame, f"FP?", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
            continue

        emb = get_embedding_from_crop(crop)
        tid, is_new, best_sim, best_orb = identify_tiger_multi(emb, crop)
 
        if is_new:
            print(f"[Frame {frame_idx}] NEW {tid} (sim={best_sim:.3f}, orb={best_orb})")
        else:
            print(f"[Frame {frame_idx}] MATCH {tid} (sim={best_sim:.3f}, orb={best_orb})")

        # create new track
        track_id = next_track_id
        next_track_id += 1
        centroid = bbox_centroid((x1,y1,x2,y2))
        tracks[track_id] = {"centroid":centroid, "tiger_id":tid, "missed":0, "last_frame":frame_idx}
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(frame, f"{tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    # cleanup stale tracks
    to_del = []
    for track_id, tdata in tracks.items():
        if tdata['last_frame'] != frame_idx:
            tdata['missed'] += 1
        if tdata['missed'] > MAX_MISSED_FRAMES:
            to_del.append(track_id)
    for tr in to_del:
        del tracks[tr]

    # overlay DB summary
    y0 = 30
    for tid, data in tiger_db.items():
        txt = f"{tid}: seen={data['count']} samples={len(data['samples'])}"
        cv2.putText(frame, txt, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        y0 += 22

    cv2.imshow("Tiger Re-ID (multi-sample, robust)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
