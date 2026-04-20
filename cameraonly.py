#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import pipeline
import os


# In[2]:


from sampleframes import load_image, load_labels, BASE_PATH

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.7
IOU_THRESHOLD = 0.5
EVAL_FRAMES = 50       # number of frames to evaluate precision/recall over

# Map COCO class names → KITTI class names
COCO_TO_KITTI = {
    "car":      "Car",
    "truck":    "Car",        # trucks are labeled as Car in KITTI
    "person":   "Pedestrian",
    "bicycle":  "Cyclist",
}


# In[3]:


# Load DETR model from HuggingFace
def load_detector():
    """
    Loads facebook/detr-resnet-50 via HuggingFace pipeline.
    'object-detection' pipeline handles preprocessing + postprocessing for us.
    We load it once and reuse across all frames.
    """
    print("Loading DETR model from HuggingFace (first run downloads weights ~160MB)...")
    detector = pipeline(
        "object-detection",
        model="facebook/detr-resnet-50",
        device=-1       # -1 = CPU; change to 0 if you have a GPU
    )
    print("Model loaded.\n")
    return detector


# In[4]:


#Run inference on a single frame
def detect_objects(detector, frame_id):
    """
    Runs DETR on one KITTI frame.
    Returns a list of dicts, each with keys:
        label      : KITTI class name (Car / Pedestrian / Cyclist)
        score      : confidence score (0–1)
        box        : [left, top, right, bottom] in pixels
    """
    img_path = os.path.join(BASE_PATH, "data_object_image_2/training/", "image_2", f"{frame_id}.png")
    pil_img  = Image.open(img_path).convert("RGB")   # DETR expects a PIL image

    raw_preds = detector(pil_img)

    detections = []
    for pred in raw_preds:
        coco_label = pred["label"].lower()

        # Skip classes irrelevant to driving
        if coco_label not in COCO_TO_KITTI:
            continue

        # Skip low-confidence predictions
        if pred["score"] < CONFIDENCE_THRESHOLD:
            continue

        b = pred["box"]   # dict with keys: xmin, ymin, xmax, ymax
        detections.append({
            "label": COCO_TO_KITTI[coco_label],
            "score": pred["score"],
            "box":   [b["xmin"], b["ymin"], b["xmax"], b["ymax"]],
        })

    return detections


# In[5]:


# Visualize predictions vs ground truth
def visualize_detections(frame_id, detections, gt_df):
    """
    Draws two sets of boxes on the camera image:
      GREEN  = DETR predictions
      RED    = KITTI ground truth
    Labels above each box show class + depth (GT only) or confidence (pred).
    """
    img = load_image(frame_id)
    fig, ax = plt.subplots(1, figsize=(14, 5))
    ax.imshow(img)

    # — Ground truth boxes (red) —
    for _, row in gt_df.iterrows():
        if row["type"] not in ["Car", "Pedestrian", "Cyclist"]:
            continue
        x1, y1, x2, y2 = row["left"], row["top"], row["right"], row["bottom"]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=2, edgecolor="red", facecolor="none", linestyle="--")
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"GT:{row['type']} {row['z_3d']:.1f}m",
                color="red", fontsize=7, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.3, pad=1))

    # — DETR predictions (green) —
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y2 + 10, f"PRED:{det['label']} {det['score']:.2f}",
                color="lime", fontsize=7, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.3, pad=1))

    ax.set_title(f"Frame {frame_id}  |  RED = Ground Truth   GREEN = DETR Prediction")
    ax.axis("off")
    plt.tight_layout()
    plt.show()



# In[6]:


#IoU computation & precision / recall evaluation

def compute_iou(box_a, box_b):
    """
    Computes Intersection over Union between two boxes.
    Each box is [left, top, right, bottom].
    """
    # Intersection rectangle
    inter_left   = max(box_a[0], box_b[0])
    inter_top    = max(box_a[1], box_b[1])
    inter_right  = min(box_a[2], box_b[2])
    inter_bottom = min(box_a[3], box_b[3])

    inter_w = max(0, inter_right  - inter_left)
    inter_h = max(0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    return inter_area / (area_a + area_b - inter_area)


def evaluate(detector, num_frames=EVAL_FRAMES):
    """
    Runs detection over num_frames frames and computes:
    """
    # Get all available frame IDs
    all_frames = sorted([
        f.replace(".png", "")
        for f in os.listdir(os.path.join(BASE_PATH, "data_object_image_2/training/", "image_2"))
        if f.endswith(".png")
    ])[:num_frames]

    tp = fp = fn = 0
    iou_scores = []

    for frame_id in all_frames:
        detections = detect_objects(detector, frame_id)
        gt_df      = load_labels(frame_id)
        gt_df      = gt_df[gt_df["type"].isin(["Car", "Pedestrian", "Cyclist"])]

        matched_gt = set()   # track which GT boxes have been matched

        for det in detections:
            best_iou   = 0.0
            best_gt_idx = -1

            for idx, gt_row in gt_df.iterrows():
                if gt_row["type"] != det["label"]:
                    continue                          # class mismatch — skip
                iou = compute_iou(det["box"],
                                  [gt_row["left"], gt_row["top"],
                                   gt_row["right"], gt_row["bottom"]])
                if iou > best_iou:
                    best_iou    = iou
                    best_gt_idx = idx

            if best_iou >= IOU_THRESHOLD and best_gt_idx not in matched_gt:
                tp += 1                               # correct detection
                matched_gt.add(best_gt_idx)
                iou_scores.append(best_iou)
            else:
                fp += 1                               # predicted something that wasn't there

        fn += len(gt_df) - len(matched_gt)            # GT objects we never found

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    mean_iou  = np.mean(iou_scores) if iou_scores else 0

    print("=" * 50)
    print(f"  Camera-Only Baseline Results ({num_frames} frames)")
    print("=" * 50)
    print(f"  True Positives  (TP) : {tp}")
    print(f"  False Positives (FP) : {fp}")
    print(f"  False Negatives (FN) : {fn}")
    print(f"  Precision            : {precision:.3f}  ({precision*100:.1f}%)")
    print(f"  Recall               : {recall:.3f}  ({recall*100:.1f}%)")
    print(f"  Mean IoU (TP only)   : {mean_iou:.3f}")
    print("=" * 50)

    return {"precision": precision, "recall": recall, "mean_iou": mean_iou,
            "tp": tp, "fp": fp, "fn": fn}


# In[7]:


detector = load_detector()

# Visualize one sample frame
SAMPLE_FRAME = "000010"
detections   = detect_objects(detector, SAMPLE_FRAME)
gt_df        = load_labels(SAMPLE_FRAME)
visualize_detections(SAMPLE_FRAME, detections, gt_df)

print(f"\nDetections in frame {SAMPLE_FRAME}:")
for d in detections:
    print(f"  {d['label']:12s}  confidence={d['score']:.3f}  box={[round(v) for v in d['box']]}")

# Run evaluation over 50 frames
print("\nRunning evaluation over 50 frames (this takes a few minutes on CPU)...")
baseline_results = evaluate(detector, num_frames=EVAL_FRAMES)


# In[ ]:




