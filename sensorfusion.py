#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import os


# In[2]:


from sampleframes import load_image, load_lidar, load_labels, load_calib, BASE_PATH

# ── CONFIG ────────────────────────────────────────────────────────────────────
FRAME_ID      = "000010"
MAX_DEPTH     = 60.0     # metres — clip depth values beyond this for display


# In[3]:


# =============================================================================
# COMPONENT 1 — Coordinate transformation pipeline
# =============================================================================
def build_projection_matrix(calib):
    """
    Builds the full 3D→2D projection matrix by chaining three calibration matrices.

    The transformation pipeline is:
      LiDAR coords  →[Tr_velo_to_cam]→  Camera 3D coords
                    →[R0_rect]        →  Rectified camera 3D coords
                    →[P2]             →  2D image pixel coords

    Mathematically: pixel = P2 @ R0_rect @ Tr_velo_to_cam @ lidar_point

    Each matrix reshaped to:
      Tr_velo_to_cam : (3, 4)  — rotation + translation from LiDAR to camera
      R0_rect        : (4, 4)  — 3×3 rectification padded to 4×4
      P2             : (3, 4)  — camera intrinsic projection matrix
    """
    Tr = calib["Tr_velo_to_cam"].reshape(3, 4)       # LiDAR → camera coords
    R0 = calib["R0_rect"].reshape(3, 3)               # rectification rotation

    # Pad R0 to 4×4 homogeneous matrix (add identity row/col)
    R0_hom = np.eye(4)
    R0_hom[:3, :3] = R0

    # Pad Tr to 4×4 homogeneous matrix
    Tr_hom = np.vstack([Tr, [0, 0, 0, 1]])

    P2 = calib["P2"].reshape(3, 4)                    # camera → pixel

    # Chain: P2 @ R0 @ Tr  (apply right to left)
    proj_matrix = P2 @ R0_hom @ Tr_hom               # shape: (3, 4)
    return proj_matrix


def project_lidar_to_image(points, proj_matrix, img_shape):
    """
    Projects LiDAR 3D points onto the 2D camera image plane.

    Steps:
      1. Convert points to homogeneous coordinates (add column of 1s)
      2. Multiply by projection matrix → get (u, v, w) for each point
      3. Divide by w (perspective division) → get pixel coords (u/w, v/w)
      4. Filter out points that fall outside the image boundaries
         or are behind the camera (w <= 0)

    Returns:
      u_valid   : pixel column coords (x) for valid points
      v_valid   : pixel row coords    (y) for valid points
      depth     : distance in metres for each valid point
      valid_idx : indices into original points array
    """
    H, W = img_shape[:2]
    N    = points.shape[0]

    # Step 1 — homogeneous coords: (N, 4) with last column = 1
    pts_hom = np.hstack([points[:, :3], np.ones((N, 1))])   # (N, 4)

    # Step 2 — project: (3, 4) @ (4, N) → (3, N)
    pts_2d = proj_matrix @ pts_hom.T                         # (3, N)

    # Step 3 — perspective division
    w     = pts_2d[2, :]          # depth in camera coords
    u     = pts_2d[0, :] / w     # pixel column (x)
    v     = pts_2d[1, :] / w     # pixel row    (y)

    # Step 4 — filter: must be in front of camera and inside image bounds
    valid = (
        (w > 0) &                  # in front of camera
        (u >= 0) & (u < W) &       # within image width
        (v >= 0) & (v < H)         # within image height
    )

    return u[valid], v[valid], w[valid], np.where(valid)[0]



# In[4]:


# =============================================================================
# COMPONENT 2 — Build depth map
# =============================================================================
def build_depth_map(img_shape, u, v, depth):
    """
    Creates a dense-ish depth map by painting projected LiDAR depths
    onto a blank canvas at the corresponding pixel locations.

    The depth map is sparse — only pixels where a LiDAR beam landed
    have a value. Everything else stays 0 (unknown depth).

    Shape: (H, W) — same spatial dimensions as the camera image.
    Value: depth in metres at each pixel (0 = no LiDAR return here).
    """
    H, W     = img_shape[:2]
    depth_map = np.zeros((H, W), dtype=np.float32)

    u_int = u.astype(int)
    v_int = v.astype(int)

    # Where multiple points project to same pixel, keep the closest (min depth)
    # Process in reverse depth order so closer points overwrite farther ones
    order = np.argsort(depth)[::-1]        # farthest first
    depth_map[v_int[order], u_int[order]] = depth[order]

    coverage = (depth_map > 0).sum() / (H * W) * 100
    print(f"Depth map coverage: {coverage:.2f}% of pixels have a LiDAR return")
    return depth_map


# In[5]:


# =============================================================================
# COMPONENT 3 — Build fused RGBD image
# =============================================================================
def build_rgbd(img, depth_map):
    """
    Concatenates the RGB image with the depth map to create a 4-channel
    RGBD representation.

    The depth channel is normalized to [0, 1] so it's on the same scale
    as the RGB channels (which are already 0–1 from plt.imread).

    Shape: (H, W, 4) — [Red, Green, Blue, NormalizedDepth]

    This 4-channel tensor is what would be passed into a fusion-aware
    CNN in a full deep learning pipeline.
    """
    depth_norm = np.clip(depth_map / MAX_DEPTH, 0, 1)     # normalize to [0,1]
    rgbd = np.dstack([img, depth_norm])                    # (H, W, 4)
    print(f"RGBD tensor shape: {rgbd.shape}  →  H={rgbd.shape[0]}, W={rgbd.shape[1]}, C={rgbd.shape[2]}")
    return rgbd



# In[6]:


# =============================================================================
# COMPONENT 4 — Depth-aware detection filter
# =============================================================================
def filter_detections_by_depth(detections, depth_map, depth_tolerance=5.0):
    """
    Enhances each DETR detection with depth information from the LiDAR.

    For each predicted bounding box:
      1. Sample all non-zero depth pixels inside the box
      2. Take the median depth as the estimated object distance
      3. Flag detections where LiDAR has insufficient coverage (< 5% of box)
         as 'depth_uncertain' — these are likely false positives

    This is the fusion payoff: camera gives us the box, LiDAR gives us
    the depth — together we get a pseudo-3D detection.

    Returns enhanced detections with added keys:
      estimated_depth  : median LiDAR depth inside the box (metres)
      depth_coverage   : fraction of box pixels with LiDAR returns
      depth_uncertain  : True if LiDAR coverage is too sparse to trust
    """
    enhanced = []
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(depth_map.shape[1]-1, x2), min(depth_map.shape[0]-1, y2)

        # Crop depth map to bounding box region
        box_depth = depth_map[y1:y2, x1:x2]
        valid_depths = box_depth[box_depth > 0]

        box_pixels   = (y2 - y1) * (x2 - x1)
        coverage     = len(valid_depths) / box_pixels if box_pixels > 0 else 0

        est_depth        = float(np.median(valid_depths)) if len(valid_depths) > 0 else -1.0
        depth_uncertain  = coverage < 0.05     # less than 5% of box has LiDAR

        enhanced.append({
            **det,
            "estimated_depth": est_depth,
            "depth_coverage":  coverage,
            "depth_uncertain": depth_uncertain,
        })
    return enhanced


# In[7]:


# =============================================================================
# COMPONENT 5 — Visualizations
# =============================================================================
def plot_projection_overlay(img, u, v, depth):
    """Overlays projected LiDAR points on the camera image, colored by depth."""
    fig, ax = plt.subplots(1, figsize=(14, 5))
    ax.imshow(img)
    sc = ax.scatter(u, v, c=depth, cmap="plasma_r",
                    s=1.5, alpha=0.7, vmin=0, vmax=MAX_DEPTH)
    plt.colorbar(sc, ax=ax, label="Depth (metres)")
    ax.set_title("LiDAR Points Projected onto Camera Image — Colored by Depth")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_depth_map(depth_map):
    """Visualizes the sparse depth map on its own."""
    fig, ax = plt.subplots(1, figsize=(14, 5))
    masked = np.ma.masked_where(depth_map == 0, depth_map)   # mask empty pixels
    im = ax.imshow(masked, cmap="plasma_r", vmin=0, vmax=MAX_DEPTH)
    plt.colorbar(im, ax=ax, label="Depth (metres)")
    ax.set_title("Sparse LiDAR Depth Map (black = no LiDAR return)")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_fusion_detections(img, enhanced_detections, gt_df):
    """
    Shows fusion-enhanced detections on the image.
    GREEN box = detection with good LiDAR depth coverage
    ORANGE box = detection flagged as depth-uncertain (likely false positive)
    RED dashed = ground truth
    """
    fig, ax = plt.subplots(1, figsize=(14, 5))
    ax.imshow(img)

    # Ground truth
    for _, row in gt_df.iterrows():
        if row["type"] not in ["Car", "Pedestrian", "Cyclist"]:
            continue
        x1, y1, x2, y2 = row["left"], row["top"], row["right"], row["bottom"]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=2, edgecolor="red",
                                  facecolor="none", linestyle="--")
        ax.add_patch(rect)
        ax.text(x1, y1-5, f"GT {row['z_3d']:.1f}m",
                color="red", fontsize=7, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.3, pad=1))

    # Fusion detections
    for det in enhanced_detections:
        x1, y1, x2, y2 = det["box"]
        color = "orange" if det["depth_uncertain"] else "lime"
        rect  = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        depth_str = f"{det['estimated_depth']:.1f}m" if det["estimated_depth"] > 0 else "no depth"
        label_str = f"{det['label']} ~{depth_str} {'⚠' if det['depth_uncertain'] else ''}"
        ax.text(x1, y2+10, label_str, color=color, fontsize=7, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.3, pad=1))

    ax.set_title("Fusion Detections  |  RED=GT   GREEN=confident   ORANGE=depth uncertain")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_rgbd_channels(rgbd):
    """Shows all 4 channels of the RGBD tensor side by side."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    titles  = ["Red channel", "Green channel", "Blue channel", "Depth channel (normalized)"]
    cmaps   = ["Reds", "Greens", "Blues", "plasma"]
    for i, (ax, title, cmap) in enumerate(zip(axes, titles, cmaps)):
        ax.imshow(rgbd[:, :, i], cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.suptitle("RGBD Tensor — 4 Channels", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# In[10]:


print(f"\n{'='*50}")
print(f"  Phase 4 — Sensor Fusion  |  Frame {FRAME_ID}")
print(f"{'='*50}\n")

# Load data
img    = load_image(FRAME_ID)
points = load_lidar(FRAME_ID)
calib  = load_calib(FRAME_ID)
gt_df  = load_labels(FRAME_ID)

# Component 1 — build projection matrix & project LiDAR → image
proj   = build_projection_matrix(calib)
u, v, depth, valid_idx = project_lidar_to_image(points, proj, img.shape)
print(f"Projected {len(u):,} of {len(points):,} LiDAR points onto image plane\n")

# Component 2 — depth map
depth_map = build_depth_map(img.shape, u, v, depth)

# Component 3 — RGBD tensor
rgbd = build_rgbd(img, depth_map)

# Visualizations
plot_projection_overlay(img, u, v, depth)
plot_depth_map(depth_map)
plot_rgbd_channels(rgbd)

# Component 4 — load Phase 2 detections and enhance with depth
from cameraonly import load_detector, detect_objects
detector     = load_detector()
detections   = detect_objects(detector, FRAME_ID)
enhanced     = filter_detections_by_depth(detections, depth_map)

print("\nFusion-enhanced detections:")
print(f"{'Label':12s} {'Conf':>6s} {'Est.Depth':>10s} {'Coverage':>10s} {'Uncertain':>10s}")
print("-" * 55)
for d in enhanced:
    print(f"{d['label']:12s} {d['score']:>6.3f} "
            f"{d['estimated_depth']:>9.1f}m "
            f"{d['depth_coverage']:>9.1%} "
            f"{'YES ⚠' if d['depth_uncertain'] else 'no':>10s}")

plot_fusion_detections(img, enhanced, gt_df)

# Save RGBD for use in Phase 5
np.save(f"rgbd_{FRAME_ID}.npy", rgbd)


# In[ ]:




