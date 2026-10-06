r"""
ThirdWheel AI: Real-Time Object Relationship Prediction & Scene Graph Engine
Supports: Live Webcam / USB Cameras, Image Files, and Video Streams.
Always watching. Figuring out what objects do.
"""

import argparse
import sys
import time
import os
import numpy as np
import torch
from PIL import Image
import cv2

from relsgg import RelateAnything

# Color palette for distinct entities (BGR format)
PALETTE = [
    (255, 128, 0),   # Vibrant Cyan/Blue
    (0, 215, 255),   # Golden Yellow
    (50, 205, 50),   # Lime Green
    (238, 130, 238), # Violet
    (0, 165, 255),   # Bright Orange
    (255, 105, 180), # Hot Pink
    (0, 255, 255),   # Bright Yellow
    (147, 20, 255),  # Deep Pink
]

def draw_hud(frame, text_lines, x=15, y=25, line_h=22):
    """Draw a clean semi-transparent HUD overlay with text lines."""
    h, w = frame.shape[:2]
    max_len = max(len(line) for line in text_lines) if text_lines else 10
    hud_w = min(max_len * 9 + 30, w - 30)
    hud_h = len(text_lines) * line_h + 15

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - 18), (x + hud_w, y + hud_h - 18), (15, 18, 24), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x - 5, y - 18), (x + hud_w, y + hud_h - 18), (60, 70, 85), 1)

    for i, line in enumerate(text_lines):
        cv2.putText(frame, line, (x, y + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 245, 250), 1, cv2.LINE_AA)

def draw_scene_graph(frame, boxes, labels, masks, triplets, fps=None, device="cuda"):
    """Draw masks, bounding boxes, labels, and relational arrows onto the frame."""
    H, W = frame.shape[:2]
    annotated = frame.copy()

    # 1. Overlay translucent masks
    if masks is not None and len(masks) > 0:
        mask_overlay = np.zeros_like(frame)
        for i, m in enumerate(masks):
            color = PALETTE[i % len(PALETTE)]
            mask_overlay[m > 0] = color
        cv2.addWeighted(mask_overlay, 0.35, annotated, 0.65, 0, annotated)

    # 2. Draw Bounding Boxes and Entity Labels
    centers = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        color = PALETTE[i % len(PALETTE)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        name = labels[i] if i < len(labels) else f"obj_{i}"
        label_text = f"{name} #{i}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(y1 - th - 6, 0)), (x1 + tw + 6, max(y1, th + 6)), color, -1)
        cv2.putText(annotated, label_text, (x1 + 3, max(y1 - 3, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))

    # 3. Draw Relational Directed Arrows
    for idx, t in enumerate(triplets):
        s_idx, o_idx = t.subject_idx, t.object_idx
        if s_idx >= len(centers) or o_idx >= len(centers) or s_idx == o_idx:
            continue

        c1 = centers[s_idx]
        c2 = centers[o_idx]

        # Draw connecting arrow
        arrow_color = (0, 240, 255) if idx == 0 else (120, 200, 255)
        cv2.arrowedLine(annotated, c1, c2, arrow_color, 2, tipLength=0.06)

        # Label along midpoint
        mid_x = int((c1[0] * 0.45) + (c2[0] * 0.55))
        mid_y = int((c1[1] * 0.45) + (c2[1] * 0.55))

        rel_text = f"{t.predicate} [{t.score:.2f}]"
        (rw, rh), _ = cv2.getTextSize(rel_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        cv2.rectangle(annotated, (mid_x - 3, mid_y - rh - 4), (mid_x + rw + 3, mid_y + 3), (20, 25, 30), -1)
        cv2.rectangle(annotated, (mid_x - 3, mid_y - rh - 4), (mid_x + rw + 3, mid_y + 3), arrow_color, 1)
        cv2.putText(annotated, rel_text, (mid_x, mid_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. HUD / Info Ticker
    hud_lines = [
        f"ThirdWheel AI - Object Relationship Engine [{device.upper()}]",
        f"FPS: {fps:.1f}" if fps is not None else "Image Analysis",
        f"Entities Tracked: {len(boxes)} | Active Relationships: {len(triplets)}"
    ]
    draw_hud(annotated, hud_lines, x=15, y=25)

    # Bottom ticker with top relationships
    if triplets:
        ticker = " | ".join([f"({t.subject_label})--{t.predicate} [{t.score:.2f}]-->({t.object_label})" for t in triplets[:3]])
        h, w = frame.shape[:2]
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 18, 24), -1)
        cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)
        cv2.putText(annotated, f"Live Interactions: {ticker}", (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    return annotated


def process_frame(frame, detector, model, device="cuda", topk=5):
    """Run detection, segmentation, and relation prediction on a single BGR frame."""
    H, W = frame.shape[:2]

    # Instance segmentation (YOLO-Seg)
    det_res = detector.predict(frame, device=device, conf=0.25, verbose=False)[0]

    boxes, labels, masks = None, [], None
    if det_res.masks is not None and len(det_res.boxes) > 0:
        boxes = det_res.boxes.xyxy.cpu().numpy()
        class_ids = det_res.boxes.cls.cpu().numpy().astype(int)
        labels = [det_res.names[c] for c in class_ids]
        raw_masks = det_res.masks.data.cpu().numpy()
        masks = np.stack([
            cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
            for m in raw_masks
        ])

    triplets = []
    if boxes is not None and len(boxes) >= 2:
        # Convert BGR frame to RGB PIL for relation model
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        triplets = model.predict(pil_img, boxes, masks=masks, box_labels=labels, topk=topk)
    elif boxes is None:
        boxes = np.empty((0, 4), dtype=np.float32)

    return boxes, labels, masks, triplets


def run_camera(cam_id, detector, model, device="cuda", topk=5, headless=False):
    """Run live webcam stream with real-time scene graph overlay."""
    print(f"\n[INFO] Opening Camera ID {cam_id}...")
    cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera with index {cam_id}.")
        print("  - Check camera connections")
        print("  - Try camera index 0, 1, 2, or 3")
        print("  - On Linux, check: v4l2-ctl --list-devices")
        return

    # Set camera resolution to 1280x720 if supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = f"ThirdWheel AI - Live Camera Feed (Camera #{cam_id})"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"[SUCCESS] Camera #{cam_id} active! Press 'q' or ESC in the video window to stop.\n")

    prev_time = time.time()
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame grab failed, exiting stream.")
                break

            now = time.time()
            elapsed = max(now - prev_time, 1e-5)
            fps = 1.0 / elapsed
            prev_time = now

            boxes, labels, masks, triplets = process_frame(frame, detector, model, device=device, topk=topk)
            annotated = draw_scene_graph(frame, boxes, labels, masks, triplets, fps=fps, device=device)

            if not headless:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: # 27 is ESC
                    print("\n[INFO] Exit key pressed. Closing camera feed...")
                    break
            else:
                # In headless mode print console updates periodically
                if int(now) % 3 == 0:
                    print(f"[LIVE] FPS: {fps:.1f} | Objects: {len(boxes)} | Relations: {len(triplets)}")

    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()


def run_image(image_path, detector, model, device="cuda", topk=5, headless=False, save=True, output_path="thirdwheel_output.jpg"):
    """Analyze a single image file."""
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        print(f"  - Check the path exists and is accessible")
        print(f"  - Try using an absolute path")
        return

    print(f"\n[INFO] Processing image: {image_path} ...")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not decode image: {image_path}")
        print(f"  - File may be corrupted or unsupported format")
        print(f"  - Supported formats: JPG, PNG, BMP, TIFF")
        return

    t0 = time.perf_counter()
    boxes, labels, masks, triplets = process_frame(frame, detector, model, device=device, topk=topk)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"\n[RESULT] Processed in {latency_ms:.1f} ms:")
    print(f" Entities detected ({len(boxes)}): {labels}")
    print(f" Predicted Relationships:")
    if triplets:
        for t in triplets:
            print(f"   {t}")
    else:
        print("   (Need at least 2 interacting objects to form relationships)")

    annotated = draw_scene_graph(frame, boxes, labels, masks, triplets, fps=None, device=device)

    if save:
        try:
            cv2.imwrite(output_path, annotated)
            print(f"\n[SAVED] Annotated scene graph saved to: {output_path}")
        except Exception as e:
            print(f"[WARNING] Could not save output image: {e}")

    if not headless:
        window_name = "ThirdWheel AI - Analyzed Image (Press any key to close)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(video_path, detector, model, device="cuda", topk=5, headless=False):
    """Process a recorded video file."""
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    print(f"\n[INFO] Opening video: {video_path} ...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        print(f"  - File may be corrupted or use an unsupported codec")
        print(f"  - Try: ffmpeg -i input.mp4 -c:v libx264 output.mp4")
        return

    window_name = "ThirdWheel AI - Video Stream (Press 'q' or ESC to stop)"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("\n[INFO] Video playback complete.")
                break

            now = time.time()
            elapsed = max(now - prev_time, 1e-5)
            fps = 1.0 / elapsed
            prev_time = now

            boxes, labels, masks, triplets = process_frame(frame, detector, model, device=device, topk=topk)
            annotated = draw_scene_graph(frame, boxes, labels, masks, triplets, fps=fps, device=device)

            if not headless:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n[INFO] Stopped video playback.")
                    break
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="ThirdWheel AI: Real-Time Object Relationship Prediction")
    parser.add_argument("--mode", default="image", choices=["camera", "image", "video"], help="Input source mode")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (0, 1, 2, 3...)")
    parser.add_argument("--image", default="assets/reel/images/horse.jpg", help="Path to input image")
    parser.add_argument("--video", default="", help="Path to input video file")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Compute device (default: auto)")
    parser.add_argument("--topk", type=int, default=5, help="Top relationships to track per pair")
    parser.add_argument("--headless", action="store_true", help="Disable GUI window popup")
    parser.add_argument("--output", default="thirdwheel_output.jpg", help="Path to save annotated output image")
    parser.add_argument("--model", default="maelic/relsgg-vits16plus", help="Model checkpoint (recommended: maelic/relsgg-vits16plus)")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARNING] CUDA requested but not available. Falling back to CPU.")
            device = "cpu"

    print("=" * 65)
    print("        THIRDWHEEL AI - Object Relationship Analyzer")
    print("        Always watching. Figuring out what objects do.")
    print("=" * 65)
    print(" Neural Engine: Vision Transformer Base (High-Accuracy)")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f" Acceleration: CUDA GPU [Active: {gpu_name}]")
    else:
        print(" Acceleration: CPU Mode")
    print("=" * 65)

    # Load Models
    print(f"\n[1/2] Loading ThirdWheel Neural Engine ({device}) ...")
    t0 = time.perf_counter()
    try:
        model = RelateAnything.from_pretrained(args.model, device=device)
    except Exception as e:
        print(f"[ERROR] Failed to load model '{args.model}': {e}")
        print("  - Check internet connection (required for first download)")
        print("  - Verify model name at: https://huggingface.co/collections/maelic/relateanything")
        print("  - Try 'maelic/relsgg-vits16plus' (recommended default)")
        sys.exit(1)
    print(f" Neural Engine ready in {(time.perf_counter() - t0):.2f}s!")

    print(f"[2/2] Loading Instance Segmentation Detector ({device}) ...")
    try:
        from ultralytics import YOLO
        detector = YOLO("yolo11n-seg.pt")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO detector: {e}")
        print("  - Run: pip install ultralytics~=8.4")
        sys.exit(1)
    print(" Detector ready!")

    # Execute selected mode
    if args.mode == "camera":
        run_camera(args.camera, detector, model, device=device, topk=args.topk, headless=args.headless)
    elif args.mode == "video":
        run_video(args.video, detector, model, device=device, topk=args.topk, headless=args.headless)
    else:
        run_image(args.image, detector, model, device=device, topk=args.topk, headless=args.headless, output_path=args.output)


if __name__ == "__main__":
    main()
