"""
Copy paste code này vào Colab notebook để debug cell detection
"""

# ========================================
# DEBUG CELL DETECTION
# ========================================

from paddlex import create_pipeline
import numpy as np
from PIL import Image

image_path = "/content/ban-scan-la-gi-scan-nhu-the-nao-e1501146396102.jpg"

print(f"🔍 Debugging: {image_path}\n")

# Create minimal pipeline (no OCR to speed up)
custom_config = {
    "pipeline_name": "table_recognition_v2",
    "use_doc_preprocessor": False,
    "use_layout_detection": True,
    "use_ocr_model": False,  # Skip OCR for debugging

    "SubModules": {
        "LayoutDetection": {
            "module_name": "layout_detection",
            "model_name": "PicoDet_layout_1x_table",
            "model_dir": None
        },
        "WiredTableStructureRecognition": {
            "module_name": "table_structure_recognition",
            "model_name": "SLANeXt_wired",
            "model_dir": None
        },
        "WiredTableCellsDetection": {
            "module_name": "table_cells_detection",
            "model_name": "RT-DETR-L_wired_table_cell_det",
            "model_dir": None
        },
    },
}

print("Creating pipeline...")
pipeline = create_pipeline(config=custom_config, device="cpu")
p = pipeline._pipeline  # Get internal pipeline

# Load image
img_pil = Image.open(image_path)
if img_pil.mode != 'RGB':
    img_pil = img_pil.convert('RGB')
img_array = np.array(img_pil)

print(f"✅ Image shape: {img_array.shape}\n")

# ========================================
# STEP 1: LAYOUT DETECTION
# ========================================
print("=" * 60)
print("STEP 1: LAYOUT DETECTION (full image)")
print("=" * 60)

layout_res = list(p.layout_det_model(img_array))[0]

if 'boxes' in layout_res and len(layout_res['boxes']) > 0:
    print(f"✅ Detected {len(layout_res['boxes'])} regions:\n")
    for idx, box in enumerate(layout_res['boxes']):
        label = box.get('label', 'unknown')
        score = box.get('score', 0)
        coord = box.get('coordinate', [])
        x1, y1, x2, y2 = [int(c) for c in coord]
        print(f"  Region {idx+1}: {label}")
        print(f"    - Box: [{x1}, {y1}, {x2}, {y2}]")
        print(f"    - Score: {score:.3f}")
        print(f"    - Size: {x2-x1} x {y2-y1} pixels\n")
else:
    print("❌ No regions detected!")

# ========================================
# STEP 2: PROCESS EACH TABLE
# ========================================
for table_idx, box_info in enumerate(layout_res['boxes']):
    if box_info['label'].lower() != 'table':
        print(f"⏭️  Skipping region {table_idx+1} (not a table)\n")
        continue

    print("=" * 60)
    print(f"STEP 2: PROCESSING TABLE {table_idx + 1}")
    print("=" * 60)

    # Crop table from original image
    x1, y1, x2, y2 = [int(c) for c in box_info['coordinate']]
    crop_img = img_array[y1:y2, x1:x2]
    crop_h, crop_w, _ = crop_img.shape

    print(f"📍 Table bounding box: [{x1}, {y1}, {x2}, {y2}]")
    print(f"📏 Cropped table size: {crop_w} x {crop_h} pixels")

    if crop_h < 10 or crop_w < 10:
        print("⚠️  WARNING: Table crop too small! Skipping...\n")
        continue

    # Optional: Save cropped table for manual inspection
    crop_pil = Image.fromarray(crop_img)
    crop_pil.save(f"/content/debug_table_{table_idx+1}_crop.jpg")
    print(f"💾 Saved cropped table: /content/debug_table_{table_idx+1}_crop.jpg\n")

    # ========================================
    # STEP 3: CELL DETECTION
    # ========================================
    print("=" * 60)
    print(f"STEP 3: CELL DETECTION (on cropped table)")
    print("=" * 60)
    print("Testing different confidence thresholds...\n")

    for threshold in [0.05, 0.1, 0.2, 0.3, 0.5]:
        print(f"📊 Threshold = {threshold}")
        try:
            # Run cell detection on CROPPED table image
            cell_res = list(
                p.wired_table_cells_detection_model(crop_img, threshold=threshold)
            )[0]

            if 'boxes' in cell_res:
                num_cells = len(cell_res['boxes'])

                if num_cells == 0:
                    print(f"   ❌ No cells detected (threshold too high?)\n")
                else:
                    print(f"   ✅ {num_cells} cells detected")

                    # Show first 5 cells
                    for i, cell in enumerate(cell_res['boxes'][:5]):
                        cx1, cy1, cx2, cy2 = [int(c) for c in cell['coordinate']]
                        cscore = cell.get('score', 0)
                        cw, ch = cx2 - cx1, cy2 - cy1
                        print(f"      Cell {i+1}: [{cx1:4d}, {cy1:4d}, {cx2:4d}, {cy2:4d}]")
                        print(f"              size={cw}x{ch}, score={cscore:.3f}")

                    if num_cells > 5:
                        print(f"      ... and {num_cells - 5} more cells")
                    print()

            else:
                print(f"   ⚠️  No 'boxes' key in result")
                print(f"      Available keys: {list(cell_res.keys())}\n")

        except Exception as e:
            print(f"   ❌ Error: {e}\n")

    # ========================================
    # STEP 4: STRUCTURE RECOGNITION
    # ========================================
    print("=" * 60)
    print(f"STEP 4: STRUCTURE RECOGNITION")
    print("=" * 60)

    try:
        struct_res = list(p.wired_table_rec_model(crop_img))[0]

        if 'html' in struct_res:
            html = struct_res['html']
            print(f"✅ HTML structure generated ({len(html)} characters)")
            print(f"\nHTML preview:")
            print(html[:400])
            print("...\n")

            # Save HTML
            with open(f"/content/debug_table_{table_idx+1}_structure.html", 'w') as f:
                f.write(f"<html><body>{html}</body></html>")
            print(f"💾 Saved: /content/debug_table_{table_idx+1}_structure.html\n")

        else:
            print(f"❌ No 'html' key in result")
            print(f"Available keys: {list(struct_res.keys())}\n")

    except Exception as e:
        print(f"❌ Structure recognition error: {e}\n")

print("=" * 60)
print("✅ DEBUG COMPLETE!")
print("=" * 60)
print("\nSummary:")
print("- Check /content/debug_table_*_crop.jpg to see cropped tables")
print("- Check /content/debug_table_*_structure.html to see table structure")
print("\nIf cells are detected at low thresholds but not at 0.3:")
print("  → Model is working, but confidence threshold is too high")
print("  → Lower the threshold in pipeline predict() call")
print("\nIf NO cells detected at any threshold:")
print("  → Table image quality issue or model mismatch")
print("  → Check cropped table images manually")
