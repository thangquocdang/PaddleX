#!/usr/bin/env python3
"""
Simple debug script for table cell detection
"""
import sys
import os

def debug_cells(image_path):
    from paddlex import create_pipeline
    import numpy as np

    print(f"🔍 Debugging: {image_path}\n")

    # Create pipeline
    custom_config = {
        "pipeline_name": "table_recognition_v2",
        "use_doc_preprocessor": False,
        "use_layout_detection": True,
        "use_ocr_model": False,  # Skip OCR for faster debug

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

    # Get internal pipeline
    p = pipeline._pipeline

    # Load and prepare image
    from PIL import Image
    import numpy as np

    img_pil = Image.open(image_path)
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')

    img_array = np.array(img_pil)
    print(f"✅ Image shape: {img_array.shape}\n")

    # Step 1: Layout Detection
    print("=" * 60)
    print("STEP 1: LAYOUT DETECTION")
    print("=" * 60)

    layout_res = list(p.layout_det_model(img_array))[0]

    if 'boxes' in layout_res:
        print(f"Detected {len(layout_res['boxes'])} regions:")
        for idx, box in enumerate(layout_res['boxes']):
            label = box.get('label', 'unknown')
            score = box.get('score', 0)
            coord = box.get('coordinate', [])
            print(f"  {idx+1}. {label}: score={score:.3f}, box={[int(c) for c in coord]}")
    else:
        print("❌ No boxes detected!")
        return

    # Process each table
    for table_idx, box_info in enumerate(layout_res['boxes']):
        if box_info['label'].lower() != 'table':
            continue

        print(f"\n{'=' * 60}")
        print(f"STEP 2: PROCESSING TABLE {table_idx + 1}")
        print(f"{'=' * 60}")

        # Crop table
        x1, y1, x2, y2 = [int(c) for c in box_info['coordinate']]
        crop_img = img_array[y1:y2, x1:x2]
        crop_h, crop_w = crop_img.shape[:2]

        print(f"Table box: [{x1}, {y1}, {x2}, {y2}]")
        print(f"Cropped size: {crop_w} x {crop_h} pixels")

        if crop_h < 10 or crop_w < 10:
            print("⚠️  WARNING: Table too small!")
            continue

        # Step 3: Cell Detection with multiple thresholds
        print(f"\n{'=' * 60}")
        print(f"STEP 3: CELL DETECTION (on cropped table)")
        print(f"{'=' * 60}")

        for threshold in [0.1, 0.2, 0.3, 0.5]:
            try:
                cell_res = list(
                    p.wired_table_cells_detection_model(crop_img, threshold=threshold)
                )[0]

                if 'boxes' in cell_res:
                    num_cells = len(cell_res['boxes'])
                    print(f"Threshold {threshold}: {num_cells} cells detected")

                    if num_cells > 0 and num_cells <= 10:
                        for i, cell in enumerate(cell_res['boxes'][:10]):
                            cx1, cy1, cx2, cy2 = [int(c) for c in cell['coordinate']]
                            cscore = cell.get('score', 0)
                            print(f"  Cell {i+1}: [{cx1:4d}, {cy1:4d}, {cx2:4d}, {cy2:4d}], score={cscore:.3f}")
                    elif num_cells > 10:
                        for i, cell in enumerate(cell_res['boxes'][:3]):
                            cx1, cy1, cx2, cy2 = [int(c) for c in cell['coordinate']]
                            cscore = cell.get('score', 0)
                            print(f"  Cell {i+1}: [{cx1:4d}, {cy1:4d}, {cx2:4d}, {cy2:4d}], score={cscore:.3f}")
                        print(f"  ... and {num_cells - 3} more cells")
                else:
                    print(f"Threshold {threshold}: No 'boxes' in result (keys: {list(cell_res.keys())})")

            except Exception as e:
                print(f"Threshold {threshold}: ERROR - {e}")

        # Step 4: Structure Recognition
        print(f"\n{'=' * 60}")
        print(f"STEP 4: STRUCTURE RECOGNITION")
        print(f"{'=' * 60}")

        try:
            struct_res = list(p.wired_table_rec_model(crop_img))[0]

            if 'html' in struct_res:
                html = struct_res['html']
                print(f"✅ HTML structure generated ({len(html)} chars)")
                print(f"HTML preview:\n{html[:300]}...")
            else:
                print(f"❌ No HTML in result (keys: {list(struct_res.keys())})")

        except Exception as e:
            print(f"❌ Structure recognition error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_cells_simple.py <image_path>")
        print("Example: python debug_cells_simple.py /content/image.jpg")
        sys.exit(1)

    debug_cells(sys.argv[1])
