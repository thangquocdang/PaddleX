#!/usr/bin/env python3
"""
Debug script to check table detection and cell detection workflow
"""
import numpy as np
from paddlex import create_pipeline
from PIL import Image, ImageDraw, ImageFont
import os

# Import cv2 from paddlex's dependencies
try:
    import cv2
except ImportError:
    from paddlex.utils import lazy_imports
    cv2 = lazy_imports.LazyImport("cv2")

def debug_table_cell_detection(image_path, output_dir="./debug_output"):
    """
    Debug table and cell detection step by step
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot load image: {image_path}")
        return

    print(f"✅ Image loaded: {img.shape}")

    # Create minimal pipeline config
    custom_config = {
        "pipeline_name": "table_recognition_v2",
        "use_doc_preprocessor": False,
        "use_layout_detection": True,
        "use_ocr_model": False,  # Tắt OCR để debug nhanh hơn

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

    print("\n🔧 Creating pipeline...")
    pipeline = create_pipeline(config=custom_config, device="cpu")

    # Step 1: Layout Detection
    print("\n📍 Step 1: Layout Detection...")
    layout_det_res = list(pipeline._pipeline.layout_det_model(img))[0]
    print(f"   Detected {len(layout_det_res['boxes'])} regions")

    # Visualize layout detection
    vis_img = img.copy()
    for idx, box_info in enumerate(layout_det_res['boxes']):
        if box_info['label'].lower() == 'table':
            x1, y1, x2, y2 = [int(c) for c in box_info['coordinate']]
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(vis_img, f"Table {idx+1}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"   - Table {idx+1}: [{x1}, {y1}, {x2}, {y2}], score={box_info['score']:.3f}")

    cv2.imwrite(f"{output_dir}/01_layout_detection.jpg", vis_img)
    print(f"   ✅ Saved: {output_dir}/01_layout_detection.jpg")

    # Step 2: Process each table
    for table_idx, box_info in enumerate(layout_det_res['boxes']):
        if box_info['label'].lower() != 'table':
            continue

        print(f"\n📊 Processing Table {table_idx+1}...")

        # Crop table
        x1, y1, x2, y2 = [int(c) for c in box_info['coordinate']]
        crop_img = img[y1:y2, x1:x2]
        crop_h, crop_w = crop_img.shape[:2]

        print(f"   Cropped table size: {crop_w}x{crop_h}")
        cv2.imwrite(f"{output_dir}/02_table_{table_idx+1}_crop.jpg", crop_img)

        # Step 3: Cell Detection on CROPPED table
        print(f"   🔍 Running cell detection...")

        # Test with different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.5]

        for threshold in thresholds:
            try:
                cell_pred = list(
                    pipeline._pipeline.wired_table_cells_detection_model(
                        crop_img,
                        threshold=threshold
                    )
                )[0]

                # Extract cell boxes
                if 'boxes' in cell_pred:
                    num_cells = len(cell_pred['boxes'])
                    print(f"   - Threshold {threshold}: {num_cells} cells detected")

                    if num_cells > 0:
                        # Visualize cells
                        cell_vis = crop_img.copy()
                        for cell_box in cell_pred['boxes']:
                            cx1, cy1, cx2, cy2 = [int(c) for c in cell_box['coordinate']]
                            cv2.rectangle(cell_vis, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)

                        cv2.imwrite(
                            f"{output_dir}/03_table_{table_idx+1}_cells_th{threshold}.jpg",
                            cell_vis
                        )
                        print(f"     ✅ Saved: {output_dir}/03_table_{table_idx+1}_cells_th{threshold}.jpg")

                        # Print first few cell coordinates
                        for i, cell_box in enumerate(cell_pred['boxes'][:5]):
                            cx1, cy1, cx2, cy2 = [int(c) for c in cell_box['coordinate']]
                            score = cell_box.get('score', 0)
                            print(f"       Cell {i+1}: [{cx1}, {cy1}, {cx2}, {cy2}], score={score:.3f}")

                        if num_cells > 5:
                            print(f"       ... and {num_cells-5} more cells")
                else:
                    print(f"   - Threshold {threshold}: No 'boxes' key in result")
                    print(f"     Result keys: {cell_pred.keys()}")

            except Exception as e:
                print(f"   ❌ Error at threshold {threshold}: {e}")

        # Step 4: Structure Recognition
        print(f"   📝 Running structure recognition...")
        try:
            structure_pred = list(
                pipeline._pipeline.wired_table_rec_model(crop_img)
            )[0]

            if 'html' in structure_pred:
                html_content = structure_pred['html']
                html_file = f"{output_dir}/04_table_{table_idx+1}_structure.html"
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(f"<html><body>{html_content}</body></html>")
                print(f"   ✅ Saved: {html_file}")
                print(f"   HTML preview: {html_content[:200]}...")
            else:
                print(f"   No HTML in structure result")
                print(f"   Result keys: {structure_pred.keys()}")

        except Exception as e:
            print(f"   ❌ Structure recognition error: {e}")

    print(f"\n✅ Debug complete! Check {output_dir}/ for outputs")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python debug_table_cells.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    debug_table_cell_detection(image_path)
