#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization script for debugging table recognition pipeline

Shows step-by-step visualizations:
1. Layout Detection (table regions)
2. Cell Detection (cell boxes)
3. Text Recognition (OCR results in cells)
4. Final table structure
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def visualize_layout_detection(image, layout_boxes, save_path=None):
    """Visualize layout detection results (table regions)"""
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Convert BGR to RGB for matplotlib
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    ax.imshow(image_rgb)
    ax.set_title("Layout Detection - Table Regions", fontsize=16, fontweight='bold')

    # Draw table regions
    for i, box in enumerate(layout_boxes):
        if box['label'].lower() == 'table':
            coord = box['coordinate']
            x1, y1, x2, y2 = coord

            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)

            # Add label
            ax.text(
                x1, y1-10, f"Table {i+1}",
                color='red', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved layout visualization: {save_path}")

    return fig


def visualize_cell_detection(image, cell_boxes, save_path=None):
    """Visualize cell detection results"""
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Convert BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    ax.imshow(image_rgb)
    ax.set_title(f"Cell Detection - {len(cell_boxes)} Cells", fontsize=16, fontweight='bold')

    # Draw cell boxes
    for i, box in enumerate(cell_boxes):
        x1, y1, x2, y2 = box

        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=1, edgecolor='blue', facecolor='none', alpha=0.7
        )
        ax.add_patch(rect)

    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved cell visualization: {save_path}")

    return fig


def visualize_text_recognition(image, cell_boxes, ocr_results, save_path=None):
    """Visualize text recognition results"""
    # Use PIL for better text rendering
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        pil_img = Image.fromarray(image_rgb)
    else:
        pil_img = image

    draw = ImageDraw.Draw(pil_img)

    # Try to use a font that supports Vietnamese
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()

    # Draw cells and text
    for i, box in enumerate(cell_boxes):
        x1, y1, x2, y2 = [int(p) for p in box]

        # Draw cell box
        draw.rectangle([x1, y1, x2, y2], outline='blue', width=1)

        # Draw text if available
        if i < len(ocr_results) and ocr_results[i]:
            text = ocr_results[i]
            # Draw text background
            text_bbox = draw.textbbox((x1+2, y1+2), text, font=font)
            draw.rectangle(text_bbox, fill='white', outline='black')
            # Draw text
            draw.text((x1+2, y1+2), text, fill='red', font=font)

    # Convert back to matplotlib
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(pil_img)
    ax.set_title(f"Text Recognition - {len(ocr_results)} Texts", fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved text visualization: {save_path}")

    return fig


def debug_pipeline(result, output_dir="./debug_viz"):
    """
    Debug visualization for table recognition pipeline

    Args:
        result: TableRecognitionResult from pipeline.predict()
        output_dir: Output directory for visualization images
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get base image
    base_image = result["doc_preprocessor_res"]["output_img"]

    print("\n" + "="*80)
    print("🔍 TABLE RECOGNITION DEBUG VISUALIZATION")
    print("="*80)

    # 1. Layout Detection
    if result["layout_det_res"] and len(result["layout_det_res"]) > 0:
        print("\n📊 Step 1: Layout Detection")
        layout_boxes = result["layout_det_res"]["boxes"]
        print(f"   Detected {len([b for b in layout_boxes if b['label'].lower() == 'table'])} table(s)")

        visualize_layout_detection(
            base_image,
            layout_boxes,
            save_path=os.path.join(output_dir, "01_layout_detection.png")
        )

    # 2. For each table
    for table_idx, table_res in enumerate(result["table_res_list"]):
        print(f"\n📋 Table {table_idx + 1}:")

        # Cell detection
        cell_boxes = table_res["cell_box_list"]
        print(f"   ✓ Detected {len(cell_boxes)} cells")

        visualize_cell_detection(
            base_image,
            cell_boxes,
            save_path=os.path.join(output_dir, f"02_table{table_idx+1}_cells.png")
        )

        # Text recognition
        if "table_ocr_pred" in table_res and table_res["table_ocr_pred"]:
            ocr_pred = table_res["table_ocr_pred"]
            if "rec_texts" in ocr_pred:
                texts = ocr_pred["rec_texts"]
                print(f"   ✓ Recognized {len(texts)} texts")
                print(f"   ✓ Sample texts: {texts[:3]}")

                visualize_text_recognition(
                    base_image,
                    cell_boxes,
                    texts,
                    save_path=os.path.join(output_dir, f"03_table{table_idx+1}_text.png")
                )

        # HTML structure
        if "pred_html" in table_res:
            html = table_res["pred_html"]
            print(f"   ✓ HTML structure ({len(html)} chars)")
            # Save HTML
            html_path = os.path.join(output_dir, f"04_table{table_idx+1}_structure.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"   ✓ Saved HTML: {html_path}")

    print("\n" + "="*80)
    print(f"✅ All visualizations saved to: {output_dir}")
    print("="*80)

    # Show summary
    print("\n📁 Generated files:")
    for filename in sorted(os.listdir(output_dir)):
        filepath = os.path.join(output_dir, filename)
        filesize = os.path.getsize(filepath) / 1024  # KB
        print(f"   - {filename} ({filesize:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug visualization for table recognition")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--vietocr_dir", type=str, default="./weight", help="VietOCR models directory")
    parser.add_argument("--output", type=str, default="./debug_viz", help="Output directory")

    args = parser.parse_args()

    # Run pipeline
    print("🚀 Running table recognition pipeline...")

    from paddlex import create_pipeline
    from paddlex.inference.models.vietocr_onnx_rec import VietOCRONNXRecognizer

    pipeline = create_pipeline(pipeline="table_recognition_v2")

    # Use VietOCR
    vietocr = VietOCRONNXRecognizer(model_dir=args.vietocr_dir, device="cpu", batch_size=8)
    pipeline.general_ocr_pipeline.text_rec_model = vietocr

    output = pipeline.predict(args.image)

    # Debug visualization
    for res in output:
        debug_pipeline(res, output_dir=args.output)

    print("\n🎉 Done! Open the images to see visualizations.")
