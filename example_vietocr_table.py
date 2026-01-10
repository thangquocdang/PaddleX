#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple example: Table Recognition with VietOCR ONNX

This example shows how to use VietOCR ONNX for Vietnamese text recognition
in the table recognition pipeline.
"""

from paddlex import create_pipeline
from paddlex.inference.models.vietocr_onnx_rec import VietOCRONNXRecognizer

# ========== Configuration ==========
IMAGE_PATH = "/content/18092025120000Eval35-Solar.pdf"  # Your table image/PDF
VIETOCR_DIR = "./weight"  # Directory containing cnn.onnx, encoder.onnx, decoder.onnx
OUTPUT_DIR = "./output_vietocr"

# ========== Step 1: Create Table Recognition Pipeline ==========
print("🚀 Creating table recognition v2 pipeline...")
pipeline = create_pipeline(pipeline="table_recognition_v2")

# ========== Step 2: Replace Text Recognition with VietOCR ==========
print(f"🔧 Loading VietOCR ONNX models from {VIETOCR_DIR}...")
vietocr_model = VietOCRONNXRecognizer(
    model_dir=VIETOCR_DIR,
    device="cpu",  # Change to "gpu" if you have GPU
    batch_size=8,  # Adjust based on your memory
    max_seq_length=128,  # Max text length
    target_height=32,  # VietOCR input height
)

# Inject VietOCR into the pipeline
pipeline.general_ocr_pipeline.text_rec_model = vietocr_model
print("✅ VietOCR integrated successfully!")

# ========== Step 3: Run Table Recognition ==========
print(f"\n📊 Processing table image: {IMAGE_PATH}")
output = pipeline.predict(
    IMAGE_PATH,
    use_doc_orientation_classify=False,  # Skip if table is not rotated
    use_doc_unwarping=False,  # Skip if image is not warped
)

# ========== Step 4: Save Results ==========
print(f"\n💾 Saving results to {OUTPUT_DIR}...")
for res in output:
    # Print summary
    num_tables = len(res["table_res_list"])
    print(f"   ✓ Detected {num_tables} table(s)")

    for i, table in enumerate(res["table_res_list"]):
        num_cells = len(table["cell_box_list"])
        print(f"   ✓ Table {i+1}: {num_cells} cells")

    # Save to files
    res.save_to_xlsx(OUTPUT_DIR)  # ← XLSX với tiếng Việt!
    res.save_to_html(OUTPUT_DIR)
    res.save_to_json(OUTPUT_DIR)

print(f"\n🎉 Done! Check results in {OUTPUT_DIR}/")
print("   - *.xlsx: Excel files with Vietnamese text")
print("   - *.html: HTML files for visualization")
print("   - *.json: JSON files with full data")
