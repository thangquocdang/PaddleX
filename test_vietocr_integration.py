#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for VietOCR ONNX integration with PaddleX Table Recognition Pipeline

Usage:
    python test_vietocr_integration.py --image table.jpg --vietocr_dir ./weight
"""

import argparse
import os
import sys

import cv2
import numpy as np


def test_vietocr_standalone():
    """Test VietOCR ONNX model standalone"""
    print("=" * 80)
    print("TEST 1: VietOCR ONNX Standalone")
    print("=" * 80)

    from paddlex.inference.models.vietocr_onnx_rec import VietOCRONNXRecognizer

    # Initialize model
    model = VietOCRONNXRecognizer(
        model_dir=args.vietocr_dir, device="cpu", batch_size=8, max_seq_length=128
    )

    # Load test image
    img = cv2.imread(args.image)
    if img is None:
        print(f"❌ Cannot load image: {args.image}")
        return False

    print(f"✅ Loaded image: {img.shape}")

    # Test with single image
    results = list(model([img]))
    print(f"\n📝 Recognition Result:")
    print(f"   Text: {results[0]['rec_text']}")
    print(f"   Score: {results[0]['rec_score']:.2f}")

    return True


def test_vietocr_in_ocr_pipeline():
    """Test VietOCR in OCR pipeline"""
    print("\n" + "=" * 80)
    print("TEST 2: VietOCR in OCR Pipeline")
    print("=" * 80)

    from paddlex import create_pipeline
    from paddlex.inference.models.vietocr_onnx_rec import VietOCRONNXRecognizer

    # Create OCR pipeline
    print("Creating OCR pipeline...")
    pipeline = create_pipeline(pipeline="general_ocr")

    # Replace text recognition model with VietOCR
    print(f"Replacing text rec model with VietOCR from {args.vietocr_dir}")
    vietocr_model = VietOCRONNXRecognizer(
        model_dir=args.vietocr_dir, device="cpu", batch_size=8, max_seq_length=128
    )
    pipeline.text_rec_model = vietocr_model

    # Run OCR
    print(f"Running OCR on {args.image}...")
    output = pipeline.predict(args.image)

    # Print results
    for i, res in enumerate(output):
        print(f"\n📄 Page {i + 1}:")
        print(f"   Detected {len(res['rec_texts'])} text regions")

        for j, text in enumerate(res["rec_texts"][:5]):  # Show first 5
            score = res["rec_scores"][j]
            print(f"   [{j+1}] {text} (score: {score:.2f})")

        if len(res["rec_texts"]) > 5:
            print(f"   ... and {len(res['rec_texts']) - 5} more")

    return True


def test_vietocr_in_table_pipeline():
    """Test VietOCR in Table Recognition Pipeline"""
    print("\n" + "=" * 80)
    print("TEST 3: VietOCR in Table Recognition V2 Pipeline")
    print("=" * 80)

    from paddlex import create_pipeline
    from paddlex.inference.models.vietocr_onnx_rec import VietOCRONNXRecognizer

    # Create table recognition pipeline
    print("Creating table recognition v2 pipeline...")
    pipeline = create_pipeline(pipeline="table_recognition_v2")

    # Replace text recognition model with VietOCR
    print(f"Replacing text rec model with VietOCR from {args.vietocr_dir}")
    vietocr_model = VietOCRONNXRecognizer(
        model_dir=args.vietocr_dir, device="cpu", batch_size=8, max_seq_length=128
    )

    # Inject VietOCR into OCR pipeline
    pipeline.general_ocr_pipeline.text_rec_model = vietocr_model

    # Run table recognition
    print(f"Running table recognition on {args.image}...")
    output = pipeline.predict(
        args.image,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    # Print results
    for i, res in enumerate(output):
        print(f"\n📊 Page {i + 1}:")
        print(f"   Detected {len(res['table_res_list'])} tables")

        for j, table in enumerate(res["table_res_list"]):
            print(f"\n   Table {j+1}:")
            print(f"      Cells: {len(table['cell_box_list'])}")
            print(f"      HTML preview:")
            html = table["pred_html"]
            # Print first 200 chars of HTML
            print(f"      {html[:200]}...")

        # Save results
        output_dir = args.output or "./output_vietocr"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n💾 Saving results to {output_dir}...")
        res.save_to_xlsx(output_dir)
        res.save_to_html(output_dir)
        print(f"✅ Saved XLSX and HTML files")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test VietOCR ONNX integration with PaddleX"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to test image (table or text)"
    )
    parser.add_argument(
        "--vietocr_dir",
        type=str,
        default="./weight",
        help="Directory containing VietOCR ONNX models (cnn.onnx, encoder.onnx, decoder.onnx)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output_vietocr",
        help="Output directory for results",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["standalone", "ocr", "table", "all"],
        default="all",
        help="Which test to run",
    )

    args = parser.parse_args()

    # Verify VietOCR models exist
    required_files = ["cnn.onnx", "encoder.onnx", "decoder.onnx"]
    for filename in required_files:
        filepath = os.path.join(args.vietocr_dir, filename)
        if not os.path.exists(filepath):
            print(f"❌ Error: {filepath} not found!")
            print(f"   Please ensure VietOCR ONNX models are in {args.vietocr_dir}")
            sys.exit(1)

    print(f"✅ Found all required VietOCR ONNX models in {args.vietocr_dir}")

    # Run tests
    success = True

    if args.test in ["standalone", "all"]:
        success = test_vietocr_standalone() and success

    if args.test in ["ocr", "all"]:
        success = test_vietocr_in_ocr_pipeline() and success

    if args.test in ["table", "all"]:
        success = test_vietocr_in_table_pipeline() and success

    print("\n" + "=" * 80)
    if success:
        print("✅ All tests completed successfully!")
    else:
        print("❌ Some tests failed. Check errors above.")
    print("=" * 80)
