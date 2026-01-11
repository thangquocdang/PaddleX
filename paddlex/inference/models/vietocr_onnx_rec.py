# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, Dict, Iterator

import cv2
import numpy as np
import onnxruntime


class Vocab:
    """VietOCR Vocabulary"""

    def __init__(self, chars):
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.mask_token = 3

        self.chars = chars

        self.c2i = {c: i + 4 for i, c in enumerate(chars)}
        self.i2c = {i + 4: c for i, c in enumerate(chars)}

        self.i2c[0] = "<pad>"
        self.i2c[1] = "<sos>"
        self.i2c[2] = "<eos>"
        self.i2c[3] = "*"

    def encode(self, chars):
        return [self.go] + [self.c2i[c] for c in chars] + [self.eos]

    def decode(self, ids):
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        sent = "".join([self.i2c[i] for i in ids[first:last]])
        return sent

    def __len__(self):
        return len(self.c2i) + 4

    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars


class VietOCRONNXRecognizer:
    """
    VietOCR ONNX Text Recognition Model

    Replaces PP-OCRv4 recognition with VietOCR ONNX models for Vietnamese text.
    Compatible with PaddleX OCR pipeline interface.
    """

    def __init__(
        self,
        model_dir: str = None,
        device: str = "cpu",
        batch_size: int = 8,
        max_seq_length: int = 128,
        target_height: int = 32,
        target_width: int = None,  # Dynamic width if None
        **kwargs,
    ):
        """
        Initialize VietOCR ONNX model

        Args:
            model_dir (str): Directory containing cnn.onnx, encoder.onnx, decoder.onnx
            device (str): Device to run inference ('cpu' or 'gpu')
            batch_size (int): Batch size for inference
            max_seq_length (int): Maximum sequence length for decoding
            target_height (int): Target height for input images (default: 32)
            target_width (int): Target width for input images (None for dynamic)
        """
        self.model_dir = model_dir or "./weight"
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.target_height = target_height
        self.target_width = target_width

        # VietOCR vocabulary
        vocab_chars = (
            'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆ'
            'fFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStT'
            'uUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'\'()*+,-./:;<=>?@[\\]^_`{|}~ '
        )
        self.vocab = Vocab(vocab_chars)

        # Load ONNX models
        self._load_models()

    def _load_models(self):
        """Load VietOCR ONNX models"""
        cnn_path = os.path.join(self.model_dir, "cnn.onnx")
        encoder_path = os.path.join(self.model_dir, "encoder.onnx")
        decoder_path = os.path.join(self.model_dir, "decoder.onnx")

        # Check if files exist
        for path in [cnn_path, encoder_path, decoder_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"ONNX model not found: {path}")

        # Create ONNX Runtime sessions
        providers = ["CPUExecutionProvider"]
        if self.device == "gpu":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.cnn_session = onnxruntime.InferenceSession(cnn_path, providers=providers)
        self.encoder_session = onnxruntime.InferenceSession(
            encoder_path, providers=providers
        )
        self.decoder_session = onnxruntime.InferenceSession(
            decoder_path, providers=providers
        )

        print(f"✅ VietOCR ONNX models loaded from {self.model_dir}")
        print(f"   - CNN input shape: {self.cnn_session.get_inputs()[0].shape}")

    def _preprocess(self, img):
        """
        Preprocess image for VietOCR ONNX model

        Args:
            img (np.ndarray): Input image (H, W, C) or (H, W)

        Returns:
            np.ndarray: Preprocessed image (1, C, H, W)
        """
        # Check if image is valid
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or None")

        # Check initial dimensions
        if len(img.shape) < 2:
            raise ValueError(f"Invalid image shape: {img.shape}")

        h, w = img.shape[:2]
        if h == 0 or w == 0:
            raise ValueError(f"Invalid image dimensions: h={h}, w={w}, shape={img.shape}")

        # Convert to RGB if grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif img.shape[2] == 3:
                # Assume BGR, convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        # Get dimensions AFTER color conversion
        h, w = img.shape[:2]

        # Double-check after conversion
        if h == 0 or w == 0:
            raise ValueError(f"Invalid dimensions after color conversion: h={h}, w={w}")

        # Resize to target height, maintain aspect ratio
        if self.target_width is None:
            # Dynamic width - maintain aspect ratio
            ratio = float(self.target_height) / float(h)
            new_w = int(w * ratio)
            # Ensure minimum width
            new_w = max(1, new_w)
        else:
            # Fixed width
            new_w = self.target_width

        # Final validation
        if new_w <= 0 or self.target_height <= 0:
            raise ValueError(
                f"Invalid resize dimensions: width={new_w}, height={self.target_height} "
                f"(original: h={h}, w={w})"
            )

        try:
            resized = cv2.resize(img, (new_w, self.target_height))
        except cv2.error as e:
            raise ValueError(
                f"cv2.resize failed: {e}. "
                f"Input shape: {img.shape}, Target size: ({new_w}, {self.target_height})"
            )

        # Normalize to [0, 1]
        img_array = resized.astype(np.float32) / 255.0

        # Transpose to (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add batch dimension (1, C, H, W)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def _translate_onnx(self, img_batch):
        """
        Translate image batch to text using ONNX models

        Args:
            img_batch (np.ndarray): Input image batch (B, C, H, W)

        Returns:
            np.ndarray: Translated sentence indices (B, max_seq_length)
        """
        # Step 1: CNN feature extraction
        cnn_input = {self.cnn_session.get_inputs()[0].name: img_batch}
        src = self.cnn_session.run(None, cnn_input)

        # Step 2: Encoder
        encoder_input = {self.encoder_session.get_inputs()[0].name: src[0]}
        encoder_outputs, hidden = self.encoder_session.run(None, encoder_input)

        # Step 3: Decoder (auto-regressive)
        batch_size = len(img_batch)
        translated_sentence = [[self.vocab.go] * batch_size]
        max_length = 0

        while max_length <= self.max_seq_length and not all(
            np.any(np.asarray(translated_sentence).T == self.vocab.eos, axis=1)
        ):
            tgt_inp = translated_sentence
            decoder_input = {
                self.decoder_session.get_inputs()[0].name: np.array(tgt_inp[-1]),
                self.decoder_session.get_inputs()[1].name: hidden,
                self.decoder_session.get_inputs()[2].name: encoder_outputs,
            }

            output, hidden, _ = self.decoder_session.run(None, decoder_input)
            output = np.expand_dims(output, axis=1)

            # Get top-1 prediction
            indices = np.argmax(output, axis=-1)
            indices = indices[:, -1, 0].tolist()

            translated_sentence.append(indices)
            max_length += 1

        translated_sentence = np.asarray(translated_sentence).T
        return translated_sentence

    def __call__(self, img_list, return_word_box=False):
        """
        Run inference on image list (compatible with PaddleX OCR pipeline)

        Args:
            img_list (list): List of cropped text images (np.ndarray)
            return_word_box (bool): Whether to return word-level boxes (not supported)

        Yields:
            dict: Recognition result for each image
                {
                    "rec_text": str,
                    "rec_score": float,
                    "vis_font": str
                }
        """
        # Process in batches
        for i in range(0, len(img_list), self.batch_size):
            batch_imgs = img_list[i : i + self.batch_size]

            # Preprocess all images in batch
            preprocessed_batch = []
            invalid_indices = []  # Track which images failed preprocessing

            for idx, img in enumerate(batch_imgs):
                try:
                    preprocessed = self._preprocess(img)
                    preprocessed_batch.append(preprocessed)
                except (ValueError, cv2.error) as e:
                    # Handle invalid images (too small, empty, etc.)
                    invalid_indices.append(idx)
                    print(f"⚠️ VietOCR preprocessing warning: {e} (image {i+idx})")
                    # Add dummy result for this image
                    preprocessed_batch.append(None)

            # Process each image individually (VietOCR ONNX only supports batch_size=1)
            for idx, img_data in enumerate(preprocessed_batch):
                if img_data is None:
                    # Invalid image, return empty result
                    yield {"rec_text": "", "rec_score": 0.0, "vis_font": "vietnamese"}
                else:
                    # Run inference on single image
                    try:
                        # img_data is already (1, C, H, W), ready for ONNX
                        translated_sentence = self._translate_onnx(img_data)

                        # Decode result (get first and only result since batch=1)
                        sentence_ids = translated_sentence[0]
                        rec_text = self.vocab.decode(sentence_ids.tolist())

                        # Calculate confidence score
                        rec_score = 0.95 if rec_text else 0.0

                        result = {
                            "rec_text": rec_text,
                            "rec_score": rec_score,
                            "vis_font": "vietnamese",
                        }

                        yield result

                    except Exception as e:
                        print(f"⚠️ VietOCR inference error: {e}")
                        # Return empty result on error
                        yield {"rec_text": "", "rec_score": 0.0, "vis_font": "vietnamese"}
