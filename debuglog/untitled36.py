# -*- coding: utf-8 -*-
"""Untitled36.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1frrZSk5OwWk7QUhl3Fjt44lvGfxeV9KG
"""

import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("/content/mobilenetv2_fruit_classifier_day13(1).h5")

# Print a summary of the model architecture (optional)
model.summary()

import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from glob import glob

# === CONFIG ===
zip_path = "/content/testimgran.zip"  # Update if your zip has a different name
extract_path = "/content/test_images"
class_names = ['ripe', 'turning', 'unripe']

# === UNZIP ===
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print("✅ Zip extracted!")

# === FIND IMAGES ===
image_paths = glob(os.path.join(extract_path, "**", "*.jpg"), recursive=True) + \
              glob(os.path.join(extract_path, "**", "*.jpeg"), recursive=True) + \
              glob(os.path.join(extract_path, "**", "*.png"), recursive=True)

print(f"🔍 Found {len(image_paths)} image(s)\n")

# === PREDICT + COLLECT RESULTS ===
images = []
labels = []

for img_path in image_paths:
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction[0])
        predicted_label = class_names[predicted_index]

        images.append(img)
        labels.append(f"{os.path.basename(img_path)}\n→ {predicted_label}")

    except Exception as e:
        print(f"⚠️ Error with image {img_path}: {e}")

# === DISPLAY GRID ===
cols = 3
rows = (len(images) + cols - 1) // cols

plt.figure(figsize=(15, 5 * rows))
for i in range(len(images)):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(images[i])
    plt.title(labels[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

import unittest
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

class_names = ['ripe', 'turning', 'unripe']
model_path = '/content/mobilenetv2_fruit_classifier.h5'
test_image_path = '/content/test_images_sample/test1.jpg'  # Update path
test_batch_paths = [
    '/content/test_images_sample/test1.jpg',
    '/content/test_images_sample/test2.jpg'
]

class TestFruitModelPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load model once for all tests
        cls.model = tf.keras.models.load_model(model_path)

    def test_model_loads(self):
        """Test model loads without error and has output layer of correct shape"""
        self.assertIsNotNone(self.model, "Model failed to load")
        self.assertEqual(self.model.output_shape[-1], 3, "Output layer must have 3 classes")

    def test_single_image_prediction(self):
        """Test prediction class is valid"""
        img = image.load_img(test_image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        preds = self.model.predict(img_array)
        pred_idx = np.argmax(preds[0])

        self.assertIn(class_names[pred_idx], class_names, "Predicted class not in class_names")

    def test_batch_prediction_shape(self):
        """Test batch prediction returns correct shape"""
        batch = []
        for path in test_batch_paths:
            img = image.load_img(path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            batch.append(img_array)
        batch = np.array(batch) / 255.0

        preds = self.model.predict(batch)
        self.assertEqual(preds.shape, (len(test_batch_paths), 3), "Prediction shape mismatch")

if __name__ == '__main__':
    # Save debug output to a file
    with open("day17_debug_log.txt", "w") as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        unittest.main(testRunner=runner, exit=False)

import unittest
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# === CONFIG ===
class_names = ['ripe', 'turning', 'unripe']
model_path = '/content/mobilenetv2_fruit_classifier_day13(1).h5'  # Update if your model path differs
test_image_path = '/content/test_images/testimgran/-6014-_jpg.rf.9441dc5b7e1ea01b56b27acb47ff94bb.jpg'  # Provide a valid test image path
test_batch_paths = [
    '/content/test_images/testimgran/953e67ed2647533c24880698d0193977_-1-1-a_jpg.rf.d5cbe2d18db09557ca5f72390337e91e.jpg',
    '/content/test_images/testimgran/fresa_272_jpg.rf.7f723b03047902fae0dae48f88cd3960.jpg'
]

class TestFruitModelPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the model once before all tests"""
        cls.model = tf.keras.models.load_model(model_path)

    def test_model_loads(self):
        """Test that the model loads and has the correct output shape"""
        self.assertIsNotNone(self.model, "❌ Model failed to load")
        self.assertEqual(self.model.output_shape[-1], 3, "❌ Output layer must have 3 classes")

    def test_single_image_prediction(self):
        """Test model predicts a class within class_names for a single image"""
        img = image.load_img(test_image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        preds = self.model.predict(img_array)
        pred_idx = np.argmax(preds[0])

        self.assertIn(class_names[pred_idx], class_names, "❌ Predicted class not in class_names")

    def test_batch_prediction_shape(self):
        """Test model returns correct prediction shape for batch input"""
        batch = []
        for path in test_batch_paths:
            img = image.load_img(path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            batch.append(img_array)
        batch = np.array(batch) / 255.0

        preds = self.model.predict(batch)
        self.assertEqual(preds.shape, (len(test_batch_paths), 3), "❌ Prediction shape mismatch")

if __name__ == '__main__':
    # Save output to a debug log file
    with open("day17_debug_log.txt", "w") as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        unittest.main(argv=['first-arg-is-ignored'], testRunner=runner, exit=False)