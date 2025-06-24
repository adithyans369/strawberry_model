
# 🍓 Fruit Classification using MobileNetV2

This project uses **MobileNetV2**, a lightweight convolutional neural network, to classify images of fruits into different categories such as ripe, unripe, and turning.

## 🧠 Overview

The notebook demonstrates how to:
- Load and preprocess a custom dataset
- Use `ImageDataGenerator` to manage training/validation
- Fine-tune MobileNetV2 on the dataset
- Evaluate the trained model
- Predict on new images

## 📂 Dataset

The dataset is a ZIP file named `new_dataset.zip` and is structured for use with `ImageDataGenerator`:
```
new_dataset/
├── train/
│   ├── ripe/
│   ├── unripe/
│   └── turning/
├── val/
│   └── (same structure as train)
```

Make sure the dataset is unzipped in your working directory before running the notebook.

## 🏗️ Model Architecture

- **Base Model:** MobileNetV2 (pretrained on ImageNet)
- **Top Layers:** Custom dense layers added for classification
- **Activation:** Softmax for multi-class output

```python
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
```

The base model is followed by global average pooling and a dense layer.

## ⚙️ Training

- Optimizer: Adam (`learning_rate=1e-5`)
- Loss: Categorical crossentropy
- Metrics: Accuracy
- Early stopping and model checkpointing may be included for better performance.

## 📊 Evaluation

The model is evaluated using:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- Visual inspection of predictions

## 🔍 Inference

To test new images:
```python
img = image.load_img('test_image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
predictions = model.predict(img_array)
```

## 💾 Saving the Model

The trained model is saved as:
```python
model.save('/content/mobilenetv2_fruit_classifier.h5')
```

## 📦 Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pillow

Install dependencies using:
```bash
pip install tensorflow matplotlib pillow
```

---

## ✍️ Author

Adithyan 
Anjana
Gouri

---
