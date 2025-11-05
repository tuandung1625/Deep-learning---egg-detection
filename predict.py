import sys
import cv2
import numpy as np
from matplotlib.pyplot import imread, imshow
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('./Egg_Model.keras')

# --- Get image path from command line ---
if len(sys.argv) < 2:
    print("Usage: python predict.py <image_file_name>")
    sys.exit()

img_path = sys.argv[1]

# --- Load and preprocess image ---
img = cv2.imread(img_path)
if img is None:
    print("Error: Could not read the image. Check the file path.")
    sys.exit()

img = cv2.resize(img, (256, 256))
x = np.expand_dims(img, axis=0)
x = x / 255.0

# --- Predict ---
preds = model.predict(x, verbose=0)
pred_class = np.argmax(preds, axis=1)[0]
confidence = preds[0][pred_class] * 100

class_names = ['Damaged', 'Not Damaged']
print(f"\nPredicted: {class_names[pred_class]} ({confidence:.2f}% confidence)")