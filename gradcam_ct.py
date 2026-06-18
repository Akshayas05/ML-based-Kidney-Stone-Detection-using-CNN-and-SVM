
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = 224
MODEL_PATH = "ct_vgg16_stone_model.h5"
LAST_CONV_LAYER = "block5_conv3"   # VGG16 last conv layer
TEST_IMAGE_PATH = r"D:\Main project\CT_Dataset\val\stone\Stone- (192).jpg"
# ^ change to any image you want to test

# ----------------------------
# LOAD MODEL
# ----------------------------
model = load_model(MODEL_PATH)

# ----------------------------
# IMAGE LOADER
# ----------------------------
def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img / 255.0
    return img, np.expand_dims(img_norm, axis=0)

# ----------------------------
# GRAD-CAM FUNCTION
# ----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    return heatmap

# ----------------------------
# RUN GRAD-CAM
# ----------------------------
orig_img, img_array = load_image(TEST_IMAGE_PATH)

prediction = model.predict(img_array)[0][0]
label = "Stone Detected" if prediction > 0.5 else "Normal"

heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)

# Resize heatmap to image size
heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
heatmap = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Overlay heatmap on image
superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)

# ----------------------------
# DISPLAY RESULT
# ----------------------------
plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title(f"{label} (Confidence: {prediction:.2f})")
plt.axis("off")
plt.show()
