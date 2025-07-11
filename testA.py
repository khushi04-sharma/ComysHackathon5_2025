import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.applications.vgg19 import preprocess_input #type: ignore
from tensorflow.keras.preprocessing.image import img_to_array #type: ignore


'''
Please put the model named "TASK_A.h5" in the models/ directory.
The model can be found at the release of this repository 
https://github.com/Circuit-Overtime/CNN_Facecom_2/releases/tag/publish102
'''

print("Preparing Script Just 1 Min...")
# ========== CONFIG ==========
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
MODEL_PATH = "models/TASK_A.h5"

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Male", "Female"]
THRESHOLD = 0.45
SHOW_GRADCAM = False  

# # Hardcoded folders
# MALE_FOLDER = "Task_A\val\male"
# FEMALE_FOLDER = "Task_A\val\female"
MALE_FOLDER = "Task_A/train/male"
FEMALE_FOLDER = "Task_A/train/female"

# ========== FOCAL LOSS ==========
def focal_loss(gamma=2., alpha=0.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        return alpha * tf.pow(1. - p_t, gamma) * ce
    return loss

# ========== LOAD MODEL ==========
print("🔍 Loading model...")
model = load_model(MODEL_PATH, custom_objects={"loss": focal_loss(gamma=2.0, alpha=0.5)})

# ========== IMAGE PREPROCESS ==========
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"❌ Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = img_to_array(image)
    image = preprocess_input(image)
    return np.expand_dims(image, axis=0)

# ========== PREDICT SINGLE ==========
def predict_image(image_path):
    image = preprocess_image(image_path)
    preds = model.predict(image, verbose=0)[0]
    female_prob = preds[1]
    return int(female_prob > THRESHOLD), female_prob

# ========== GRAD-CAM ==========
def gradcam_visualize(image_path, model, layer_name="block5_conv4", class_index=1):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(rgb_image, IMAGE_SIZE)
    x = preprocess_input(np.expand_dims(input_image.astype(np.float32), axis=0))

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap + 1e-8)

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    colored = (cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(rgb_image, 0.6, colored, 0.4, 0)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.title("Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Overlay: {CLASS_NAMES[class_index]}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ========== EVALUATE BATCH ==========
def evaluate_gender_classification(male_dir, female_dir, show_cam=False):
    y_true, y_pred = [], []
    male_images = sorted([os.path.join(male_dir, f) for f in os.listdir(male_dir) if f.lower().endswith((".jpg", ".png"))])
    female_images = sorted([os.path.join(female_dir, f) for f in os.listdir(female_dir) if f.lower().endswith((".jpg", ".png"))])

    print(f"📁 Found {len(male_images)} male and {len(female_images)} female images.")

    for img_path in male_images:
        pred_class, conf = predict_image(img_path)
        y_true.append(0)
        y_pred.append(pred_class)
        if show_cam and tf.config.list_physical_devices('GPU'):
            gradcam_visualize(img_path, model, class_index=pred_class)

    for img_path in female_images:
        pred_class, conf = predict_image(img_path)
        y_true.append(1)
        y_pred.append(pred_class)
        if show_cam and tf.config.list_physical_devices('GPU'):
            gradcam_visualize(img_path, model, class_index=pred_class)

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# ========== MAIN ==========
if __name__ == "__main__":
    evaluate_gender_classification(MALE_FOLDER, FEMALE_FOLDER, show_cam=SHOW_GRADCAM)
