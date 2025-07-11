import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
print("Preparing Script Just 1 Min...")

'''
Please put the model named "TASK_B.h5" in the models/ directory.
The model can be found at the release of this repository 
https://github.com/Circuit-Overtime/CNN_Facecom_2/releases/tag/publish101
'''


# ======= Configuration =======
EMBEDDING_MODEL_PATH = "models/TASK_B.h5"
# TEST_FOLDER = r"E:\CNN_vedic\Data\Task_B\val"
TEST_FOLDER= "Task_B/val"
IMG_SIZE = (224, 224)
THRESHOLD = 0.945

# ======= Load Model =======
print("🔍 Loading embedding model...")
embedding_model = load_model(EMBEDDING_MODEL_PATH, compile=False)

# ======= Preprocess Image =======
def preprocess_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img = img_to_array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return np.expand_dims(img, axis=0)

# ======= Get Embedding =======
def get_embedding(image_path):
    image = preprocess_image(image_path)
    embedding = embedding_model.predict(image, verbose=0)[0]
    return embedding / np.linalg.norm(embedding)

# ======= Match Function =======
def is_match(reference_paths, test_path, threshold=THRESHOLD):
    test_embedding = get_embedding(test_path)
    for ref_path in reference_paths:
        ref_embedding = get_embedding(ref_path)
        distance = np.linalg.norm(ref_embedding - test_embedding)
        print(f"🔍 {os.path.basename(test_path)} vs {os.path.basename(ref_path)} — Distance: {distance:.4f}")
        if distance < threshold:
            return True
    return False

# ======= Evaluate Folder =======
def evaluate_face_matching(test_root):
    total = 0
    correct = 0

    identities = sorted(os.listdir(test_root))
    for identity in identities:
        identity_path = os.path.join(test_root, identity)
        if not os.path.isdir(identity_path):
            continue

        # Get all image files (excluding the distortion folder)
        all_files = [
            os.path.join(identity_path, f)
            for f in os.listdir(identity_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        distortion_folder = os.path.join(identity_path, "distortion")

        # Exclude distortion folder files
        reference_images = [f for f in all_files if "distortion" not in f.lower()]

        if not reference_images or not os.path.isdir(distortion_folder):
            print(f"⚠️ Skipping: {identity} (missing reference images or distortion folder)")
            continue

        distorted_images = [
            os.path.join(distortion_folder, f)
            for f in os.listdir(distortion_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        for test_img in distorted_images:
            total += 1
            if is_match(reference_images, test_img):
                print("✅ MATCH")
                correct += 1
            else:
                print("❌ NO MATCH")

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n📊 Final Results:")
    print(f"Total Pairs: {total}")
    print(f"Correct Matches: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

# ======= Run Evaluation =======
if __name__ == "__main__":
    evaluate_face_matching(TEST_FOLDER)
