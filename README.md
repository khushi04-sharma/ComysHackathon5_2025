# COMSYS-Hackathon-5,2025

## Quick SetUp Procedure

### 1. Setup Environment (via PyCharm )

The easiest way to run the models is by using 3.10 virtual environment (tf-gpu-env). This ensures seamless compatibility with TensorFlow-GPU and other dependencies — with zero manual installations

> 🧠 Tested on:  <br>
   ✅ Windows 11 + PyCharm + WSL <br>
   ✅ Ubuntu 22.04 (x86_64, AMD CPU, with NVIDIA GPU) <br>
   ✅ macOS (M1/M2 not recommended unless using CPU-only) <br>
   
   ⏳ Time to setup: ~30 minutes (with 22 Mbps internet)
  - **WSL2 backend enabled** (for Windows)
  - **GPU support enabled** (if using TensorFlow-GPU)
- Internet connection ≥ **22 Mbps** (see time note below)
- GPU drivers installed:
  - NVIDIA Driver (≥ R515+)
  


### ⚡ Setup Steps

### Clone the repo
```bash
git clone https://github.com/khushi04-sharma/ComysHackathon5_2025.git
```
### 🖥️ If you're on Windows with WSL2:
```bash
wsl bash install.sh
```
### 🐧 If you're on native Linux/macOS:
```bash
bash install.sh
```
### ✅ Try to Activate Existing Virtual Environment
```bash
source tf-gpu-env/bin/activate
```
⚠️ If this command fails, follow the steps below to create and set up a new environment.
#### 🔹Create a New Virtual Environment
**In your project directory:**
```
python3 -m venv tf-gpu-env
```
#### 🔹Activate the Virtual Environment

```
source tf-gpu-env/bin/activate
```
####  🔹Install Project Dependencies
**With the virtual environment active:**

```
pip install -r requirements.txt
```
#### ▶️ Run the test scripts
```bash
python testA.py   # Gender Classification
python testB.py   # Face Verification
```
---


## 2. Download Pretrained Models(No Need to download)
- **Task A (Gender Classification):**  
    Download `TASK_A.h5` from [releases:](https://github.com/khushi04-sharma/ComysHackathon5_2025.git/models)

- **Task B (Face Verification):**  
    Download `TASK_B.h5` from [releases:](https://github.com/khushi04-sharma/ComysHackathon5_2025.git/models) 

---

## 3. Prepare Data( To give path for test in testA.py and testB.py)

- **Task A:**  
    - Place validation/test images in:
        - `Task_A/val/male`
        - `Task_A/val/female`

- **Task B:**  
    - Place validation/test folders in:
        - `Task_B/val`
    - Each identity folder should contain one or many reference image and a `distortion/` subfolder with distorted images.

---

## 4. Run the Scripts

### Task A: Gender Classification

```bash
python testA.py
```

- Evaluates all images in the male and female folders.
- Prints a classification report : precision, recall, F1-score
- Optionally, set `SHOW_GRADCAM = True` in `TEST_A.py` to visualize Grad-CAM overlays (requires GPU).
### Task B: Face Verification

```bash
python testB.py
```

- Evaluates face matching for all identities and their distorted images.
- Prints match results and overall accuracy.

### Output:

**On Run test_A.py**<br>
**Testing Results on Val dataset**

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://github.com/khushi04-sharma/ComysHackathon5_2025/raw/076ed63835873fcc434a2714824288438081ff53/Screenshot%202025-07-11%20143915.png" width="600" alt="Training Metrics">
        <br>
        <em>Training Progress Metrics</em>
      </td>
      <td align="center">
        <img src="https://github.com/khushi04-sharma/ComysHackathon5_2025/raw/076ed63835873fcc434a2714824288438081ff53/Screenshot%202025-07-11%20143953.png" width="600" alt="Validation Results">
        <br>
        <em>Validation Performance</em>
      </td>
    </tr>
  </table>
</div>

**On Run test_B.py**

[Testing Results on Val dataset](https://github.com/khushi04-sharma/ComysHackathon5_2025/blob/c58bea43ebe005b669e12809d00a75aa198abf34/Task_B_result.txt)

---  

##  Challenge Overview 

### TASK A: GENDER CLASSIFICATION👩‍🦰🧓
- **Objective:** 
Develop a robust gender classification model capable of accurately predicting male/female labels from facial images captured under adverse environmental conditions (e.g., low light, shadows, rain, or haze)
- **Dataset Structure:**
```
dataset/
├── train/
│     ├── male/ # 1532 images
│     └── female/ # 394 images
└── val/
     ├── male/ # 317 images
     └── female/ # 105 images
```
- **Model Goal:** 
Train a model to predict gender from faces that generalizes well to non-ideal images—low light, motion blur, or weather effects (binary Classifier).
---
###  Task B: Face Verification
- **Objective:** 
    Build a **face verification system** that reliably matches distorted face images to their correct identities using metric learning, without relying on traditional classification approaches.

### 🌟 Key Challenges
- ✅ Handle **various distortions** (blur, noise, occlusions) in input images  
- ✅ Generalize to **unseen identities** during inference  
- ✅ Maintain high accuracy despite **low-quality references**

---

**Dataset Structure:**
```
Directory Structure  
        Comys_Hackathon5/
        └── Task_B/
            ├── train/
            │   ├── person_1/
            │   │   ├── image1.jpg
            │   │   ├── distortion/
            │   │   │   ├── distorted1.jpg  # Same person, different conditions
            │   ├── person_2/
            │   │   ├── image1.jpg
            │   │   ├── distortion/
            │   │   │   ├── distorted1.jpg
            └── val/  # Same structure as train

```

## 🧠 Model Description:  Gender Classification Using Transfer Learning (VGG19)
This diagram outlines the workflow for a face recognition system using transfer learning. It includes stages such as data preparation, preprocessing with augmentation, training with VGG19, hyperparameter tuning, deployment, and model evaluation.
<div align="center">
  <img src="https://github.com/khushi04-sharma/ComysHackathon5_2025/blob/47b08c2e8b7c01251166ee28796f7ccd497be519/Image/tranfer_learning.png" alt="Distance Formula" width="600" style="max-width:100%; height:auto;"/>
</div>

#### Architecture of the VGG19 Convolutional Neural Network
This image illustrates the detailed architecture of the VGG19 model, including the sequence of convolutional layers, max-pooling operations, fully connected layers (FC1, FC2), and the final softmax classification layer.
<div align="center">
  <img src="https://github.com/khushi04-sharma/ComysHackathon5_2025/blob/47b08c2e8b7c01251166ee28796f7ccd497be519/Image/Screenshot%202025-07-11%20134525.png" alt="Distance Formula" width="400" style="max-width:50%; height:auto;"/>
</div>

#### 🧠VGG19 Fine-Tuning
- Leverages ImageNet-pre-trained VGG19 backbone
- Custom classification head optimized for gender prediction
- Layer-wise learning rate adaptation
#### 🎯 Dynamic Threshold Optimization
- Auto-tunes decision threshold to maximize F1 score
- Validation-set driven optimization
- Threshold range: 0.3-0.7 with 0.01 increments
#### ⚖️ Class Balancing
- WeightedRandomSampler with inverse class frequency
- Batch-level normalization
- Oversampling for minority class (female)
 


### 🏆📈  Performance Metrics
| Metric                   | Value  |
|--------------------------|--------|
| Test Accuracy            | ~96.85%   |
| precision                | 0.9712 |
| Recall                   | 0.9684 |
| F1-Score                 |0.9684  |

#### Evaluation Result 

<div align="center">
  <div style="display: flex; justify-content: space-between;">
    <div style="text-align: center; margin: 0 10px;">
      <img src="https://github.com/khushi04-sharma/ComysHackathon5_2025/blob/47b08c2e8b7c01251166ee28796f7ccd497be519/Image/Vgg19_Training.png" width="400" alt="VGG19 Training Pipeline">
      <p><strong>VGG19 Training Pipeline</strong></p>
    </div>
    <div style="text-align: center; margin: 0 10px;">
      <img src="https://github.com/khushi04-sharma/ComysHackathon5_2025/blob/47b08c2e8b7c01251166ee28796f7ccd497be519/Image/Vgg19_Fine_Tuning.png" width="400" alt="VGG19 Fine-Tuning Architecture">
      <p><strong>VGG19 Fine-Tuning Architecture</strong></p>
    </div>
  </div>
</div>

#### Excepted Output
📁 Found 100 male and 100 female images.(balanced dataset)

📊 Classification Report:
                     precision    recall  f1-score   support

            Male     0.9700    0.9800    0.9750       100
          Female     0.9800    0.9700    0.9750       100

        accuracy                         0.9750       200
       macro avg     0.9750    0.9750    0.9750       200
    weighted avg     0.9750    0.9750    0.9750       200

 ✅ Our trained model with a VGG19 backbone achieved an impressive ~96.85% classification accuracy on the validation/test set for gender classification.

 > 📌 Note: This high accuracy demonstrates the effectiveness  in gender classification tasks, especially when combined with  high-quality and inbalanced datasets.



## 🧠 Model Description: Triplet Network with a ResNet50 backbone for Face Verification

This Triplet Network leverages a ResNet50 backbone to learn discriminative embeddings through metric learning. The core objective is to minimize intra-class distances (pulling similar samples closer in the embedding space) while maximizing inter-class distances (pushing dissimilar samples apart). The model employs triplet loss, which trains on groups of three items(positive,negative,anchor).The optimization ensures that the anchor is always closer to the positive than to the negative by a defined margin :<div align="center">
                     **distance(anchor, positive) < distance(anchor, negative) + margin**. </div>
                     
<div align="center">
  <img src="https://github.com/khushi04-sharma/Comys_Hackathon5_2025_Task_B/blob/b6ae81fc2e1de9c1ce6573e0a750f06dab7aaa0c/Screenshot%202025-07-02%20012446.png" alt="Distance Formula" width="600" style="max-width:100%; height:auto;"/>
</div>

 **Model Goal**
- Learn a similarity-based system that embeds faces such that:
- Similar identities are **close in embedding space**
- Dissimilar faces are **far apart**
  
**Triplet Loss Implementation**

The standard triplet loss with a margin α:

<div align="center">
  <img src="https://github.com/khushi04-sharma/ComysHackathon5_2025/blob/47b08c2e8b7c01251166ee28796f7ccd497be519/Image/formula.png" alt="Distance Formula" width="800" style="max-width:100%; height:auto;"/>
</div>

**Anchor (xa):** The reference image (red).

**Positive (x⁺):** A similar image of the same identity (green).

**Negative (x⁻):** An image of a different identity (blue).

### 🔍 **Why Triplet Loss?**


- **Handles Large Classes:**: Works well when there are thousands/millions of identities (e.g., in face recognition, where each person is a class).  
- **Focuses on Relative Similarity:**: Enforces that a face is closer to all other faces of the same person than to any face of a different person.  
- **Metric Learning**:  It directly optimizes the ["embedding space for distance-based comparisons "](https://www.researchgate.net/publication/357529033_Triplet_Loss)(unlike softmax classification).  


### 🔍 Why ResNet50?

- **Strong Feature Extraction**: Its deep residual layers capture hierarchical facial features (edges → textures → parts → whole face).  
- **Pretrained Advantage**: Pretrained on ImageNet, it already understands generic visual features, speeding up convergence.  
- **Balance of Speed and Accuracy**: Deeper than ["ResNet18"](https://www.researchgate.net/publication/348248500_ResNet_50) but more efficient than ResNet152, making it practical for deployment
 

### 🔍 How They Work Together

- ✔**Input**: Three face images (anchor, positive, negative). 
- ✔**ResNet50**:  Extracts features for each face.  
- ✔ **Embedding Layer**:  Maps features to a low-dimensional space (e.g., 128-D).  
- ✔ **Triplet Loss**:Computes distances and updates the model to satisfy:d(A,P)+α<d(A,N)
                       where : α is a margin (e.g., 0.2). 

### 🛠 Core Architecture
The twin networks (CNNs) encode input face images into high-dimensional embedding vectors using a shared backbone.These embeddings are then compared using Euclidean distance to determine similarity:
<div align="center">
  <img src="https://github.com/khushi04-sharma/ComysHackathon5_2025/blob/626e7c8bec6dfdb052d9304e20aa5df6a9cdc4a5/Image/tripetlossfullimage.png" alt="Distance Formula"/>
  <p>Metric learning with Triplet Loss: Embeddings from shared-weight CNNs are adjusted to cluster similar (anchor/positive) and separate dissimilar (anchor/negative) samples</p>
  
</div>

### 🔄 Verification Workflow
**1. Preprocessing**
- Input: Two face images (Image A and Image B)
- Steps:
  - Face detection & alignment (MTCNN recommended)
  - Resize to `224×224` (ResNet standard input)
  - Normalize pixel values

**2. Feature Extraction**
```
# Pseudocode
embedding_a = resnet50(Image_A)  
embedding_b = resnet50(Image_B)  

1. Two face images are passed through the Triplet Network.
2. Each branch (with shared weights) generates a feature embedding.
3. A distance metric (e.g., Euclidean distance) computes the similarity between embeddings.
4. The result is compared against a predefined **threshold**:
   - If distance < threshold → ✅ Same Person
   - If distance ≥ threshold → ❌ Different Person
```
### 🖼️ Visualizing Training Triplets

Below is a sample visualization of the triplet structure used in training the Triplet Network:

![Alt text](https://github.com/khushi04-sharma/Comys_Hackathon5_2025_Task_B/blob/152dd25b9c8b9d0cd3f6b0113e7aa9c67211828d/positivenegative.png)
<div  align="center"> Anchor-positive pairs (same class) vs. anchor-negative pairs (different classes) for metric learning</div>

- **Anchor**: The reference face image.
- **Positive**: A different image of the *same person* as the anchor.
- **Negative**: An image of a *different person* from the anchor.

This setup enables the model to learn embeddings where:
- Distance(anchor, positive) → **small**
- Distance(anchor, negative) → **large**

Such training ensures that the model can effectively distinguish between similar and dissimilar faces using a distance threshold.

### ⚙️ Model Specifications
| Parameter          | Value                          |
|--------------------|--------------------------------|
| Input Size         | 100×100 RGB                    |
| Base Model         | Custom CNN (4 Conv Blocks)     |
| Embedding Size     | 4096-D (sigmoid-activated)     |
| Loss Function      | Contrastive Loss               |
| Distance Metric    | Euclidean Distance             |

### Hardware Requirements

| Hardware       | Configuration                | Training Time Estimate | Notes                          |
|----------------|------------------------------|------------------------|--------------------------------|
| **High-End GPU** | NVIDIA RTX 3090 (24GB VRAM)  | ~2 hours               | Recommended for full batch size |
| **Mid-Range GPU** | NVIDIA RTX 2080 (8GB VRAM)   | ~3-4 hours            | Reduce batch size to 16        |
| **CPU Only**    | Modern 8-core CPU            | 5-6 hours             | Use batch size 8-12            |
| **Cloud**       | Google Colab Pro (T4/V100)   | 1.5-3 hours           | Free tier may have limitations |

**Notes:**
- Batch size: 32 recommended for GPUs, reduce for lower VRAM
- Training times based on 50k samples dataset
- SSD storage recommended for faster data loading

### 🏆📈  Performance Metrics
| Metric                   | Value  |
|--------------------------|--------|
| Test Accuracy            |~96-97% |
| precision                | 0.9729 |
| Recall                   | 0.9841 |
| F1-Score                 |0.9785  |
|Threshold                 |0.945   |
|micro-averaged F1-score   |0.9785  |

  ✅ Our trained Triplet Network with a ResNet50 backbone achieved an impressive ~97% verification accuracy on the validation/test set.

> 📌 Note: This high accuracy underscores the effectiveness of Triplet Networks in face verification tasks, especially when using embedding-based similarity with well-curated datasets.


## 🤝 Acknowledgements

Developed by [AI-dentifiers](https://github.com/khushi04-sharma/Comys_Hackathon5_2025_Task_B) and contributor.  
For academic/educational use only.


## Contact:
For inquiries about permitted uses or collaborations, please contact: [dollysharma12.ab@gmail.com]



