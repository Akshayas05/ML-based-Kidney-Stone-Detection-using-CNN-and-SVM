# NephroFusion AI — Kidney Stone Detection System

NephroFusion AI is a hybrid kidney stone detection system that combines clinical data analysis with CT scan image processing to deliver accurate and reliable predictions. By integrating **Machine Learning (SVM)** and **Deep Learning (CNN)**, the system enhances diagnostic performance and provides risk-based insights.

---

## How This Project Works & Workflow

NephroFusion follows a hybrid pipeline combining structured clinical data and medical imaging:

### 1. Clinical Data Processing (SVM)

- Input parameters: Gravity, pH, Osmolarity, Conductivity, Urea, Calcium
- Data preprocessing using **StandardScaler**
- Class imbalance handled using **SMOTE**
- Prediction using **Support Vector Machine (SVM)**
- Outputs:
  - Stone / No Stone
  - Severity Level (`LOW` / `MEDIUM` / `HIGH`)

### 2. CT Scan Image Processing (CNN - VGG16)

- Image resizing to **224x224**
- Normalization and preprocessing
- Feature extraction using **VGG16**
- Binary classification:
  - Kidney Stone Detected
  - Normal Kidney
- Heatmap generation using **Grad-CAM** for visual explanation

### 3. Hybrid Decision System (NephroFusion Model)

- Combines:
  - Clinical severity
  - CT scan prediction
- Applies rule-based fusion logic
- Produces final output:
  - Strong Confirmation
  - Early Stage Detection
  - Risk Warning

---

## Project Models & Tech Stack

### AI & Machine Learning Models

| Component       | Technology                           |
|-----------------|--------------------------------------|
| Clinical Model  | Support Vector Machine (SVM)         |
| Image Model     | Convolutional Neural Network (VGG16) |
| Explainability  | Grad-CAM Visualization               |

### Data Processing

- **SMOTE** — Handling imbalanced data
- **StandardScaler** — Normalization

### Backend

- Flask (Python)
- SQLite Database
- Authentication system

### Frontend

- HTML, CSS, JavaScript
- Interactive UI with visualization

---

## Dataset

- **Clinical Dataset:** Kidney stone dataset (structured data)
- **CT Scan Dataset:** Medical imaging dataset with stone & normal classes
- Data augmentation applied for improving performance

---

## How and Where to Use This Project

### Use Cases

1. Early detection of kidney stones
2. Clinical decision support system
3. Medical AI research and education
4. Risk prediction and preventive healthcare

---

## How to Run the Project

### Prerequisites

- Python 3.8+
- Required libraries installed
- Browser (Chrome recommended)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Application

```bash
python app.py
```

### Step 3: Open in Browser

```
http://127.0.0.1:5000
```

### Step 4: Use the System

- Register / Login
- Choose:
  - Clinical Prediction
  - CT Scan Prediction
  - Hybrid Prediction
- View results with explanation and visualization

---

## Performance

| Model              | Accuracy |
|--------------------|----------|
| Existing Systems   | ~90%     |
| NephroFusion Model | ~97–98%  |

> Improved accuracy due to the **hybrid learning approach**.


