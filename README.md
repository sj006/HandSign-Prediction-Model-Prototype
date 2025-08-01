# 🤟 ASL Hand Sign Prediction Model

This project is a **prototype** that predicts **alphabets from American Sign Language (ASL)** using hand gestures.

> ⚠️ **Note:**  
> - This is an early prototype trained on a **small dataset**.  
> - It currently supports only **single-hand gestures**.  
> - The performance may vary based on lighting, background, and gesture clarity.

---

## 📌 Features

- 🧠 Predicts **ASL alphabets** from live hand gestures.
- ✋ Supports **only one hand** at a time.
- 🧪 Lightweight and easy to train with your own dataset.

---

## 🗂️ Project Structure

```
📁 ASL-HandSign-Predictor/
├── dataset/                 # Folder containing ASL images for training
├── ModelTrainer.py          # Script to train the model on ASL dataset
├── HandsignPrediction.py    # Script for live hand gesture prediction using webcam
├── asl_model.h5             # Trained model file (generated after training)
├── requirements.txt         # Python dependencies
└── README.md                # You are here!
```

---

## 🚀 How to Use

Follow the steps below to train and run the ASL hand sign prediction model:

### 🔧 Step 1: Install Dependencies

Make sure Python 3 is installed. Then run:

```bash
pip install -r requirements.txt
```

This installs required libraries like OpenCV, MediaPipe, NumPy, etc.

---

### 🧠 Step 2: Train the Model

Use the provided ASL dataset or your own. Place the images inside the `dataset/` folder, organized in subfolders (A-Z).

Run the model training script:

```bash
python ModelTrainer.py
```

This will:
- Load and preprocess the dataset.
- Train the model on the ASL gestures.
- Save the trained model to `model.pkl`.

---

### 📸 Step 3: Run the Prediction

Once training is complete, launch the real-time prediction system:

```bash
python HandsignPrediction.py
```

- Your webcam will open.
- Show a single-hand ASL gesture.
- The system will detect and predict the alphabet on the screen.

---

## 📅 Weekly Tip

If today is, say, the **2nd day of the week** (starting from **Sunday**), try practicing **2 random ASL letters** today!  
Keep building daily — the week starts on Sunday and ends on Saturday. Stay consistent and improve your hand gestures every day.

---

## 💡 Future Improvements

- 🫱🫲 Add support for **two-hand gestures**
- 🧠 Improve accuracy using a **larger dataset**
- 📱 Deploy on web or mobile using **TensorFlow.js** or **Kivy**
- 🧾 Add gesture history and real-time sentence formation

---

## 🧑‍💻 Author

 - SJ006

Made with curiosity and care to explore the possibilities of real-time sign language translation.

---

## 📬 Contributions

Got an idea or improvement? Feel free to fork, modify, and create a pull request!

---