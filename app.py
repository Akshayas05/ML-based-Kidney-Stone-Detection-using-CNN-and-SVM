from flask import Flask, render_template, request, redirect, url_for, flash, session
import numpy as np
import joblib
import os
import cv2
import sqlite3
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from utils.gradcam import generate_heatmap
from utils.severity import estimate_severity, natural_suggestions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.stdout.reconfigure(encoding='utf-8')

def hybrid_decision(severity, ct_label, confidence):

    if severity == "HIGH" and ct_label == "Kidney Stone Detected":
        return "Strong Confirmation of Kidney Stone Formation"

    elif severity == "HIGH" and ct_label == "Normal Kidney":
        return "High Biochemical Risk. Further Radiological Evaluation Recommended."

    elif severity == "LOW" and ct_label == "Kidney Stone Detected":
        return "Possible Early Stage Stone. Clinical Correlation Advised."

    elif severity == "LOW" and ct_label == "Normal Kidney":
        return "Low Risk. Maintain Hydration and Routine Monitoring."

    else:
        return "Moderate Risk. Follow-up Examination Suggested."

def generate_ai_explanation(severity, prediction, confidence):

    if prediction == "Kidney Stone Detected":
        return f"""
        The CT scan analysis detected calcified high-density regions 
        with a confidence level of {confidence}%. Combined with 
        clinical severity classified as {severity}, this indicates 
        probable kidney stone formation.
        """
    else:
        return f"""
        No abnormal calcified density was detected in the CT scan.
        Clinical severity level is {severity}. Preventive hydration 
        and periodic monitoring are recommended.
        """
    
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        type TEXT,
        inputs TEXT,
        prediction TEXT,
        confidence REAL,
        severity TEXT,
        explanation TEXT,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()

app = Flask(__name__)
app.config["SECRET_KEY"] = "supersecretkey"
init_db()
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["UPLOAD_FOLDER"] = "static/uploads"

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ---------------- MODELS ----------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    inputs = db.Column(db.String(500), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=True) # For CT
    severity = db.Column(db.String(50), nullable=True) # For Clinical
    explanation = db.Column(db.String(500), nullable=True) # For CT mainly, or added to Clinical
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load ML models
clinical_model = joblib.load("clinical_svm_model.pkl")
scaler = joblib.load("scaler.pkl")
ct_model = load_model("ct_vgg16_stone_model.h5")

IMG_SIZE = 224

with app.app_context():
    db.create_all()

@app.route("/")
def home():
    return render_template("home.html")

# ---------------- AUTHENTICATION ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("home"))
        flash("Invalid username or password", "danger")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if User.query.filter_by(username=username).first():
            flash("Username already exists", "warning")
            return redirect(url_for("register"))
        
        new_user = User(username=username, password=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        flash("Account created! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ---------------- CLINICAL ----------------
@app.route("/clinical", methods=["GET", "POST"])
@login_required
def clinical():
    if request.method == "POST":
        gravity = float(request.form["gravity"])
        ph = float(request.form["ph"])
        osmo = float(request.form["osmo"])
        cond = float(request.form["cond"])
        urea = float(request.form["urea"])
        calc = float(request.form["calc"])

        data = np.array([[gravity, ph, osmo, cond, urea, calc]])
        data_scaled = scaler.transform(data)

        prob = clinical_model.predict_proba(data_scaled)[0][1]

        if prob > 0.3:   # you can tune this (0.3–0.5)
            pred_text = "Kidney Stone Detected"
        else:
            pred_text = "No Stone"  
        severity = estimate_severity(gravity, ph, osmo, calc)
        suggestions = natural_suggestions(severity)
        
        
        # Save to History
        new_entry = History(
            user_id=current_user.id,
            type="Clinical",
            inputs=f"Gravity:{gravity}, pH:{ph}, Osmo:{osmo}, Cond:{cond}, Urea:{urea}, Calc:{calc}",
            prediction=pred_text,
            severity=severity,
            confidence=None, # Not applicable for SVM in this setup usually, or use probability if available
            explanation="Based on clinical features."
        )
        db.session.add(new_entry)
        db.session.commit()

        return render_template("result.html",
                               type="clinical",
                               prediction=pred_text,
                               severity=severity,
                               suggestions=suggestions,
                               report_id=new_entry.id)

    return render_template("clinical.html")

# ---------------- CT SCAN ----------------
@app.route("/ct", methods=["GET", "POST"])
@login_required
def ct():
    if request.method == "POST":
        file = request.files["image"]
        if not file:
            return "No file uploaded", 400
            
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_norm = img / 255.0
        img_input = np.expand_dims(img_norm, axis=0)

        prediction_value = ct_model.predict(img_input, verbose=0)[0][0]

        if prediction_value > 0.5:
            label = "Kidney Stone Detected"
            confidence = round(prediction_value * 100, 2)
            explanation = "High-intensity region detected in CT scan indicating possible calcified stone formation."
        else:
            label = "Normal Kidney"
            confidence = round((1 - prediction_value) * 100, 2)
            explanation = "No abnormal calcified density detected in the CT scan."

        heatmap_path = generate_heatmap(ct_model, path, IMG_SIZE)
        
        # Save to History
        new_entry = History(
            user_id=current_user.id,
            type="CT Scan",
            inputs=f"Image: {file.filename}",
            prediction=label,
            confidence=confidence,
            explanation=explanation,
            severity="N/A"
        )
        db.session.add(new_entry)
        db.session.commit()

        return render_template("result.html",
                       type="ct",
                       prediction=label,
                       confidence=confidence,
                       explanation=explanation,
                       original_image=path,
                       heatmap_image=heatmap_path,
                       report_id=new_entry.id)

    return render_template("ct.html")

# ---------------- Hybrid ----------------

@app.route("/hybrid", methods=["GET", "POST"])
@login_required
def hybrid():

    if request.method == "POST":

        # ---- Clinical Inputs ----
        gravity = float(request.form["gravity"])
        ph = float(request.form["ph"])
        osmo = float(request.form["osmo"])
        cond = float(request.form["cond"])
        urea = float(request.form["urea"])
        calc = float(request.form["calc"])

        clinical_data = np.array([[gravity, ph, osmo, cond, urea, calc]])
        clinical_scaled = scaler.transform(clinical_data)

        severity = estimate_severity(gravity, ph, osmo, calc)

        # ---- CT Image ----
        file = request.files["image"]
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        img_norm = img / 255.0
        img_input = np.expand_dims(img_norm, axis=0)

        ct_prediction_value = ct_model.predict(img_input, verbose=0)[0][0]
        ct_label = "Kidney Stone Detected" if ct_prediction_value > 0.5 else "Normal Kidney"
        ct_confidence = round(ct_prediction_value * 100, 2)

        heatmap_path = generate_heatmap(ct_model, path, 224)

        # ---- Hybrid Decision ----
        hybrid_result = hybrid_decision(severity, ct_label, ct_confidence)
        ai_explanation = generate_ai_explanation(severity, ct_label, ct_confidence)
        
        new_entry = History(
            user_id=current_user.id,
            type="Hybrid",
            inputs="Clinical + CT Combined",
            prediction=hybrid_result,
            confidence=ct_confidence,
            severity=severity,
            explanation=ai_explanation
        )
        db.session.add(new_entry)
        db.session.commit()
        
        return render_template("hybrid_result.html",
                               severity=severity,
                               ct_label=ct_label,
                               confidence=ct_confidence,
                               heatmap_image=heatmap_path,
                               original_image=path,
                               hybrid_result=hybrid_result,
                               ai_explanation=ai_explanation)

    return render_template("hybrid.html")
    

# ---------------- HISTORY & REPORTS ----------------
@app.route("/history")
@login_required
def history():
    records = History.query.filter_by(user_id=current_user.id).order_by(History.timestamp.desc()).all()
    return render_template("history.html", records=records)

@app.route("/report/<int:id>")
@login_required
def report_page(id):
    record = History.query.get_or_404(id)
    if record.user_id != current_user.id:
        return "Unauthorized", 403
    return render_template("report.html", record=record)

if __name__ == "__main__":
    if not os.path.exists("static/uploads"):
        os.makedirs("static/uploads")
    if not os.path.exists("static/heatmaps"):
        os.makedirs("static/heatmaps")
    app.run(debug=True)