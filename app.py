from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np
import os
import zipfile
import cv2
import scipy

# Initialize the Flask app
app = Flask(__name__)

# Function to extract static.zip at runtime
def setup_static():
    static_dir = os.path.join(os.getcwd(), 'static')
    zip_path = os.path.join(os.getcwd(), 'static.zip')

    if not os.path.exists(static_dir):  # Extract only if not already extracted
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.getcwd())

# Call the setup function
setup_static()

# Load the saved models
diabetes_model = joblib.load(os.path.join('static', 'diabetes_rf_model.pkl'))
tumor_model = joblib.load(os.path.join('static', 'tumor_detection_svm_model.pkl'))
heart_attack_model = joblib.load(os.path.join('static', 'heart_model.pkl'))

# Feature extraction for tumor detection
def extract_features(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Example code to extract features (actual logic will vary based on your feature extraction):
    mean = np.mean(img)
    variance = np.var(img)
    std_dev = np.std(img)
    skewness = scipy.stats.skew(img.flatten())
    kurtosis = scipy.stats.kurtosis(img.flatten())

    # Placeholder for second-order features
    contrast, energy, asm, entropy, homogeneity, dissimilarity, correlation, coarseness = (0,)*8

    # Create a numpy array of all features
    features = np.array(
        [mean, variance, std_dev, skewness, kurtosis, contrast, energy, asm, entropy, homogeneity, dissimilarity,
         correlation, coarseness])

    return features

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        # Form data
        gender = 1 if request.form['gender'].lower() == 'male' else 0
        age = int(request.form['age'])
        hypertension = 1 if request.form['hypertension'].lower() == 'yes' else 0
        heart_disease = 1 if request.form['heart_disease'].lower() == 'yes' else 0
        smoking_history = int(request.form['smoking_history'])
        bmi = float(request.form['bmi'])
        HbA1c_level = float(request.form['HbA1c_level'])
        blood_glucose_level = float(request.form['blood_glucose_level'])

        input_features = np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]])
        prediction = diabetes_model.predict(input_features)
        result = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes"
        return render_template('diabetes.html', result=result)
    return render_template('diabetes.html')

@app.route('/tumor', methods=['GET', 'POST'])
def tumor():
    if request.method == 'POST':
        if 'image' not in request.files or request.files['image'].filename == '':
            return redirect(request.url)

        file = request.files['image']
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)

        features = extract_features(file_path).reshape(1, -1)
        prediction = tumor_model.predict(features)
        probability = tumor_model.predict_proba(features)

        if prediction[0] == 1:
            result = f"Tumor detected with {probability[0][1] * 100:.2f}% confidence."
        else:
            result = f"No tumor detected with {probability[0][0] * 100:.2f}% confidence."

        return render_template('tumor.html', result=result, image_path=file.filename)
    return render_template('tumor.html')

@app.route('/heart_attack', methods=['GET', 'POST'])
def heart_attack():
    if request.method == 'POST':
        # Form data
        age = int(request.form['age'])
        sex = 1 if request.form['sex'].lower() == 'male' else 0
        cholesterol = float(request.form['cholesterol'])
        systolic_bp, diastolic_bp = map(float, request.form['blood_pressure'].split('/'))
        heart_rate = float(request.form['heart_rate'])
        diabetes = int(request.form['diabetes'])
        family_history = int(request.form['family_history'])
        smoking = int(request.form['smoking'])
        obesity = int(request.form['obesity'])
        alcohol = int(request.form['alcohol'])
        exercise_hours = float(request.form['exercise_hours'])
        diet = int(request.form['diet'])
        previous_heart_problems = int(request.form['previous_heart_problems'])
        medication_use = int(request.form['medication_use'])
        stress_level = float(request.form['stress_level'])
        sedentary_hours = float(request.form['sedentary_hours'])
        income = float(request.form['income'])
        bmi = float(request.form['bmi'])
        triglycerides = float(request.form['triglycerides'])
        physical_activity_days = int(request.form['physical_activity_days'])
        sleep_hours = float(request.form['sleep_hours'])

        input_features = np.array([[age, sex, cholesterol, systolic_bp, diastolic_bp, heart_rate, diabetes,
                                    family_history, smoking, obesity, alcohol, exercise_hours, diet,
                                    previous_heart_problems, medication_use, stress_level, sedentary_hours,
                                    income, bmi, triglycerides, physical_activity_days, sleep_hours]])
        prediction = heart_attack_model.predict(input_features)
        result = "Heart Attack Risk Detected" if prediction[0] == 1 else "No Heart Attack Risk Detected"
        return render_template('heart_attack.html', result=result)
    return render_template('heart_attack.html')

if __name__ == '__main__':
    app.run(debug=True)
