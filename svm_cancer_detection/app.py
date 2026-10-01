import pickle
import json
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load artifacts on startup
with open("model.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)
with open("sample_data.json", "r") as f:
    sample_data = json.load(f)

@app.route("/")
def home():
    return render_template("index.html", 
                           features=selected_features,
                           sample_data=sample_data)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data_in = request.json if request.is_json else request.form
        
        # Extract exactly the 5 selected features in exact order
        input_features = []
        for feature in selected_features:
            val = data_in.get(feature)
            if val is None or str(val).strip() == "":
                return jsonify({"error": f"Missing value for {feature}"}), 400
            input_features.append(float(val))
            
        input_array = np.array(input_features).reshape(1, -1)
        
        # Scale inputs using the saved scaler
        input_scaled = scaler.transform(input_array)
        
        # Predict using SVM
        prediction = svm_model.predict(input_scaled)
        probability = svm_model.predict_proba(input_scaled)[0]
        
        # Target mapping: 0 = Malignant, 1 = Benign
        result = "Malignant" if prediction[0] == 0 else "Benign"
        conf = float(probability[prediction[0]]) * 100
        
        if request.is_json:
            return jsonify({
                "prediction": result,
                "confidence": round(conf, 2)
            })
        else:
            return render_template("index.html", 
                                   features=selected_features,
                                   sample_data=sample_data,
                                   prediction_text=f"Prediction: {result}", 
                                   confidence=f"{round(conf, 2)}%")
            
    except Exception as e:
        if request.is_json:
            return jsonify({"error": str(e)}), 400
        return render_template("index.html", 
                               features=selected_features,
                               sample_data=sample_data,
                               error=str(e))

if __name__ == "__main__":
    app.run(debug=True, port=5002)
