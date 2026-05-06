from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "expense_model.joblib")
model = joblib.load(MODEL_PATH)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    # check content type is json
    data = request.get_json(silent=True)

    # if body is not json at all
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    # if description key is missing
    if "description" not in data:
        return jsonify({"error": "Missing description field."}), 400

    description = data["description"]

    # if description is empty or only spaces
    if not description or not description.strip():
        return jsonify({"error": "Description cannot be empty."}), 400

    # if description is too short
    if len(description.strip()) < 3:
        return jsonify({"error": "Description too short."}), 400

    # if description is too long
    if len(description) > 300:
        return jsonify({"error": "Description too long."}), 400

    # run the model
    category = model.predict([description])[0]
    proba = model.predict_proba([description])[0]
    confidence = round(float(max(proba)), 4)

    return jsonify({
        "description": description,
        "category": category,
        "confidence": confidence
    }), 200


@app.route("/categories", methods=["GET"])
def categories():
    category_list = sorted(model.classes_.tolist())
    return jsonify({
        "categories": category_list,
        "total": len(category_list)
    }), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)