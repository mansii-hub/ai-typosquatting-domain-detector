from flask import Flask, request, jsonify
from flask_cors import CORS

from scripts.check_domain import analyze_domain

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json or {}
    domain = data.get("domain") or data.get("url")
    if not domain:
        return jsonify({"error": "Please provide a domain or url field"}), 400

    result = analyze_domain(domain)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)