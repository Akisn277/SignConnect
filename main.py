from flask import Flask, jsonify, render_template
from Integration.integration import get_combined_output

app = Flask(__name__)

@app.route("/api/live")
def live():
    return jsonify(get_combined_output())

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)