from flask import Flask, jsonify, render_template
from Integration.integration import get_combined_output
from Voice_to_text.voice_to_text import voicetotxt

app = Flask(__name__)

@app.route("/api/live")
def live():
    return jsonify(get_combined_output())

@app.route("/")
def home():
    return render_template("index.html")
@app.route('/speech', methods=['GET'])
def speech():
    text = voicetotxt()
    return jsonify({"text": text})
if __name__ == "__main__":
    app.run(debug=True)