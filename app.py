# app.py – Flask prediction app (skeleton)
from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
def index():
    return "Bankruptcy Forecaster – coming soon!"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
