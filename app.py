from flask import Flask, render_template
import joblib

app = Flask(__name__)

# Load the model and accuracy
model = joblib.load('logistic_model.pkl')
accuracy = joblib.load('accuracy.pkl')

@app.route('/')
def home():
    return render_template('index.html', accuracy=accuracy)

if __name__ == "__main__":
    app.run(debug=True)
