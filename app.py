from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from utils import preprocess_image

app = Flask(__name__)
model = load_model("model/mnist_cnn.h5")

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = preprocess_image(file)
            pred = model.predict(img)
            prediction = int(pred.argmax())
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
