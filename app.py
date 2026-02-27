from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    input_data = {
        'year': int(request.form['year']),
        'km_driven': int(request.form['km_driven']),
        'seats': int(request.form['seats']),
        'fuel': request.form['fuel'],
        'seller_type': request.form['seller_type'],
        'transmission': request.form['transmission'],
        'owner': request.form['owner']
    }

    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)

    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)

    return f"Predicted Price: ₹{int(prediction[0])}"

if __name__ == "__main__":
    app.run(debug=True)