import flask 
from flask import Flask, render_template, request
import joblib
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction' , methods = ['GET' , 'POST'])
def prediction():
    if request.method == 'POST':
        rev = request.form.get('reviews')

        data_point = [rev]

        model = joblib.load('best_models/save_filesvc.pkl')

        prediction = model.predict(data_point)

        if prediction == 'Positive':
            result = "😄 Positive Review 😄"
        else:
            result = "☹️ Negative Review ☹️"

        return render_template('output.html', prediction=result)

    return render_template('output.html', prediction="")

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")