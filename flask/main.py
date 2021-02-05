from flask import Flask, render_template, request, redirect, url_for
from wtforms import Form, TextAreaField, validators, FormField
from wtforms.fields import FieldList
from flask_wtf import FlaskForm

import jinja2
env = jinja2.Environment()
env.globals.update(zip=zip)

import os
import joblib
import numpy as np
import pandas as pd

# App config
app = Flask(__name__)
app.config["SECRET_KEY"] = "RandomString"
app.jinja_env.globals.update(zip=zip) # enable zipping lists in templates

# Global variables, input files
cur_dir = os.path.dirname(__file__)
clf = joblib.load(os.path.join(cur_dir, '..', 'jupyter',
                                    'cardio_model.pkl'))
pipeline = joblib.load(os.path.join(cur_dir, '..', 'jupyter',
                                    'pipeline.pkl'))
features = joblib.load(os.path.join(cur_dir, '..', 'jupyter',
                                    'features.pkl'))
print("FEATURES", features)

def classify(params):
    print("classify input shape:", params.shape)
    label = {1: 'death', 0: 'survive'}
    params = pd.DataFrame(params, columns=['time', 'serum_creatinine'])
    X = pipeline.transform(params)
    y = clf.predict(X)
    return [label[score] for score in y]

class OneParamForm(FlaskForm):
    field = TextAreaField("Value")

class InputDataForm(FlaskForm):
    fields = FieldList(FormField(OneParamForm), min_entries=len(features))

    #params = {f : TextAreaField(f"Please provide {f} value") for f in features}
    #params = FieldList(TextAreaField(f"Provide next feature"))
    # print("Params:", para)
    # param1 = TextAreaField('It has been how many days after the event'
    #                        '("time" feature)?',
    #                        [validators.DataRequired()])
    # param2 = TextAreaField('Serum creatinine [units]',
    #                        [validators.DataRequired()])

    def validate(self):
        # TODO: implement validation (check if numerical)
        # try:
        #     float(self.param1.data)
        #     float(self.param2.data)
        # except ValueError:
        #     print("You must enter numerical value!")
        #     return redirect(url_for('index'))
        #     #return False
        # print("Super valid")
        return super().validate()



@app.route('/')
def index():
    form = InputDataForm(request.form)
    return render_template('input_data.html', form=form, labels=features)
    #return render_template('input_data.html',
    #                      form_features=zip(form, features))



@app.route('/results', methods=['POST'])
def results():
    print("Invoked results function")
    form = InputDataForm(request.form)
    if request.method == 'POST' and form.validate():
        param1 = request.form['param1']
        param2 = request.form['param2']
        X = np.array([[param1, param2]]).astype(float)
        y = classify(X)
        return render_template('results.html',
                               params=(param1, param2), outcome=y)
    return render_template('input_data.html', form=form)
if __name__ == '__main__':
    app.run(debug=True)
    print('clf', clf)
    print('pipe', pipeline)

