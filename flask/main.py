from flask import Flask, render_template, request, redirect, url_for
from wtforms import Form, TextAreaField, validators, FormField
from wtforms.fields import FieldList
from flask_wtf import FlaskForm

import jinja2 # Needed for templates configuration

import os
import joblib
import numpy as np
import pandas as pd

# App config
app = Flask(__name__)
app.config["SECRET_KEY"] = "RandomString"

# Enable zipping lists in templates
app.jinja_env.globals.update(zip=zip)
env = jinja2.Environment()
env.globals.update(zip=zip)

# Global variables, input files
cur_dir = os.path.dirname(__file__)
# Model (pipeline with all preprocessing needed)
clf = joblib.load(os.path.join(cur_dir, '..', 'jupyter',
                                    'cardio_model.pkl'))
features = joblib.load(os.path.join(cur_dir, '..', 'jupyter',
                                    'features.pkl'))
print("FEATURES", features)
#features = features[:2] # debug

def classify(params):
    print("classify input shape:", params.shape)
    label = {1: 'death', 0: 'survive'}
    df = pd.DataFrame(params, columns=features)
    print("Data Frame\n", df)
    y = clf.predict(df)
    return [label[score] for score in y]

class OneParamForm(FlaskForm):
    field = TextAreaField("Value:", [validators.DataRequired()])


class InputDataForm(FlaskForm):
    fields = FieldList(FormField(OneParamForm), min_entries=len(features))

    # def validate(self):
    #     # TODO: implement validation (check if numerical) or use boolean, numerical form fields instead
    #     # try:
    #     #     float(self.param1.data)
    #     #     float(self.param2.data)
    #     # except ValueError:
    #     #     print("You must enter numerical value!")
    #     #     return redirect(url_for('index'))
    #     #     #return False
    #     # print("Super valid")
    #     return super().validate()



@app.route('/')
def index():
    form = InputDataForm()
    return render_template('input_data.html', form=form, labels=features)

@app.route('/results', methods=['POST'])
def results():
    print("Invoked results function")
    form = InputDataForm()
    print("Form validate?", form.validate())
    if request.method == 'POST':# and form.validate():
        fields = request.form
        field_contents = list({ x : fields[x] for x in fields.keys() if 'fields-' in x and 'token' not in x}.values())
        print(field_contents)
        X = np.array([field_contents]).astype(float)
        print(X)
        y = classify(X)[0]
        print("Results:\n", y)
        return render_template('results.html',
                               fields=field_contents, features=features, outcome=y)
    print("Form not validated")
    return render_template('input_data.html', form=form, labels=features)


if __name__ == '__main__':
    app.run(debug=True)

