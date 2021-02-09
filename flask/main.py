import os

import jinja2  # Needed for templates configuration
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import validators, FormField
from wtforms.fields import FieldList, SelectField, DecimalField

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

# Boolean and non-boolean features are not intertwined (the booleans are at the end of the _features_).
# This makes for an easy form display in input_data.html, where two loops are used for these categories.
# One has to be careful to show them there in the correct order, required by the ML algorithm.
bool_features = [name for name in features if features[name]['is_binary']]
non_bool_features = [name for name in features if not features[name]['is_binary']]

features['sex']['text'] = 'Male?'


def classify(params):
    print("classify input shape:", params.shape)
    label = {1: 'death', 0: 'survive'}
    df = pd.DataFrame(params, columns=features)
    print("Data Frame\n", df)
    y = clf.predict(df)
    return [label[score] for score in y]


class BinaryParamForm(FlaskForm):
    field = SelectField("", choices=[(1.0, "Yes"), (0.0, "No")])


class DecimalParamForm(FlaskForm):
    field = DecimalField("", [validators.DataRequired()])


class InputDataForm(FlaskForm):
    non_bool_fields = FieldList(FormField(DecimalParamForm), min_entries=len(non_bool_features))
    bool_fields = FieldList(FormField(BinaryParamForm), min_entries=len(bool_features))


@app.route('/')
def index():
    form = InputDataForm()
    return render_template('input_data.html',
                           form=form,
                           non_bool_names=non_bool_features,
                           bool_names=bool_features,
                           labels=features)


@app.route('/results', methods=['POST'])
def results():
    print("Invoked results function")
    form = InputDataForm()
    if request.method == 'POST':
        fields = request.form
        print(fields)
        field_contents = [fields[x] for x in fields.keys() if 'fields-' in x and 'token' not in x]
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
