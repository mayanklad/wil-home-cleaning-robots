from flask import Flask, request, jsonify, render_template
import pandas as pd
from dill import dump, load
import os

app = Flask(__name__)

def recommend_cleaning(surface_type, room_size, dirtiness_level):    
    df_input = pd.DataFrame({'surface_type': [surface_type], 'room_size': [room_size], 'dirtiness_level': [dirtiness_level]})
    
    with open(os.path.join(app.root_path, 'ct_lr_pipeline.dill.pkl'),'rb') as file:
        ct_pipeline = load(file)
    
    with open(os.path.join(app.root_path, 'ce_lr_pipeline.dill.pkl'),'rb') as file:
        ce_pipeline = load(file)
    
    time_prediction = ct_pipeline.predict(df_input)[0]
    efficiency_prediction = ce_pipeline.predict(df_input)[0]
    
    return time_prediction, efficiency_prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    surface_type = request.form['surface_type']
    room_size = int(request.form['room_size'])
    dirtiness_level = int(request.form['dirtiness_level'])

    time_prediction, efficiency_prediction = recommend_cleaning(surface_type, room_size, dirtiness_level)

    return render_template('result.html', time_prediction=time_prediction, efficiency_prediction=efficiency_prediction)


if __name__ == '__main__':
    app.run(debug=True)
