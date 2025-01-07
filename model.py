import pickle
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

df = pd.read_csv('df_HasC.csv')
X = df[['npol_auto', 'client_sex', 'client_age', 'lic_age', 'client_nother', 'cities2', 'north', 'rest']]
y = df[['nclaims_md', 'cost_md']]

#X = [[1, 25, 18,9,2,4,8,5], [0, 30, 10,4,2,6,3,8], [1, 45, 3,1,3,7,5,9]]  # ตัวแปร input เช่น gender, age, rest
#y = [[2, 500], [3, 600], [1, 300]]  # ตัวแปร target เช่น claims frequency, claims severity

# สร้างโมเดล
model = RandomForestRegressor()
model.fit(X, y)

# บันทึกโมเดล
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)




from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# โหลดโมเดล
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # รับข้อมูล JSON จาก frontend
    npol = data['npol']
    gender = data['gender']
    age = data['age']
    lic_age = data['lic_age']
    client_norther = data['client_norther']
    city = data['city']
    North = data['North']
    rest = data['rest']

    # เตรียม input ให้กับโมเดล
    input_data = [[npol,gender, age,lic_age,client_norther,city,North, rest]]
    prediction = model.predict(input_data)
    
    claims_frequency = prediction[0][0]
    claims_severity = prediction[0][1]

    return jsonify({
        'claims_frequency': claims_frequency,
        'claims_severity': claims_severity
    })

if __name__ == '__main__':
    app.run(debug=True)



