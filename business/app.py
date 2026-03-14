# ============================================================
#  BizForecast — Flask Backend
#  Author  : Ankur Pratap Singh
#  Run     : python app.py
#  Access  : http://localhost:5000
# ============================================================

from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder='.')

# ── Load model bundle once at startup ──
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'all_models.pkl')

with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)

svr_model          = bundle['svr']
linear_model       = bundle['linear_regression']
logistic_model     = bundle['logistic_regression']
scaler             = bundle['scaler']
features           = bundle['features']

print("✓ Models loaded successfully")
print(f"  Features : {features}")
print(f"  SVR R²   : {bundle['metrics']['svr_r2']}")
print(f"  LR  R²   : {bundle['metrics']['lr_r2']}")
print(f"  Log Acc  : {bundle['metrics']['log_acc']}")


# ── Serve the frontend HTML ──
@app.route('/')
def index():
    return send_from_directory('.', 'sales_dashboard_ui.html')


# ── Prediction Endpoint ──
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features in correct order
        # features = ['Item_MRP','Outlet_Type','Outlet_Age',
        #             'Item_Visibility','Item_Weight','Item_Fat_Content']
        input_values = [
            float(data['Item_MRP']),
            float(data['Outlet_Type']),
            float(data['Outlet_Age']),
            float(data['Item_Visibility']),
            float(data['Item_Weight']),
            float(data['Item_Fat_Content'])
        ]

        X = np.array(input_values).reshape(1, -1)

        # Scale input (mandatory — models were trained on scaled data)
        X_scaled = scaler.transform(X)

        # SVR prediction (continuous sales value)
        svr_pred = float(svr_model.predict(X_scaled)[0])
        svr_pred = max(33.0, round(svr_pred, 2))

        # Linear Regression prediction
        lr_pred  = float(linear_model.predict(X_scaled)[0])
        lr_pred  = max(33.0, round(lr_pred, 2))

        # Logistic Regression prediction (sales category)
        log_class     = int(logistic_model.predict(X_scaled)[0])
        log_proba     = logistic_model.predict_proba(X_scaled)[0].tolist()
        class_labels  = {0: 'Low', 1: 'Medium', 2: 'High'}
        log_label     = class_labels[log_class]

        # Determine category from SVR prediction
        if svr_pred < 1000:
            category = 'Low'
        elif svr_pred < 3000:
            category = 'Medium'
        else:
            category = 'High'

        return jsonify({
            'success'       : True,
            'svr_prediction': svr_pred,
            'lr_prediction' : lr_pred,
            'log_class'     : log_label,
            'log_proba'     : {
                'Low'   : round(log_proba[0], 3),
                'Medium': round(log_proba[1], 3),
                'High'  : round(log_proba[2], 3)
            },
            'category'      : category,
            'features_used' : features
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# ── Model Info Endpoint ──
@app.route('/model-info')
def model_info():
    return jsonify({
        'models': {
            'SVR'               : {'type': 'Regression', 'r2': bundle['metrics']['svr_r2']},
            'Linear Regression' : {'type': 'Regression', 'r2': bundle['metrics']['lr_r2']},
            'Logistic Regression': {'type': 'Classification', 'accuracy': bundle['metrics']['log_acc']}
        },
        'features' : features,
        'train_size': '80% (6818 records)',
        'test_size' : '20% (1705 records)'
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  BizForecast Server Starting...")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
