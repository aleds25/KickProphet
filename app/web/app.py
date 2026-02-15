import sys
import os
from flask import Flask, render_template, request, jsonify

# Add project root to sys.path to allow imports from app.src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"DEBUG: sys.path[0] = {sys.path[0]}")


from app.src.predict import predict_success
from app.src.config import DATA_DIR
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/categories')
def get_categories():
    try:
        categories_path = os.path.join(DATA_DIR, 'category_hierarchy.json')
        with open(categories_path, 'r') as f:
            categories = json.load(f)
        return jsonify(categories)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Prepare data for predict_success
        project_data = {
            'name': data.get('name'),
            'blurb': data.get('blurb'),
            'goal': float(data.get('goal')),
            'duration_days': int(data.get('duration_days')),
            'category': data.get('category'),
            'sub_category': data.get('sub_category'),
            'country': data.get('country'),
            'currency': data.get('currency'),
            'has_video': int(data.get('has_video', False)),
            'prelaunch_activated': int(data.get('prelaunch_activated', False)),
            'preparation_days': float(data.get('preparation_days', 0)),
            'launch_date': data.get('launch_date')
        }

        result = predict_success(project_data)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
