"""
api.py

Standalone minimal API that loads a trained model and serves /predict and /health.
This complements app.py and can be used by dashboard or external systems.
"""
import os
import numpy as np
from flask import Flask, request, jsonify
try:
    from tensorflow.keras.models import load_model
except Exception:
    try:
        from keras.saving import load_model as _kload_model
        load_model = _kload_model
    except Exception:
        try:
            import keras as _k
            load_model = _k.models.load_model
        except Exception:
            def load_model(*args, **kwargs):
                raise RuntimeError("keras/tensorflow unavailable")
import pickle

app = Flask(__name__)
MODEL_PATH = os.environ.get('API_MODEL_PATH', 'models/best_aqi_model.h5')
SCALER_PATH = os.environ.get('API_SCALER_PATH', 'models/scaler.pkl')

_model = None
_scaler = None
_feature_cols = ['PM2.5','PM10','NO2','SO2','CO','O3','Temperature','Humidity','Wind_Speed','Pressure']
_timesteps = int(os.environ.get('API_TIMESTEPS', 24))

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model not found at {MODEL_PATH}")
        _model = load_model(MODEL_PATH)
    return _model

def get_scaler():
    global _scaler
    if _scaler is None and os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH,'rb') as f:
                _scaler = pickle.load(f)
        except Exception:
            _scaler = None
    return _scaler

@app.route('/health')
def health():
    model_exists = os.path.exists(MODEL_PATH)
    return jsonify({'status':'ok','model_exists': model_exists}), 200

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({'error':'missing json payload'}), 400
    features = payload.get('features')
    if features is None:
        return jsonify({'error':'provide features array or dict'}), 400

    # Accept both dict of named features or array
    if isinstance(features, dict):
        X = []
        for c in _feature_cols:
            X.append(float(features.get(c, 0.0)))
        X = np.array([X])
    else:
        X = np.array(features)
        if X.ndim == 1:
            X = X.reshape(1, -1)

    scaler = get_scaler()
    if scaler is not None:
        try:
            Xs = scaler.transform(X)
        except Exception:
            Xs = X / 100.0
    else:
        Xs = X / 100.0

    # repeat for timesteps
    Xs = np.repeat(Xs, _timesteps, axis=0).reshape(1, _timesteps, Xs.shape[-1])

    model = get_model()
    preds = model.predict(Xs)
    preds = np.array(preds)
    if preds.ndim == 3 and preds.shape[-1] == 1:
        preds = preds.reshape(preds.shape[0], preds.shape[1])
    elif preds.ndim > 2:
        preds = preds.reshape(preds.shape[0], preds.shape[1], -1)[:,:,0]

    return jsonify({'predictions': preds.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('API_PORT', 5001)))
