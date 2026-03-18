from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
from datetime import datetime, timedelta
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.models import model_from_json
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
    try:
        from keras.saving import model_from_json as _k_model_from_json
        model_from_json = _k_model_from_json
    except Exception:
        try:
            from keras.models import model_from_json
        except Exception:
            def model_from_json(*args, **kwargs):
                raise RuntimeError("keras/tensorflow unavailable")
import pickle
from joblib import load as joblib_load
import os
import json
import urllib.request
import urllib.parse
import h5py

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey")

users = {}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users.get(username)
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        email = request.form['email']
        address = request.form['address']

        if username in users:
            flash('Username already exists!', 'warning')
            return redirect(url_for('register'))

        users[username] = {
            'password': password,
            'email': email,
            'address': address
        }

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ===============================================================
# ✅ Hybrid AQIPredictor (uses real CNN-LSTM model if available)
# ===============================================================
class AQIPredictor:
    def __init__(self, model_path='models/aqi_cnn_lstm_model.h5'):
        try:
            mp = os.environ.get('MODEL_PATH') or os.environ.get('AQI_MODEL_PATH') or model_path
            if not os.path.exists(mp):
                candidates = [
                    'models/aqi_cnn_bilstm_attention_model_91.h5'
                ]
                for p in candidates:
                    if os.path.exists(p):
                        mp = p
                        break
                if not os.path.exists(mp) and os.path.exists('models'):
                    for name in os.listdir('models'):
                        if name.endswith('.h5') or name.endswith('.keras'):
                            mp = os.path.join('models', name)
                            break
            try:
                self.model = load_model(mp)
            except Exception:
                try:
                    import keras as k
                    try:
                        from keras.saving import load_model as kload
                        self.model = kload(mp)
                    except Exception:
                        self.model = k.models.load_model(mp)
                except Exception:
                    with h5py.File(mp, 'r') as f:
                        mc = f.attrs.get('model_config')
                        if isinstance(mc, bytes):
                            mc = mc.decode('utf-8')
                        if isinstance(mc, str):
                            cfg = json.loads(mc)
                            def fix(o):
                                if isinstance(o, dict):
                                    if o.get('class_name') == 'DTypePolicy' and isinstance(o.get('config',{}).get('name',None), str):
                                        return o['config']['name']
                                    n = {}
                                    for k,v in o.items():
                                        nk = 'batch_input_shape' if k == 'batch_shape' else k
                                        if nk == 'groups' and not isinstance(v, int):
                                            continue
                                        n[nk] = fix(v)
                                    return n
                                if isinstance(o, list):
                                    return [fix(i) for i in o]
                                return o
                            cfg = fix(cfg)
                            mj = json.dumps(cfg)
                            m = model_from_json(mj)
                            g = None
                            if 'model_weights' in f:
                                g = f['model_weights']
                            else:
                                for k in f.keys():
                                    if 'weight' in k.lower():
                                        g = f[k]
                                        break
                            if g is not None:
                                from tensorflow.python.keras.saving import hdf5_format
                                hdf5_format.load_weights_from_hdf5_group_by_name(g, m.layers)
                            self.model = m
            print(f"✅ Model loaded from {mp}")
            
            try:
                sp = os.environ.get('SCALER_PATH')
                scaler_loaded = False
                if sp and os.path.exists(sp):
                    try:
                        with open(sp, 'rb') as f:
                            self.scaler = pickle.load(f)
                        scaler_loaded = True
                    except Exception:
                        try:
                            self.scaler = joblib_load(sp)
                            scaler_loaded = True
                        except Exception:
                            pass
                if not scaler_loaded:
                    for alt in ['models/scaler.pkl','models/scaler.save','models/scaler.joblib','models/scaler91.save']:
                        if os.path.exists(alt):
                            try:
                                with open(alt, 'rb') as f:
                                    self.scaler = pickle.load(f)
                                scaler_loaded = True
                                break
                            except Exception:
                                try:
                                    self.scaler = joblib_load(alt)
                                    scaler_loaded = True
                                    break
                                except Exception:
                                    pass
                if not scaler_loaded:
                    from sklearn.preprocessing import MinMaxScaler
                    self.scaler = MinMaxScaler()
                    print("⚠️ Using default scaler (scaler not found)")
                else:
                    print("✅ Scaler loaded")
            except:
                from sklearn.preprocessing import MinMaxScaler
                self.scaler = MinMaxScaler()
                print("⚠️ Using default scaler (scaler not found)")
            
            self.use_real_model = True
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            print("📊 Using simulated predictions instead")
            self.use_real_model = False
        
        self.pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        self.weather_params = ['Temperature', 'Humidity', 'Wind_Speed', 'Pressure']

    # ===============================================================
    # 🔹 Prediction Entry Point
    # ===============================================================
    def predict(self, current_data, hours=12):
        if self.use_real_model:
            return self.predict_with_model(current_data, hours)
        else:
            return self.predict_simulated(current_data, hours)

    # ===============================================================
    # 🔹 Real Model Prediction
    # ===============================================================
    def predict_with_model(self, current_data, hours=12):
        predictions = []
        base_time = datetime.now()
        
        req_steps = 24
        req_features = 10
        try:
            shp = getattr(self.model, 'input_shape', None)
            if isinstance(shp, (list, tuple)) and len(shp) >= 3 and isinstance(shp[1], int) and isinstance(shp[2], int):
                req_steps = shp[1]
                req_features = shp[2]
        except:
            pass
        if req_features == 6:
            feature_cols = self.pollutants
        else:
            feature_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
                            'Temperature', 'Humidity', 'Wind_Speed', 'Pressure']
        
        # Prepare model input
        input_data = np.array([[current_data.get(col, 0.0) for col in feature_cols]])
        
        # Scale input
        try:
            scaled_input = self.scaler.transform(input_data)
        except:
            scaled_input = input_data / 100.0  # fallback normalization
        
        model_input = np.repeat(scaled_input, req_steps, axis=0).reshape(1, req_steps, len(feature_cols))
        
        # Model predictions
        model_predictions = self.model.predict(model_input, verbose=0)
        
        for hour in range(min(hours, len(model_predictions[0]))):
            pred_value = model_predictions[0][hour]
            
            # Estimate pollutants from model output
            pm25 = float(current_data.get('PM2.5', 35.0))
            pm10 = float(current_data.get('PM10', 50.0))
            pred_pm25 = pm25 * (1.0 + pred_value * 0.1)
            pred_pm10 = pm10 * (1.0 + pred_value * 0.08)
            pred_no2 = float(current_data.get('NO2', 40.0)) * (1.0 + pred_value * 0.12)
            pred_so2 = float(current_data.get('SO2', 20.0)) * (1.0 + pred_value * 0.06)
            pred_co = float(current_data.get('CO', 1.0)) * (1.0 + pred_value * 0.05)
            pred_o3 = float(current_data.get('O3', 30.0)) * (1.0 + pred_value * 0.15)
            
            # Calculate AQI
            aqi = self.calculate_aqi(pred_pm25, pred_pm10, pred_no2, pred_so2, pred_co, pred_o3)
            category, color, description = self.get_aqi_category(aqi)
            prediction_time = base_time + timedelta(hours=hour+1)
            
            predictions.append({
                'hour': hour + 1,
                'timestamp': prediction_time.strftime('%Y-%m-%d %H:%M:%S'),
                'aqi': round(aqi, 2),
                'category': category,
                'color': color,
                'description': description,
                'confidence_interval': [round(aqi * 0.90, 2), round(aqi * 1.10, 2)],
                'pollutants': {
                    'PM2.5': round(pred_pm25, 2),
                    'PM10': round(pred_pm10, 2),
                    'NO2': round(pred_no2, 2),
                    'SO2': round(pred_so2, 2),
                    'CO': round(pred_co, 3),
                    'O3': round(pred_o3, 2)
                }
            })
        
        return predictions

    # ===============================================================
    # 🔹 Simulated Prediction Fallback
    # ===============================================================
    def predict_simulated(self, current_data, hours=12):
        predictions = []
        base_time = datetime.now()
        
        pm25 = float(current_data.get('PM2.5', 35.0))
        pm10 = float(current_data.get('PM10', 50.0))
        no2 = float(current_data.get('NO2', 40.0))
        so2 = float(current_data.get('SO2', 20.0))
        co = float(current_data.get('CO', 1.0))
        o3 = float(current_data.get('O3', 30.0))

        for hour in range(1, hours + 1):
            time_factor = np.sin(2.0 * np.pi * hour / 24.0)
            noise = np.random.normal(0, 0.05)
            
            pred_pm25 = pm25 * (1.0 + 0.1 * time_factor + noise)
            pred_pm10 = pm10 * (1.0 + 0.08 * time_factor + noise)
            pred_no2 = no2 * (1.0 + 0.12 * time_factor + noise * 1.2)
            pred_so2 = so2 * (1.0 + 0.06 * time_factor + noise * 0.8)
            pred_co = co * (1.0 + 0.05 * time_factor + noise * 0.7)
            pred_o3 = o3 * (1.0 + 0.15 * time_factor + noise * 1.3)
            
            pred_pm25, pred_pm10, pred_no2, pred_so2, pred_co, pred_o3 = map(lambda x: max(x, 0.0),
                                                                             [pred_pm25, pred_pm10, pred_no2, pred_so2, pred_co, pred_o3])
            
            aqi = self.calculate_aqi(pred_pm25, pred_pm10, pred_no2, pred_so2, pred_co, pred_o3)
            category, color, description = self.get_aqi_category(aqi)
            prediction_time = base_time + timedelta(hours=hour)
            
            predictions.append({
                'hour': hour,
                'timestamp': prediction_time.strftime('%Y-%m-%d %H:%M:%S'),
                'aqi': round(aqi, 2),
                'category': category,
                'color': color,
                'description': description,
                'confidence_interval': [round(aqi * 0.90, 2), round(aqi * 1.10, 2)],
                'pollutants': {
                    'PM2.5': round(pred_pm25, 2),
                    'PM10': round(pred_pm10, 2),
                    'NO2': round(pred_no2, 2),
                    'SO2': round(pred_so2, 2),
                    'CO': round(pred_co, 3),
                    'O3': round(pred_o3, 2)
                }
            })
        return predictions

    # ===============================================================
    # 🔹 Utility Methods
    # ===============================================================
    def calculate_aqi(self, pm25, pm10, no2, so2, co, o3):
        aqi_values = []

        # PM2.5 AQI
        if pm25 <= 12.0:
            aqi_pm25 = (50.0 / 12.0) * pm25
        elif pm25 <= 35.4:
            aqi_pm25 = 50 + ((100 - 50) / (35.4 - 12.1)) * (pm25 - 12.1)
        elif pm25 <= 55.4:
            aqi_pm25 = 100 + ((150 - 100) / (55.4 - 35.5)) * (pm25 - 35.5)
        elif pm25 <= 150.4:
            aqi_pm25 = 150 + ((200 - 150) / (150.4 - 55.5)) * (pm25 - 55.5)
        elif pm25 <= 250.4:
            aqi_pm25 = 200 + ((300 - 200) / (250.4 - 150.5)) * (pm25 - 150.5)
        else:
            aqi_pm25 = 300 + ((500 - 300) / (500.4 - 250.5)) * max(pm25 - 250.5, 0)
        aqi_values.append(aqi_pm25)

        # PM10 AQI
        if pm10 <= 54:
            aqi_pm10 = (50.0 / 54.0) * pm10
        elif pm10 <= 154:
            aqi_pm10 = 50 + ((100 - 50) / (154 - 55)) * (pm10 - 55)
        elif pm10 <= 254:
            aqi_pm10 = 100 + ((150 - 100) / (254 - 155)) * (pm10 - 155)
        elif pm10 <= 354:
            aqi_pm10 = 150 + ((200 - 150) / (354 - 255)) * (pm10 - 255)
        elif pm10 <= 504:
            aqi_pm10 = 200 + ((300 - 200) / (504 - 355)) * (pm10 - 355)
        else:
            aqi_pm10 = 300 + ((500 - 300) / (1000 - 505)) * max(pm10 - 505, 0)
        aqi_values.append(aqi_pm10)

        return max(aqi_values)

    def get_aqi_category(self, aqi):
        if aqi <= 50:
            return "Good", "#00E400", "Air quality is satisfactory"
        elif aqi <= 100:
            return "Moderate", "#FFFF00", "Air quality is acceptable"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups", "#FF7E00", "Sensitive groups may experience health effects"
        elif aqi <= 200:
            return "Unhealthy", "#FF0000", "Everyone may begin to experience health effects"
        elif aqi <= 300:
            return "Very Unhealthy", "#8F3F97", "Health alert: everyone may experience serious effects"
        else:
            return "Hazardous", "#7E0023", "Health warnings of emergency conditions"


# ===============================================================
# Flask Routes
# ===============================================================
predictor = AQIPredictor()

@app.route('/index')
def index():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'success': False, 'error': 'Invalid or missing JSON payload.'}), 400
    
    required_fields = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
                       'Temperature', 'Humidity', 'Wind_Speed', 'Pressure']
    
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({'success': False, 'error': f'Missing fields: {missing}'}), 400
    
    parsed = {}
    parsing_errors = {}
    for key in required_fields:
        try:
            parsed[key] = float(data[key])
        except (ValueError, TypeError):
            parsing_errors[key] = f"Could not convert value '{data.get(key)}' to float"
    
    if parsing_errors:
        return jsonify({'success': False, 'error': 'Parsing errors', 'details': parsing_errors}), 400

    try:
        predictions = predictor.predict(parsed, hours=12)
        current_aqi = predictor.calculate_aqi(parsed['PM2.5'], parsed['PM10'], parsed['NO2'],
                                              parsed['SO2'], parsed['CO'], parsed['O3'])
        category, color, description = predictor.get_aqi_category(current_aqi)
        current_status = {
            'aqi': round(current_aqi, 2),
            'category': category,
            'color': color,
            'description': description,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        alerts = []
        alert_threshold = 150.0
        for pred in predictions:
            if pred['aqi'] > alert_threshold:
                alerts.append({
                    'time': pred['timestamp'],
                    'aqi': pred['aqi'],
                    'category': pred['category'],
                    'message': f"Alert: AQI expected to reach {pred['category']} level ({pred['aqi']}) at {pred['timestamp']}"
                })
        
        return jsonify({
            'success': True,
            'current': current_status,
            'predictions': predictions,
            'alerts': alerts
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': 'Prediction failed', 'details': str(e)}), 500


@app.route('/historical', methods=['GET'])
def historical():
    data = []
    base_time = datetime.now() - timedelta(hours=24)
    for hour in range(24):
        time_factor = np.sin(2.0 * np.pi * hour / 24.0)
        aqi = 80 + 30 * time_factor + np.random.normal(0, 10)
        aqi = max(0, min(500, aqi))
        data.append({
            'timestamp': (base_time + timedelta(hours=hour)).strftime('%Y-%m-%d %H:%M:%S'),
            'aqi': round(aqi, 2)
        })
    return jsonify({'success': True, 'historical': data}), 200


@app.route('/autofill_waqi', methods=['GET'])
def autofill_waqi():
    try:
        token = os.environ.get('WAQI_TOKEN', '489b7744e305ef7cd2392b6434db1c0f09668d9a')
        uid = request.args.get('uid')
        city = request.args.get('city')
        if uid:
            url = f"https://api.waqi.info/feed/@{uid}/?token={token}"
        elif city:
            url = f"https://api.waqi.info/feed/{city}/?token={token}"
        else:
            url = f"https://api.waqi.info/feed/here/?token={token}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            waqi = json.loads(resp.read().decode('utf-8'))
        if waqi.get('status') != 'ok':
            return jsonify({'success': False, 'error': 'WAQI status not ok'}), 502
        data = waqi.get('data', {})
        iaqi = data.get('iaqi', {})
        city_info = data.get('city', {}) or {}
        geo_list = city_info.get('geo') or []
        geo_obj = None
        if isinstance(geo_list, (list, tuple)) and len(geo_list) >= 2:
            try:
                geo_obj = {'lat': float(geo_list[0]), 'lon': float(geo_list[1])}
            except Exception:
                geo_obj = None
        def v(key, default=0.0):
            node = iaqi.get(key, {})
            try:
                return float(node.get('v', default))
            except Exception:
                return float(default)
        mapped = {
            'PM2.5': v('pm25', 0.0),
            'PM10': v('pm10', 0.0),
            'NO2': v('no2', 0.0),
            'SO2': v('so2', 0.0),
            'CO': v('co', 0.0),
            'O3': v('o3', 0.0),
            'Temperature': v('t', 25.0),
            'Humidity': v('h', 50.0),
            'Wind_Speed': v('w', 1.0),
            'Pressure': v('p', 1010.0)
        }
        predictions = predictor.predict(mapped, hours=12)
        current_aqi = predictor.calculate_aqi(mapped['PM2.5'], mapped['PM10'], mapped['NO2'],
                                              mapped['SO2'], mapped['CO'], mapped['O3'])
        category, color, description = predictor.get_aqi_category(current_aqi)
        current_status = {
            'aqi': round(current_aqi, 2),
            'category': category,
            'color': color,
            'description': description,
            'timestamp': data.get('time', {}).get('iso') or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return jsonify({
            'success': True,
            'source': {
                'city': city_info.get('name'),
                'station': city_info.get('name'),
                'geo': geo_obj,
                'timestamp': data.get('time', {}).get('iso'),
                'dominentpol': data.get('dominentpol')
            },
            'mapped_input': mapped,
            'current': current_status,
            'predictions': predictions
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': 'WAQI fetch failed', 'details': str(e)}), 500

@app.route('/waqi/stations', methods=['GET'])
def waqi_stations():
    try:
        token = os.environ.get('WAQI_TOKEN', '489b7744e305ef7cd2392b6434db1c0f09668d9a')
        city = request.args.get('city', '').strip()
        state = request.args.get('state', '').strip()
        if not city:
            return jsonify({'success': False, 'error': 'city is required'}), 400
        # Aggregate stations from multiple queries for better locality
        queries = []
        if state:
            queries.append(f"{city}, {state}, India")
            queries.append(f"{state}, India")
        queries.append(f"{city}, India")
        queries.append(city)
        stations = []
        seen_uid = set()
        for key in queries:
            try:
                url = f"https://api.waqi.info/search/?token={token}&keyword={urllib.parse.quote(key)}"
                with urllib.request.urlopen(url, timeout=10) as resp:
                    result = json.loads(resp.read().decode('utf-8'))
                if result.get('status') != 'ok':
                    continue
                for item in result.get('data', []):
                    uid = item.get('uid')
                    if uid in seen_uid:
                        continue
                    station = item.get('station', {}) or {}
                    country = station.get('country')
                    name = station.get('name')
                    if not name or country and country != 'IN':
                        continue
                    geo = station.get('geo') or []
                    url_path = station.get('url') or ''
                    lat = lon = None
                    if isinstance(geo, (list, tuple)) and len(geo) >= 2:
                        try:
                            lat = float(geo[0]); lon = float(geo[1])
                        except Exception:
                            lat = lon = None
                    obj = {'uid': uid, 'name': name, 'url': url_path}
                    if lat is not None and lon is not None:
                        obj['lat'] = lat; obj['lon'] = lon
                    stations.append(obj)
                    seen_uid.add(uid)
            except Exception:
                continue
        # Curated Hyderabad stations enrichment
        if city.lower() == 'hyderabad':
            curated_names = [
                "Bollaram Industrial Area",
                "Pashamylaram",
                "Sanathnagar",
                "Zoo Park",
                "Hyderabad Central University",
                "Gachibowli",
                "Punjagutta",
                "Somajiguda",
                "Paradise",
                "Secunderabad",
                "Charminar",
                "Nacharam Industrial Area",
                "Kukatpally",
                "Lingampally",
                "Miyapur",
                "Banjara Hills",
                "Jubilee Hills"
            ]
            for cname in curated_names:
                try:
                    # Try more specific first, then fall back
                    extra_queries = [
                        f"{cname}, Hyderabad, Telangana, India",
                        f"{cname}, Hyderabad, India",
                        f"{cname}, Hyderabad"
                    ]
                    chosen = None
                    for key in extra_queries:
                        q = f"https://api.waqi.info/search/?token={token}&keyword={urllib.parse.quote(key)}"
                        with urllib.request.urlopen(q, timeout=10) as resp:
                            r = json.loads(resp.read().decode('utf-8'))
                        if r.get('status') != 'ok':
                            continue
                        cl = cname.lower()
                        for it in r.get('data', []):
                            st = it.get('station', {}) or {}
                            nm = (st.get('name') or '')
                            if 'hyderabad' in nm.lower() and cl in nm.lower():
                                chosen = it
                                break
                        if chosen:
                            break
                        # As a softer match, accept hyderabad even if cname not in name
                        for it in r.get('data', []):
                            st = it.get('station', {}) or {}
                            nm = (st.get('name') or '')
                            if 'hyderabad' in nm.lower():
                                chosen = it
                                break
                        if chosen:
                            break
                    if chosen:
                        uid = chosen.get('uid')
                        if uid in seen_uid:
                            continue
                        st = chosen.get('station', {}) or {}
                        nm = st.get('name') or cname
                        geo = st.get('geo') or []
                        url_path = st.get('url') or ''
                        lat = lon = None
                        if isinstance(geo, (list, tuple)) and len(geo) >= 2:
                            try:
                                lat = float(geo[0]); lon = float(geo[1])
                            except Exception:
                                lat = lon = None
                        obj = {'uid': uid, 'name': nm, 'url': url_path}
                        if lat is not None and lon is not None:
                            obj['lat'] = lat
                            obj['lon'] = lon
                        stations.append(obj)
                        seen_uid.add(uid)
                except Exception:
                    continue
        # Tighten to city using WAQI URL path if available
        slug_city = city.strip().lower().replace(' ', '-')
        if slug_city:
            by_url = [s for s in stations if ('url' in s and f"/india/{slug_city}/" in (s.get('url') or '').lower())]
            if by_url:
                stations = by_url
            else:
                by_name = [s for s in stations if slug_city in (s.get('name','').lower())]
                if by_name:
                    stations = by_name
        # State-level bounding box filter for Telangana
        if state.strip().lower() == 'telangana':
            ts_min_lat, ts_max_lat = 15.0, 20.0
            ts_min_lon, ts_max_lon = 77.0, 81.5
            stations = [s for s in stations if (s.get('lat') is not None and s.get('lon') is not None and
                        ts_min_lat <= s['lat'] <= ts_max_lat and ts_min_lon <= s['lon'] <= ts_max_lon)]
        center_lat = None
        center_lon = None
        try:
            with urllib.request.urlopen(f"https://api.waqi.info/feed/{urllib.parse.quote(city)}/?token={token}", timeout=10) as resp:
                cfeed = json.loads(resp.read().decode('utf-8'))
            if cfeed.get('status') == 'ok':
                cgeo = (cfeed.get('data', {}) or {}).get('city', {}) or {}
                cgeo_list = cgeo.get('geo') or []
                if isinstance(cgeo_list, (list, tuple)) and len(cgeo_list) >= 2:
                    center_lat = float(cgeo_list[0])
                    center_lon = float(cgeo_list[1])
        except Exception:
            center_lat = None
            center_lon = None
        # Fallback center by averaging coordinates if WAQI feed center missing
        if center_lat is None or center_lon is None:
            if state.strip().lower() == 'telangana':
                ts_coords = {
                    'hyderabad': (17.3850, 78.4867),
                    'secunderabad': (17.4399, 78.4983),
                    'warangal': (17.9689, 79.5941),
                    'hanamkonda': (18.0078, 79.5506),
                    'nizamabad': (18.6725, 78.0941),
                    'karimnagar': (18.4386, 79.1288),
                    'khammam': (17.2473, 80.1514),
                    'sangareddy': (17.6240, 78.0867),
                    'medak': (18.0326, 78.2609),
                    'adilabad': (19.6640, 78.5320),
                    'ramagundam': (18.7597, 79.4803),
                    'suryapet': (17.1400, 79.6200),
                    'mahabubnagar': (16.7375, 78.0081),
                    'mahbubnagar': (16.7375, 78.0081),
                    'nalgonda': (17.0544, 79.2674),
                    'jagtial': (18.7903, 78.9120),
                    'kamareddy': (18.3200, 78.3400),
                    'siddipet': (18.1018, 78.8520),
                    'vikarabad': (17.3380, 77.9040),
                    'wanaparthy': (16.3610, 78.0669),
                    'nagarkurnool': (16.4833, 78.3167),
                    'bhadradri kothagudem': (17.5560, 80.6178),
                    'jangaon': (17.7280, 79.1520),
                    'yadadri bhuvanagiri': (17.3750, 78.9480)
                }
                slug = city.strip().lower()
                if slug in ts_coords:
                    center_lat, center_lon = ts_coords[slug]
            coords = [(s.get('lat'), s.get('lon')) for s in stations if s.get('lat') is not None and s.get('lon') is not None]
            if coords:
                lat_sum = sum(a for a,_ in coords)
                lon_sum = sum(b for _,b in coords)
                center_lat = lat_sum / len(coords)
                center_lon = lon_sum / len(coords)
        if center_lat is not None and center_lon is not None:
            def hav(a, b, c, d):
                import math
                r = 6371.0
                p1 = math.radians(a)
                p2 = math.radians(c)
                dp = math.radians(c - a)
                dl = math.radians(d - b)
                s = (math.sin(dp/2)**2 +
                     math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2)
                return 2 * r * math.asin(min(1.0, math.sqrt(s)))
            filtered = []
            no_geo = []
            for s in stations:
                la = s.get('lat')
                lo = s.get('lon')
                if la is None or lo is None:
                    no_geo.append(s)
                    continue
                dist = hav(center_lat, center_lon, la, lo)
                radius = 60.0 if city.lower() == 'hyderabad' else (180.0 if state.strip().lower() == 'telangana' else 100.0)
                if dist <= radius:
                    filtered.append((dist, s))
            filtered.sort(key=lambda x: x[0])
            stations = [s for _, s in filtered][:50] + no_geo[:10]
        uniq = []
        seen = set()
        for s in stations:
            k = (s['uid'], s['name'])
            if k in seen:
                continue
            seen.add(k)
            uniq.append(s)
        if not uniq and state.strip().lower() == 'telangana' and city.strip().lower() != 'hyderabad':
            try:
                extra = []
                extra_seen = set()
                hy_queries = [
                    "Hyderabad",
                    "Hyderabad, Telangana, India",
                    "Hyderabad, India"
                ]
                for key in hy_queries:
                    url = f"https://api.waqi.info/search/?token={token}&keyword={urllib.parse.quote(key)}"
                    with urllib.request.urlopen(url, timeout=10) as resp:
                        r = json.loads(resp.read().decode('utf-8'))
                    if r.get('status') != 'ok':
                        continue
                    for it in r.get('data', []):
                        uid = it.get('uid')
                        if uid in extra_seen:
                            continue
                        st = it.get('station', {}) or {}
                        nm = st.get('name') or ''
                        geo = st.get('geo') or []
                        url_path = st.get('url') or ''
                        la = lo = None
                        if isinstance(geo, (list, tuple)) and len(geo) >= 2:
                            try:
                                la = float(geo[0]); lo = float(geo[1])
                            except Exception:
                                la = lo = None
                        o = {'uid': uid, 'name': nm, 'url': url_path}
                        if la is not None and lo is not None:
                            o['lat'] = la; o['lon'] = lo
                        extra.append(o)
                        extra_seen.add(uid)
                hy_slug = 'hyderabad'
                extra = [e for e in extra if (hy_slug in (e.get('name','').lower()) or f"/india/{hy_slug}/" in (e.get('url') or '').lower())]
                hy_c = (17.3850, 78.4867)
                def hav(a,b,c,d):
                    import math
                    r=6371.0
                    p1=math.radians(a); p2=math.radians(c)
                    dp=math.radians(c-a); dl=math.radians(d-b)
                    s=(math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2)
                    return 2*r*math.asin(min(1.0, math.sqrt(s)))
                filtered=[]
                for e in extra:
                    la=e.get('lat'); lo=e.get('lon')
                    if la is None or lo is None:
                        continue
                    if hav(hy_c[0], hy_c[1], la, lo) <= 60.0:
                        filtered.append(e)
                uniq = filtered[:20]
            except Exception:
                pass
        return jsonify({'success': True, 'stations': uniq}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': 'WAQI stations fetch failed', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
