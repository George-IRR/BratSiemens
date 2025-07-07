# Flask server pentru primirea datelor de detectie si servirea catre Streamlit
# Creat pentru integrarea cu robotic arm prin BLE

from flask import Flask, request, jsonify
import asyncio
import sys
import os
import threading
import copy

# Hack pentru importuri - adaug directorul curent la path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import BLE handler daca exista
try:
    from ble_handler import BLEHandler
    print("BLEHandler loaded OK")
except Exception as e:
    print(f"Nu pot incarca BLEHandler: {e}")
    BLEHandler = None

# Import data storage
try:
    from data_storage import data_store
    print("Data store loaded")
except Exception as e:
    print(f"Nu pot incarca data_store: {e}")
    data_store = None

# Config pentru server
try:
    from config import FLASK_HOST, FLASK_PORT
    print(f"Config loaded - {FLASK_HOST}:{FLASK_PORT}")
except Exception as e:
    print(f"Config nu e disponibil: {e}")
    # Fallback values
    FLASK_HOST = '0.0.0.0'
    FLASK_PORT = 5001

def strip_image_data_for_log(payload):
    # Curăță payload-ul de date mari pentru logging decent
    # Clonam obiectul ca sa nu stricam originalul
    clean_data = copy.deepcopy(payload)
    
    # Scap de imaginile mari din log
    if 'image' in clean_data:
        img_data = clean_data['image']
        if isinstance(img_data, str) and len(img_data) > 100:
            # Pastrez doar inceputul si sfarsitul pentru debug
            clean_data['image'] = f"{img_data[:50]}...[{len(img_data)} chars]...{img_data[-20:]}"
    
    # Alte campuri mari daca exista
    if 'raw_image' in clean_data:
        clean_data['raw_image'] = f"[RAW_IMG_DATA {len(str(clean_data['raw_image']))} chars]"
    
    return clean_data


class FlaskServer:
    def __init__(self):
        self.app = Flask(__name__)
        # Încerc să inițializez BLE handler-ul dacă există
        self.ble_handler = BLEHandler() if BLEHandler else None
        self._setup_routes()
        print("Flask server ready")

    def _setup_routes(self):
        # Setup pentru toate rutele Flask
        
        @self.app.route('/data', methods=['POST'])
        def receive_data():
            try:
                # Verific daca primesc JSON
                if not request.is_json:
                    return jsonify({'error': 'Trebuie sa fie JSON'}), 415

                payload = request.get_json()
                
                # Verific daca e raw image
                is_raw_image = payload.get('raw_image', False)
                
                # Curăț logging-ul - scap de datele mari
                clean_payload = strip_image_data_for_log(payload)
                
                # Log diferit pentru raw vs processed
                if is_raw_image:
                    print(f"📸 RAW image primit: {clean_payload.get('crop_shape', 'unknown size')}")
                else:
                    detection_count = len(payload.get('detections', []))
                    print(f"🎯 PROCESSED image primit cu {detection_count} detectii")
                
                # Salvez datele pentru Streamlit
                if data_store:
                    data_store.store_data(payload)
                
                # Pentru raw images, nu trimit prin BLE (nu are sens sa procesez)
                if is_raw_image:
                    print("📸 Raw image - nu trimit prin BLE")
                    status = 'raw_image_received'
                else:
                    # Trimit prin BLE doar pentru imagini procesate cu detectii
                    if self.ble_handler:
                        status = asyncio.run(self.ble_handler.send_data(payload))
                        print(f"BLE: {status}")
                    else:
                        print("BLE nu e disponibil")
                        status = 'no_ble'
                
                return jsonify({
                    'status': status, 
                    'received_count': len(payload.get('detections', [])),
                    'is_raw': is_raw_image
                }), 200
            except Exception as e:
                print(f"Eroare in receive_data: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500

        @self.app.route('/ready', methods=['GET'])
        def check_ready():
            # Verific daca bratul e gata pentru urmatoarea comanda
            try:
                if self.ble_handler:
                    ready = self.ble_handler.is_ready()
                    return jsonify({'ready': ready}), 200
                else:
                    return jsonify({'ready': False, 'error': 'BLE nu e disponibil'}), 200
            except Exception as e:
                print(f"Eroare in check_ready: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/get', methods=['GET'])
        def get_data():
            # Pentru Streamlit sa poata lua datele
            try:
                if data_store:
                    latest_data = data_store.get_latest_data()
                else:
                    latest_data = {'detections': [], 'message': 'Nu am data store'}
                return jsonify(latest_data)
            except Exception as e:
                print(f"Eroare in get_data: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/get_raw', methods=['GET'])
        def get_raw_data():
            # Pentru Streamlit sa poata lua raw images
            try:
                if data_store:
                    latest_raw_data = data_store.get_latest_raw_data()
                    if not latest_raw_data:
                        return jsonify({'message': 'Nu am raw images', 'raw_image': False})
                    return jsonify(latest_raw_data)
                else:
                    return jsonify({'error': 'Nu am data store', 'raw_image': False}), 500
            except Exception as e:
                print(f"Eroare in get_raw_data: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/status', methods=['GET'])
        def get_status():
            # Status general al serverului
            try:
                if data_store:
                    detection_count = data_store.get_detection_count()
                    timestamp = data_store.get_latest_data().get('timestamp', 0)
                else:
                    detection_count = 0
                    timestamp = 0
                
                ble_status = {}
                if self.ble_handler:
                    ble_status = self.ble_handler.get_status()
                    
                return jsonify({
                    'status': 'running',
                    'detection_count': detection_count,
                    'timestamp': timestamp,
                    'ble_available': self.ble_handler is not None,
                    'ble_status': ble_status,
                    'data_store_available': data_store is not None
                })
            except Exception as e:
                print(f"Eroare in get_status: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/clear', methods=['POST'])
        def clear_data():
            # Golesc datele stocate
            try:
                if data_store:
                    data_store.clear_data()
                    return jsonify({'status': 'cleared'})
                else:
                    return jsonify({'status': 'nu am data store'})
            except Exception as e:
                print(f"Eroare in clear_data: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/test', methods=['GET'])
        def test_endpoint():
            # Test simplu ca sa vad daca merge serverul
            return jsonify({
                'status': 'Server merge!',
                'message': 'Flask functioneaza OK',
                'ble_available': self.ble_handler is not None
            })

        print("Rutele Flask sunt gata")

    def run(self, host=None, port=None, debug=False):
        # Pornesc serverul Flask
        host = host or FLASK_HOST
        port = port or FLASK_PORT
        
        print(f"Pornesc Flask pe {host}:{port}")
        try:
            # use_reloader=False ca sa nu se incurce cu threading-ul
            self.app.run(host=host, port=port, debug=debug, use_reloader=False)
        except Exception as e:
            print(f"Eroare la pornirea serverului: {e}")
            raise


# Creez instanta de server
print("Creez instanta FlaskServer...")
flask_server = FlaskServer()
app = flask_server.app  # pentru compatibilitate cu alte script-uri
print("FlaskServer creat")
