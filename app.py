import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Importa as classes do projeto
from src.model import DeforestationModel
from src.processor import SatelliteImageProcessor

app = Flask(__name__)

# Configurações do modelo
MODEL_PATH = 'models/cnn_deforestation.h5'
image_processor = SatelliteImageProcessor()
deforestation_model = DeforestationModel(model_type='cnn')

# Carregar o modelo treinado
try:
    deforestation_model.load_model(MODEL_PATH)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}. Certifique-se de que o modelo foi treinado e salvo em '{MODEL_PATH}'.")

@app.route('/')
def index():
    """Rota principal que renderiza a página com o mapa."""
    # Obter a chave da API do Mapbox das variáveis de ambiente
    mapbox_access_token = os.getenv('MAPBOX_ACCESS_TOKEN', 'YOUR_MAPBOX_TOKEN')
    return render_template('index.html', mapbox_access_token=mapbox_access_token)

@app.route('/predict', methods=['POST'])
def predict():
    """Rota da API que faz a previsão de desmatamento para uma coordenada."""
    data = request.json
    lat = float(data['lat'])
    lon = float(data['lon'])

    try:
        # Busca e processa a imagem de satélite
        print(f"Buscando imagem para as coordenadas: {lat}, {lon}")
        satellite_image = image_processor.fetch_and_process_image(lat, lon)
        
        if satellite_image is None:
            return jsonify({'error': 'Não foi possível obter a imagem para as coordenadas fornecidas.'}), 400

        # Faz a previsão usando o modelo de IA
        prediction_result = deforestation_model.predict(satellite_image)
        
        # Gera dados de série temporal simulados para a visualização
        time_series_data = generate_time_series_data()

        return jsonify({
            'status': prediction_result['status'],
            'confidence': prediction_result['confidence'],
            'coordinates': [lat, lon],
            'time_series': time_series_data
        })
    except Exception as e:
        print(f"Erro na previsão: {e}")
        return jsonify({'error': str(e)}), 500

def generate_time_series_data():
    """Gera dados temporais simulados para o gráfico de histórico."""
    months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    return {
        'labels': months,
        'deforestation': [np.random.randint(0, 100) for _ in range(12)],
        'risk': [np.random.randint(0, 100) for _ in range(12)],
        'vegetation': [np.random.randint(0, 100) for _ in range(12)]
    }

if __name__ == '__main__':
    # Usar gunicorn para produção é a prática recomendada
    # Para desenvolvimento, o servidor Flask padrão é suficiente
    app.run(debug=True)
