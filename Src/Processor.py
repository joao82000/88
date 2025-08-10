import numpy as np
import rasterio
from rasterio.windows import Window
import os
import warnings

# Ignora avisos de falta de CRS para arquivos GeoTIFF (o que pode ocorrer em alguns casos)
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

class SatelliteImageProcessor:
    def __init__(self, data_dir='data/processed'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_and_process_image(self, lat, lon, buffer=256, date_range=None):
        """
        Busca uma imagem de satélite para uma coordenada e a pré-processa.
        
        Este método é um esqueleto. A implementação real exigiria uma API,
        como a do Google Earth Engine ou Landsat-8 on AWS.
        """
        print(f"Buscando e processando imagem para lat={lat}, lon={lon}...")
        
        # --- Lógica de busca de imagem (simulada) ---
        # Em um projeto real, você usaria uma biblioteca como 'earthengine-api'
        # para buscar os dados. Aqui, vamos gerar uma imagem simulada.
        
        try:
            # Dimensões da imagem simulada (por exemplo, 64x64 pixels com 3 bandas)
            image_size = 64
            # Gere uma imagem simulada com 3 bandas (RGB)
            simulated_image = np.random.rand(image_size, image_size, 3).astype(np.float32)
            
            # Normalização (como no seu código original)
            image_normalized = self.normalize_image(simulated_image)
            
            return image_normalized
        except Exception as e:
            print(f"Erro na busca/processamento da imagem simulada: {e}")
            return None

    def normalize_image(self, image_array):
        """Normaliza os valores de pixel da imagem para o intervalo [0, 1]."""
        # Certifica-se de que a imagem é float32 para o TensorFlow
        normalized_image = image_array.astype(np.float32) / np.max(image_array)
        return normalized_image
