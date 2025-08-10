document.addEventListener('DOMContentLoaded', () => {
    const latInput = document.getElementById('lat');
    const lonInput = document.getElementById('lon');
    const analyzeBtn = document.getElementById('analyze-btn');
    const statusDisplay = document.getElementById('status-display');
    const loadingDiv = document.getElementById('loading');
    const resultContainer = document.getElementById('result-container');
    const statusValue = document.getElementById('status-value');
    const confidenceValue = document.getElementById('confidence-value');
    const errorMessage = document.getElementById('error-message');

    // Inicializa o mapa com Leaflet.js
    const map = L.map('map').setView([-10, -55], 5);
    
    // Adiciona o provedor de mapas (OpenStreetMap como padrão)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    let marker;

    map.on('click', function(e) {
        if (marker) {
            map.removeLayer(marker);
        }
        marker = L.marker(e.latlng).addTo(map);
        latInput.value = e.latlng.lat.toFixed(6);
        lonInput.value = e.latlng.lng.toFixed(6);
    });

    analyzeBtn.addEventListener('click', async () => {
        const lat = latInput.value;
        const lon = lonInput.value;
        
        if (!lat || !lon) {
            alert('Por favor, selecione uma área no mapa.');
            return;
        }

        // Mostra o estado de carregamento
        loadingDiv.classList.remove('hidden');
        resultContainer.classList.add('hidden');
        errorMessage.classList.add('hidden');
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ lat: lat, lon: lon })
            });

            const data = await response.json();

            if (response.ok) {
                // Atualiza a interface com os resultados
                statusValue.textContent = data.status;
                confidenceValue.textContent = `${(data.confidence * 100).toFixed(2)}%`;
                document.getElementById('last-analysis-date').textContent = new Date().toLocaleDateString();
                
                // Atualiza o gráfico
                updateChart(data.time_series);

                loadingDiv.classList.add('hidden');
                resultContainer.classList.remove('hidden');
                statusDisplay.textContent = 'Análise concluída!';
            } else {
                loadingDiv.classList.add('hidden');
                errorMessage.classList.remove('hidden');
                errorMessage.textContent = `Erro: ${data.error}`;
            }
        } catch (error) {
            loadingDiv.classList.add('hidden');
            errorMessage.classList.remove('hidden');
            errorMessage.textContent = 'Ocorreu um erro ao conectar com o servidor.';
            console.error('Error:', error);
        }
    });

    let deforestationChart;

    function updateChart(data) {
        const ctx = document.getElementById('deforestationChart').getContext('2d');
        if (deforestationChart) {
            deforestationChart.destroy();
        }

        deforestationChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [
                    {
                        label: 'Desmatamento',
                        data: data.deforestation,
                        backgroundColor: 'rgba(255, 99, 132, 0.5)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Risco',
                        data: data.risk,
                        backgroundColor: 'rgba(255, 206, 86, 0.5)',
                        borderColor: 'rgba(255, 206, 86, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Vegetação',
                        data: data.vegetation,
                        backgroundColor: 'rgba(75, 192, 192, 0.5)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
});
