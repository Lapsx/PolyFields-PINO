const canvas = document.getElementById('inputCanvas');
const ctx = canvas.getContext('2d');
const chargeSlider = document.getElementById('chargeSlider');
const chargeValueLabel = document.getElementById('chargeValue');
const radiusSlider = document.getElementById('radiusSlider');
const clearBtn = document.getElementById('clearBtn');
const outputImage = document.getElementById('outputImage');
const loading = document.getElementById('loading');
const latencyLabel = document.getElementById('latency');

const kuhnSlider = document.getElementById('kuhnSlider');
const kuhnValue = document.getElementById('kuhnValue');
const debyeSlider = document.getElementById('debyeSlider');
const debyeValue = document.getElementById('debyeValue');
const florySlider = document.getElementById('florySlider');
const floryValue = document.getElementById('floryValue');

const lossImage = document.getElementById('lossImage');
const lossLoading = document.getElementById('lossLoading');

// Novos Elementos UI
const massValue = document.getElementById('massValue');
const rgValue = document.getElementById('rgValue');
const comValue = document.getElementById('comValue');
const phaseThermometer = document.getElementById('phaseThermometer');
const phaseValue = document.getElementById('phaseValue');

const saveStateBtn = document.getElementById('saveStateBtn');
const compareBtn = document.getElementById('compareBtn');
const diffContainer = document.getElementById('diffContainer');
const diffImage = document.getElementById('diffImage');
const diffLoading = document.getElementById('diffLoading');

const phaseDiagramBtn = document.getElementById('phaseDiagramBtn');
const phaseDiagramContainer = document.getElementById('phaseDiagramContainer');
const phaseDiagramImage = document.getElementById('phaseDiagramImage');
const phaseDiagramLoading = document.getElementById('phaseDiagramLoading');

let charges = [];
const N = 100; // Resolução do modelo FNO

let savedStateA = null;
let currentPayload = null;

// Atualiza labels dos parâmetros físicos
kuhnSlider.addEventListener('input', (e) => { kuhnValue.textContent = e.target.value; requestPrediction(); });
debyeSlider.addEventListener('input', (e) => { debyeValue.textContent = e.target.value; requestPrediction(); });
florySlider.addEventListener('input', (e) => { floryValue.textContent = e.target.value; requestPrediction(); });

// Atualiza o label do slider de carga
chargeSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    if(val > 0) {
        chargeValueLabel.textContent = `Repulsivo (+${val})`;
        chargeValueLabel.style.color = '#fca5a5';
    } else if (val < 0) {
        chargeValueLabel.textContent = `Atrativo (${val})`;
        chargeValueLabel.style.color = '#38bdf8';
    } else {
        chargeValueLabel.textContent = `Neutro (0)`;
        chargeValueLabel.style.color = '#cbd5e1';
    }
});

// Desenha a esfera, o grid e as partículas
function drawCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Fundo
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Variáveis Físicas
    const L = 8.0;
    const a = 1.0;
    const pxPerUnit = canvas.width / L; // 400 / 8 = 50 pixels por unidade

    // Desenhar Grid Físico
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    for(let i = -4; i <= 4; i++) {
        const px = (i + L/2) * pxPerUnit; // Normaliza de -4..4 para 0..8
        
        // Linhas de Grade
        ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, canvas.height); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, px); ctx.lineTo(canvas.width, px); ctx.stroke();
        
        // Textos dos eixos
        if(i !== 0) {
            ctx.fillText(i.toString(), px, canvas.height/2 + 10); // Eixo X
            ctx.fillText((-i).toString(), canvas.width/2 - 10, px); // Eixo Z (invertido visualmente para y)
        }
    }

    // Interior Sólido (Nanopartícula)
    ctx.beginPath();
    ctx.arc(canvas.width/2, canvas.height/2, a * pxPerUnit, 0, 2*Math.PI);
    ctx.fillStyle = 'rgba(30, 41, 59, 0.8)'; // Cor escura para o sólido
    ctx.fill();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Desenha as cargas
    charges.forEach(c => {
        const px = (c.x / N) * canvas.width;
        const py = (c.z / N) * canvas.height;
        
        ctx.beginPath();
        ctx.arc(px, py, c.r * 2, 0, 2*Math.PI);
        
        if(c.q > 0) {
            ctx.fillStyle = `rgba(239, 68, 68, ${Math.min(1.0, c.q/5)})`; // Vermelho (Repulsivo)
            ctx.shadowColor = '#ef4444';
        } else {
            ctx.fillStyle = `rgba(56, 189, 248, ${Math.min(1.0, Math.abs(c.q)/5)})`; // Azul (Atrativo)
            ctx.shadowColor = '#38bdf8';
        }
        ctx.shadowBlur = 15;
        ctx.fill();
        ctx.shadowBlur = 0; // reset
    });
}

// Limpa tudo
clearBtn.addEventListener('click', () => {
    charges = [];
    kuhnSlider.value = 1.0;
    kuhnValue.textContent = '1.0';
    debyeSlider.value = 1.0;
    debyeValue.textContent = '1.0';
    florySlider.value = 0.0;
    floryValue.textContent = '0.0';
    chargeSlider.value = 5;
    chargeValueLabel.textContent = `Repulsivo (+5)`;
    chargeValueLabel.style.color = '#fca5a5';
    radiusSlider.value = 5;
    
    savedStateA = null;
    compareBtn.disabled = true;
    diffContainer.style.display = 'none';
    phaseDiagramContainer.style.display = 'none';

    drawCanvas();
    outputImage.style.display = 'none';
    document.getElementById('d1Image').style.display = 'none';
    document.getElementById('d2Image').style.display = 'none';
    requestPrediction();
});

// Clique no Canvas
canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    let grid_x = Math.floor((px / rect.width) * N);
    let grid_z = Math.floor((py / rect.height) * N);
    


    const q = parseFloat(chargeSlider.value);
    const r = parseFloat(radiusSlider.value);

    charges.push({x: grid_x, z: grid_z, q: q, r: r});
    drawCanvas();
    requestPrediction();
});

// Botão direito para remover carga
canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault(); // Impede o menu do navegador
    if (charges.length === 0) return;

    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    const grid_x = Math.floor((px / rect.width) * N);
    const grid_z = Math.floor((py / rect.height) * N);

    let closestIdx = -1;
    let minDist = Infinity;

    for (let i = 0; i < charges.length; i++) {
        const c = charges[i];
        const dist = Math.sqrt(Math.pow(c.x - grid_x, 2) + Math.pow(c.z - grid_z, 2));
        if (dist < minDist) {
            minDist = dist;
            closestIdx = i;
        }
    }

    // Tolerância de ~10 pixels/grid para o clique
    if (closestIdx !== -1 && minDist < 10) {
        charges.splice(closestIdx, 1);
        drawCanvas();
        requestPrediction();
    }
});

// Salvar Estado e Comparar
saveStateBtn.addEventListener('click', () => {
    if (currentPayload) {
        savedStateA = JSON.parse(JSON.stringify(currentPayload));
        compareBtn.disabled = false;
        saveStateBtn.textContent = "Estado A Salvo! ✓";
        setTimeout(() => saveStateBtn.textContent = "Salvar Estado Atual (A)", 2000);
    }
});

compareBtn.addEventListener('click', async () => {
    if (!savedStateA || !currentPayload) return;
    
    diffContainer.style.display = 'flex';
    diffImage.style.display = 'none';
    diffLoading.style.display = 'block';
    
    try {
        const payload = {
            stateA: savedStateA,
            stateB: currentPayload
        };
        const response = await fetch('http://localhost:8000/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if(response.ok) {
            const data = await response.json();
            diffImage.src = "data:image/png;base64," + data.image;
            diffImage.style.display = 'block';
            diffLoading.style.display = 'none';
        }
    } catch (err) {
        console.error("Erro na Comparação:", err);
        diffLoading.style.display = 'none';
    }
});

// Diagrama de Fases e Experimentos
const experimentTypeSelect = document.getElementById('experimentType');
const multipoleSweepBtn = document.getElementById('multipoleSweepBtn');

multipoleSweepBtn.addEventListener('click', async () => {
    if (!currentPayload) return;
    
    phaseDiagramContainer.style.display = 'flex';
    phaseDiagramImage.style.display = 'none';
    phaseDiagramLoading.style.display = 'block';
    
    // Pega a magnitude atual do slider de carga para usar nas cargas multipolares simuladas
    const q_mag = Math.abs(parseFloat(chargeSlider.value));
    
    const payload = {
        q_magnitude: q_mag,
        b: parseFloat(kuhnSlider.value),
        kappa: parseFloat(debyeSlider.value),
        u: parseFloat(florySlider.value),
        polymer_charge: document.getElementById('polymerChargeType') ? parseInt(document.getElementById('polymerChargeType').value) : 0,
        polymer_charge_intensity: document.getElementById('polymerChargeIntensitySlider') ? parseFloat(document.getElementById('polymerChargeIntensitySlider').value) : 1.0,
        sweep_type: experimentTypeSelect ? experimentTypeSelect.value : 'u'
    };
    
    try {
        const response = await fetch('http://localhost:8000/experiment_multipole', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if(response.ok) {
            const data = await response.json();
            phaseDiagramImage.src = "data:image/png;base64," + data.image;
            phaseDiagramImage.style.display = 'block';
            phaseDiagramLoading.style.display = 'none';
        } else {
            console.error("Erro do servidor:", response.status);
            phaseDiagramLoading.style.display = 'none';
        }
    } catch (err) {
        console.error("Erro ao gerar diagrama multipolar:", err);
        phaseDiagramLoading.style.display = 'none';
    }
});

phaseDiagramBtn.addEventListener('click', async () => {
    if (!currentPayload) return;
    
    phaseDiagramContainer.style.display = 'flex';
    phaseDiagramImage.style.display = 'none';
    phaseDiagramLoading.style.display = 'block';
    
    const experimentPayload = {
        ...currentPayload,
        sweep_type: experimentTypeSelect ? experimentTypeSelect.value : 'u'
    };
    
    try {
        const response = await fetch('http://localhost:8000/experiment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(experimentPayload)
        });

        if(response.ok) {
            const data = await response.json();
            phaseDiagramImage.src = "data:image/png;base64," + data.image;
            phaseDiagramImage.style.display = 'block';
            phaseDiagramLoading.style.display = 'none';
        } else {
            console.error("Erro do servidor:", response.status);
            phaseDiagramLoading.style.display = 'none';
        }
    } catch (err) {
        console.error("Erro ao gerar diagrama de fases:", err);
        phaseDiagramLoading.style.display = 'none';
    }
});

// Comunicação com o Servidor Python (FastAPI)
async function requestPrediction() {
    const startTime = performance.now();
    
    if(charges.length > 0) {
        outputImage.style.display = 'none';
        loading.style.display = 'block';
        document.getElementById('d1Image').style.display = 'none';
        document.getElementById('d1Loading').style.display = 'block';
        document.getElementById('d2Image').style.display = 'none';
        document.getElementById('d2Loading').style.display = 'block';
    }

    currentPayload = {
        charges: charges,
        b: parseFloat(kuhnSlider.value),
        kappa: parseFloat(debyeSlider.value),
        u: parseFloat(florySlider.value),
        polymer_charge: document.getElementById('polymerChargeType') ? parseInt(document.getElementById('polymerChargeType').value) : 0,
        polymer_charge_intensity: document.getElementById('polymerChargeIntensitySlider') ? parseFloat(document.getElementById('polymerChargeIntensitySlider').value) : 1.0
    };

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentPayload)
        });

        if(response.ok) {
            const data = await response.json();
            outputImage.src = "data:image/png;base64," + data.image;
            outputImage.style.display = 'block';
            loading.style.display = 'none';
            
            if(data.image_d1) {
                document.getElementById('d1Image').src = "data:image/png;base64," + data.image_d1;
                document.getElementById('d1Image').style.display = 'block';
                document.getElementById('d1Loading').style.display = 'none';
            }
            if(data.image_d2) {
                document.getElementById('d2Image').src = "data:image/png;base64," + data.image_d2;
                document.getElementById('d2Image').style.display = 'block';
                document.getElementById('d2Loading').style.display = 'none';
            }
            
            // Atualizar Métricas
            if (data.metrics) {
                massValue.textContent = data.metrics.mass.toFixed(2);
                rgValue.textContent = data.metrics.rg.toFixed(3);
                comValue.textContent = `(${data.metrics.com_x.toFixed(2)}, ${data.metrics.com_z.toFixed(2)})`;
                
                phaseValue.textContent = data.metrics.phase;
                
                // Atualizar Cor do Termômetro
                phaseThermometer.className = "alert-box"; // reset
                if (data.metrics.phase.includes('Colapsado')) {
                    phaseThermometer.classList.add('alert-globule');
                } else if (data.metrics.phase.includes('Inchado')) {
                    phaseThermometer.classList.add('alert-coil');
                }
            }
            
            const endTime = performance.now();
            latencyLabel.textContent = `Latência FNO: ${Math.round(endTime - startTime)} ms`;
        }
    } catch (err) {
        console.error("Erro ao chamar a FNO:", err);
        latencyLabel.textContent = "Servidor Offline";
        loading.style.display = 'none';
    }
}

async function fetchLoss() {
    try {
        const response = await fetch('http://localhost:8000/loss');
        if(response.ok) {
            const data = await response.json();
            if(data.image) {
                lossImage.src = "data:image/png;base64," + data.image;
                lossImage.style.display = 'block';
                lossLoading.style.display = 'none';
            } else {
                lossLoading.style.display = 'none';
                lossImage.alt = "Histórico não disponível ainda";
            }
        }
    } catch (err) {
        console.log("Loss endpoint not available yet");
    }
}

// Init
drawCanvas();
requestPrediction();
fetchLoss();
setInterval(fetchLoss, 10000); // Atualiza o gráfico de loss a cada 10 segundos

// Lógica de Troca de Abas
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        const targetId = btn.getAttribute('data-tab');
        document.getElementById(targetId).classList.add('active');
    });
});

const polymerChargeSelect = document.getElementById('polymerChargeType');
if (polymerChargeSelect) {
    polymerChargeSelect.addEventListener('change', requestPrediction);
}

const polymerChargeIntensitySlider = document.getElementById('polymerChargeIntensitySlider');
const polymerChargeIntensityValue = document.getElementById('polymerChargeIntensityValue');

if (polymerChargeIntensitySlider) {
    polymerChargeIntensitySlider.addEventListener('input', (e) => {
        polymerChargeIntensityValue.textContent = e.target.value;
        requestPrediction();
    });
}

// High-Throughput Screening (HTS) Logic
const htsFileInput = document.getElementById('htsFileInput');
const htsUploadBtn = document.getElementById('htsUploadBtn');
const htsStatus = document.getElementById('htsStatus');

if (htsUploadBtn && htsFileInput) {
    htsUploadBtn.addEventListener('click', () => htsFileInput.click());

    htsFileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        htsStatus.textContent = "Lendo arquivo JSON...";
        htsStatus.style.color = "#fbbf24";

        try {
            const text = await file.text();
            const payload = JSON.parse(text);

            if (!payload.items || !Array.isArray(payload.items)) {
                throw new Error("Formato inválido. O JSON deve ter um array 'items'.");
            }

            htsStatus.textContent = `Enviando batch de ${payload.items.length} moléculas para a PINO...`;

            const response = await fetch('http://localhost:8000/batch_screen', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: text
            });

            if (!response.ok) throw new Error("Erro no servidor da PINO.");

            const data = await response.json();
            
            htsStatus.textContent = "Processamento concluído! Gerando CSV...";
            htsStatus.style.color = "#10b981";

            // Convert to CSV
            const results = data.results;
            if (results.length === 0) {
                htsStatus.textContent = "Batch retornou vazio.";
                return;
            }

            const headers = Object.keys(results[0]).join(",");
            const rows = results.map(r => Object.values(r).join(","));
            const csvContent = headers + "\n" + rows.join("\n");

            const blob = new Blob([csvContent], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.setAttribute('hidden', '');
            a.setAttribute('href', url);
            a.setAttribute('download', 'hts_results.csv');
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);

            htsStatus.textContent = "Download do CSV concluído! HTS Sucesso.";

        } catch (err) {
            htsStatus.textContent = "Erro: " + err.message;
            htsStatus.style.color = "#ef4444";
        }
    });
}
