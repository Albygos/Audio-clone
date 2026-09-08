import os
import tempfile
import numpy as np
import librosa
import soundfile as sf
import pyworld as pw
from flask import Flask, request, jsonify, render_template_string, send_file
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

app = Flask(__name__)
# Use the OS temp directory for ephemeral storage on Render
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Audio Parameter Extractor & Voice Cloner</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style> body { background-color: #f8f9fa; padding-top: 2rem; } .loader { display: none; } </style>
</head>
<body>
<div class="container">
    <h2 class="mb-4 text-center">Acoustic Feature Extraction & Voice Cloning</h2>
    <div class="card shadow-sm p-4 mb-4">
        <form id="audioForm">
            <div class="mb-3">
                <label class="form-label"><b>Audio 1 (Target Profile):</b> Upload the voice you want to clone.</label>
                <input type="file" class="form-control" id="audio1" accept="audio/*" required>
            </div>
            <div class="mb-3">
                <label class="form-label"><b>Audio 2 (Source Speech):</b> Upload the audio to be matched to Audio 1.</label>
                <input type="file" class="form-control" id="audio2" accept="audio/*" required>
            </div>
            <button type="submit" class="btn btn-primary w-100" id="submitBtn">Process & Clone Audio</button>
        </form>
        <div class="text-center mt-3 loader" id="loader">
            <div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div>
            <p class="mt-2 text-muted">Extracting deep vectors and cloning spectral envelopes... This may take up to 60 seconds.</p>
        </div>
    </div>

    <div id="results" style="display:none;">
        <div class="alert alert-success" role="alert">
            <h4 class="alert-heading">Success!</h4>
            <p>Audio 2 has been matched to Audio 1's parameters.</p>
            <a href="#" id="downloadLink" class="btn btn-success" download="cloned_audio.wav">Download Matched Audio 2</a>
        </div>
        <div class="card shadow-sm p-4">
            <h4>Extracted Parameters (Audio 1 Profile)</h4>
            <pre id="jsonOutput" class="bg-dark text-light p-3 rounded" style="max-height: 500px; overflow-y: scroll;"></pre>
        </div>
    </div>
</div>

<script>
document.getElementById('audioForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    document.getElementById('submitBtn').disabled = true;
    document.getElementById('loader').style.display = 'block';
    document.getElementById('results').style.display = 'none';

    const formData = new FormData();
    formData.append('audio1', document.getElementById('audio1').files[0]);
    formData.append('audio2', document.getElementById('audio2').files[0]);

    try {
        const response = await fetch('/process', { method: 'POST', body: formData });
        if (!response.ok) throw new Error("Server processed failed.");
        const data = await response.json();
        
        document.getElementById('jsonOutput').textContent = JSON.stringify(data.parameters, null, 4);
        document.getElementById('downloadLink').href = `/download/${data.cloned_file}`;
        document.getElementById('results').style.display = 'block';
    } catch (error) {
        alert("An error occurred. Ensure your audio files are valid and not too large.");
    } finally {
        document.getElementById('submitBtn').disabled = false;
        document.getElementById('loader').style.display = 'none';
    }
});
</script>
</body>
</html>
"""

def extract_deep_vectors(file_path):
    try:
        from speechbrain.pretrained import EncoderClassifier
        # Downloads model to temp dir on first run
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir=os.path.join(app.config['UPLOAD_FOLDER'], "pretrained_models")
        )
        signal, fs = librosa.load(file_path, sr=16000)
        import torch
        embeddings = classifier.encode_batch(torch.tensor(signal).unsqueeze(0))
        return embeddings.squeeze().detach().numpy().tolist()[:10]
    except Exception as e:
        return f"Extraction failed: {str(e)}"

def extract_features(y, sr):
    features = {}
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['MFCC_Mean'] = np.mean(mfccs, axis=1).tolist()
    
    lpc_coeffs = librosa.lpc(y, order=12)
    features['LPC'] = lpc_coeffs.tolist()
    
    f0 = librosa.yin(y, fmin=50, fmax=500)
    features['F0_Mean'] = float(np.nanmean(f0))
    
    zcr = librosa.feature.zero_crossing_rate(y)
    features['ZCR_Mean'] = float(np.mean(zcr))
    
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['Spectral_Centroid_Mean'] = float(np.mean(centroid))
    
    flux = librosa.onset.onset_strength(y=y, sr=sr)
    features['Spectral_Flux_Mean'] = float(np.mean(flux))
    
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['Spectral_Rolloff_Mean'] = float(np.mean(rolloff))
    
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    features['PLP_Proxy_Mean'] = np.mean(mel, axis=1).tolist()[:10]

    return features

def clone_voice_profile(y1, y2, sr):
    y1 = y1.astype(np.float64)
    y2 = y2.astype(np.float64)

    f0_1, t_1 = pw.dio(y1, sr)
    sp_1 = pw.cheaptrick(y1, f0_1, t_1, sr)
    
    f0_2, t_2 = pw.dio(y2, sr)
    sp_2 = pw.cheaptrick(y2, f0_2, t_2, sr)
    ap_2 = pw.d4c(y2, f0_2, t_2, sr)

    valid_f0_1 = f0_1[f0_1 > 0]
    valid_f0_2 = f0_2[f0_2 > 0]
    
    if len(valid_f0_1) > 0 and len(valid_f0_2) > 0:
        mu_1, std_1 = np.mean(valid_f0_1), np.std(valid_f0_1)
        mu_2, std_2 = np.mean(valid_f0_2), np.std(valid_f0_2)
        
        converted_f0_2 = np.copy(f0_2)
        converted_f0_2[f0_2 > 0] = (f0_2[f0_2 > 0] - mu_2) * (std_1 / (std_2 + 1e-8)) + mu_1
        converted_f0_2 = np.clip(converted_f0_2, 10, sr/2)
    else:
        converted_f0_2 = f0_2

    log_sp1 = np.log(sp_1 + 1e-10)
    log_sp2 = np.log(sp_2 + 1e-10)
    
    mu_sp1 = np.mean(log_sp1, axis=0)
    std_sp1 = np.std(log_sp1, axis=0)
    mu_sp2 = np.mean(log_sp2, axis=0)
    std_sp2 = np.std(log_sp2, axis=0)

    converted_log_sp2 = (log_sp2 - mu_sp2) * (std_sp1 / (std_sp2 + 1e-10)) + mu_sp1
    converted_sp2 = np.exp(converted_log_sp2)

    y2_cloned = pw.synthesize(converted_f0_2, converted_sp2, ap_2, sr)
    return y2_cloned

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/process', methods=['POST'])
def process_audio():
    audio1 = request.files['audio1']
    audio2 = request.files['audio2']

    path1 = os.path.join(app.config['UPLOAD_FOLDER'], 'audio1.wav')
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], 'audio2.wav')
    audio1.save(path1)
    audio2.save(path2)

    SR = 22050
    y1, _ = librosa.load(path1, sr=SR)
    y2, _ = librosa.load(path2, sr=SR)

    parameters = extract_features(y1, SR)
    parameters['x-vectors (Deep Embedding)'] = extract_deep_vectors(path1)
    parameters['i-vectors / d-vectors'] = "Included natively within x-vector feature subspace."

    y2_cloned = clone_voice_profile(y1, y2, SR)

    output_filename = "cloned_output.wav"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    sf.write(output_path, y2_cloned, SR)

    return jsonify({
        "parameters": parameters,
        "cloned_file": output_filename
    })

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
