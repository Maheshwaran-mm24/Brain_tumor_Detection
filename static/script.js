document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadConfig = document.getElementById('uploadConfig');
    
    const previewSection = document.getElementById('previewSection');
    const imagePreview = document.getElementById('imagePreview');
    const reUploadBtn = document.getElementById('reUploadBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = document.querySelector('.btn-text');
    const loadingSpinner = document.getElementById('loadingSpinner');
    
    const resultsSection = document.getElementById('resultsSection');
    const predictionValue = document.getElementById('predictionValue');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceBar = document.getElementById('confidenceBar');
    const errorMessage = document.getElementById('errorMessage');

    let currentFile = null;

    // Drag and Drop Events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    // Click to upload
    dropZone.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    reUploadBtn.addEventListener('click', () => {
        resetUI();
        fileInput.click();
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        
        // Check if image
        if (!file.type.match('image.*')) {
            showError("Please upload an image file (JPG, PNG).");
            return;
        }

        currentFile = file;
        
        // Preview
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            imagePreview.src = reader.result;
            uploadConfig.classList.add('hidden');
            previewSection.classList.remove('hidden');
            resultsSection.classList.add('hidden');
            errorMessage.classList.add('hidden');
        }
    }

    function resetUI() {
        currentFile = null;
        fileInput.value = '';
        uploadConfig.classList.remove('hidden');
        previewSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        
        // Reset results styles
        predictionValue.className = 'value';
        confidenceBar.style.width = '0%';
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorMessage.classList.remove('hidden');
        resultsSection.classList.remove('hidden');
    }

    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI Loading State
        analyzeBtn.disabled = true;
        btnText.classList.add('hidden');
        loadingSpinner.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        errorMessage.classList.add('hidden');
        
        // Prepare data
        const formData = new FormData();
        formData.append('image', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Server error occurred');
            }

            // Display Results
            displayResults(data);
            
        } catch (error) {
            showError(error.message);
        } finally {
            // Restore UI
            analyzeBtn.disabled = false;
            btnText.classList.remove('hidden');
            loadingSpinner.classList.add('hidden');
        }
    });

    function displayResults(data) {
        resultsSection.classList.remove('hidden');
        
        const prediction = data.prediction;
        const confidenceStr = data.confidence;
        const confidenceNum = parseFloat(confidenceStr);
        
        predictionValue.textContent = prediction;
        confidenceValue.textContent = confidenceStr;
        
        // Trigger reflow for animation
        confidenceBar.style.width = '0%';
        setTimeout(() => {
            confidenceBar.style.width = confidenceStr;
        }, 50);

        // Styling based on prediction
        predictionValue.className = 'value'; // Reset
        
        if (prediction === 'No Tumor') {
            predictionValue.classList.add('tumor-negative');
            confidenceBar.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
        } else {
            predictionValue.classList.add('tumor-positive');
            confidenceBar.style.background = 'linear-gradient(90deg, #ef4444, #f87171)';
        }
    }
});
