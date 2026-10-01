document.addEventListener('DOMContentLoaded', () => {
    
    const form = document.getElementById('predictionForm');
    const predictBtn = document.getElementById('predictBtn');
    const clearBtn = document.getElementById('clearBtn');
    
    const resultCard = document.getElementById('resultCard');
    const loadingState = document.getElementById('loadingState');
    const predictionResult = document.getElementById('predictionResult');
    const resultValue = document.getElementById('resultValue');
    const resultConfidence = document.getElementById('resultConfidence');

    // Load sample data parsing
    let sampleData = {};
    const sampleDataEl = document.getElementById('sampleData');
    if (sampleDataEl) {
        try {
            sampleData = JSON.parse(sampleDataEl.textContent);
        } catch(e) {
            console.error("Could not parse sample data", e);
        }
    }

    const loadMalignantBtn = document.getElementById('loadMalignantBtn');
    const loadBenignBtn = document.getElementById('loadBenignBtn');

    function fillForm(dataObj) {
        if (!dataObj) return;
        for (const [key, value] of Object.entries(dataObj)) {
            const input = form.querySelector(`[name="${key}"]`);
            if (input) {
                // Formatting to 4 decimal places for cleanliness, or just raw value
                input.value = parseFloat(value).toFixed(4);
            }
        }
    }

    if (loadMalignantBtn) {
        loadMalignantBtn.addEventListener('click', () => fillForm(sampleData.malignant));
    }
    
    if (loadBenignBtn) {
        loadBenignBtn.addEventListener('click', () => fillForm(sampleData.benign));
    }

    // Smooth scrolling for navigation links
    document.querySelectorAll('a.nav-link, a.btn').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if(href && href.startsWith('#')) {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });

    // Clear Form handler
    clearBtn.addEventListener('click', () => {
        form.reset();
        resultCard.classList.add('d-none');
        window.scrollTo({
            top: form.offsetTop - 100,
            behavior: 'smooth'
        });
    });

    // Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Validation check (HTML5 does most of this, but just in case)
        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }

        // Prepare UI for loading
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Analyzing...';
        
        resultCard.classList.remove('d-none');
        loadingState.classList.remove('d-none');
        predictionResult.classList.add('d-none');

        // Scroll to result slightly
        setTimeout(() => {
            resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 100);

        // Gather data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error('Server responded with an error.');
            }

            const result = await response.json();
            
            // Artificial delay to show loading state (enhances user experience for local fast models)
            setTimeout(() => {
                displayResult(result);
            }, 800);
            
        } catch (error) {
            console.error('Prediction error:', error);
            alert('An error occurred during prediction. Please try again.');
            resetButton();
            resultCard.classList.add('d-none');
        }
    });

    function displayResult(data) {
        // Hide loading, show result
        loadingState.classList.add('d-none');
        predictionResult.classList.remove('d-none');

        const pred = data.prediction;
        const conf = data.confidence;

        resultValue.textContent = pred;
        resultConfidence.textContent = `Confidence: ${conf}%`;

        // Styling based on result
        if (pred.toLowerCase() === 'benign') {
            resultValue.className = 'display-5 fw-bold mb-3 text-success';
            resultConfidence.className = 'mb-4 badge rounded-pill px-3 py-2 fs-6 bg-success text-white';
        } else {
            resultValue.className = 'display-5 fw-bold mb-3 text-danger';
            resultConfidence.className = 'mb-4 badge rounded-pill px-3 py-2 fs-6 bg-danger text-white';
        }

        resetButton();
    }

    function resetButton() {
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i class="fa-solid fa-magnifying-glass me-2"></i> Predict Result';
    }
});
