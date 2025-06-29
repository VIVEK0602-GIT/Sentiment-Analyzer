// Sentiment analysis web interface
document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const textInput = document.getElementById('text-input');
    const modelSelect = document.getElementById('model-select');
    const analyzeBtn = document.getElementById('analyze-btn');
    const compareBtn = document.getElementById('compare-btn');
    const resultsSection = document.getElementById('results-section');
    const comparisonSection = document.getElementById('comparison-section');
    const sentimentIcon = document.getElementById('sentiment-icon');
    const sentimentText = document.getElementById('sentiment-text');
    const modelUsed = document.getElementById('model-used');
    const originalText = document.getElementById('original-text');
    const processedText = document.getElementById('processed-text');
    const modelResultsList = document.getElementById('model-results-list');

    // Chart objects
    let sentimentScoresChart = null;
    let comparisonChart = null;

    // Event listeners
    analyzeBtn.addEventListener('click', analyzeSentiment);
    compareBtn.addEventListener('click', compareModels);

    // Functions
    async function analyzeSentiment() {
        const text = textInput.value.trim();
        if (!text) {
            alert('Please enter some text to analyze');
            return;
        }

        // Show loading state
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner loading"></i> Analyzing...';
        comparisonSection.style.display = 'none';

        try {
            const model = modelSelect.value;
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text, model })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            displayResults(data, model);
        } catch (error) {
            console.error('Error analyzing sentiment:', error);
            alert('An error occurred while analyzing the sentiment. Please try again.');
        } finally {
            // Reset button
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze';
        }
    }

    async function compareModels() {
        const text = textInput.value.trim();
        if (!text) {
            alert('Please enter some text to analyze');
            return;
        }

        // Show loading state
        compareBtn.disabled = true;
        compareBtn.innerHTML = '<i class="fas fa-spinner loading"></i> Comparing...';
        resultsSection.style.display = 'none';

        try {
            const response = await fetch('/compare', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            displayComparison(data);
        } catch (error) {
            console.error('Error comparing models:', error);
            alert('An error occurred while comparing models. Please try again.');
        } finally {
            // Reset button
            compareBtn.disabled = false;
            compareBtn.innerHTML = '<i class="fas fa-balance-scale"></i> Compare All Models';
        }
    }

    function displayResults(data, modelType) {
        // Show results section
        resultsSection.style.display = 'block';

        // Set original and processed text
        originalText.textContent = data.text;
        processedText.textContent = data.processed_text;

        // Get sentiment and model result
        const sentiment = data.results[modelType];
        
        // Set sentiment icon and text
        if (sentiment === 'positive') {
            sentimentIcon.innerHTML = '<i class="fas fa-smile-beam sentiment-icon-positive"></i>';
            sentimentText.textContent = 'Positive';
            sentimentText.className = 'mb-3 sentiment-positive';
        } else if (sentiment === 'negative') {
            sentimentIcon.innerHTML = '<i class="fas fa-frown sentiment-icon-negative"></i>';
            sentimentText.textContent = 'Negative';
            sentimentText.className = 'mb-3 sentiment-negative';
        } else {
            sentimentIcon.innerHTML = '<i class="fas fa-meh sentiment-icon-neutral"></i>';
            sentimentText.textContent = 'Neutral';
            sentimentText.className = 'mb-3 sentiment-neutral';
        }

        // Set model used text
        let modelName;
        let modelBadgeClass;
        
        if (modelType === 'vader') {
            modelName = 'VADER (Rule-based)';
            modelBadgeClass = 'model-badge-vader';
        } else if (modelType === 'logistic_regression') {
            modelName = 'Logistic Regression';
            modelBadgeClass = 'model-badge-traditional';
        } else if (modelType === 'svm') {
            modelName = 'Support Vector Machine';
            modelBadgeClass = 'model-badge-traditional';
        } else if (modelType === 'lstm' || modelType === 'cnn' || modelType === 'bilstm') {
            modelName = modelType.toUpperCase() + ' Neural Network';
            modelBadgeClass = 'model-badge-neural';
        }

        modelUsed.innerHTML = `Model: <span class="model-badge ${modelBadgeClass}">${modelName}</span>`;

        // Show sentiment scores (always use VADER scores)
        const vaderScores = data.detailed_results.vader;
        updateSentimentScoresChart(vaderScores);
    }

    function displayComparison(data) {
        // Show comparison section
        comparisonSection.style.display = 'block';

        // Clear previous model results
        modelResultsList.innerHTML = '';

        // Prepare data for the chart
        const modelLabels = [];
        const sentiments = [];
        const colors = [];

        // Add models to the list and collect data for chart
        for (const [modelName, sentiment] of Object.entries(data.results)) {
            let modelDisplayName;
            let modelBadgeClass;
            let iconClass;
            let colorClass;

            // Set display name and badge class
            if (modelName === 'vader') {
                modelDisplayName = 'VADER';
                modelBadgeClass = 'model-badge-vader';
            } else if (modelName === 'logistic_regression') {
                modelDisplayName = 'Logistic Regression';
                modelBadgeClass = 'model-badge-traditional';
            } else if (modelName === 'svm') {
                modelDisplayName = 'SVM';
                modelBadgeClass = 'model-badge-traditional';
            } else {
                modelDisplayName = modelName.toUpperCase();
                modelBadgeClass = 'model-badge-neural';
            }

            // Set icon and color based on sentiment
            if (sentiment === 'positive') {
                iconClass = 'fa-smile-beam sentiment-icon-positive';
                colorClass = 'sentiment-positive';
            } else if (sentiment === 'negative') {
                iconClass = 'fa-frown sentiment-icon-negative';
                colorClass = 'sentiment-negative';
            } else {
                iconClass = 'fa-meh sentiment-icon-neutral';
                colorClass = 'sentiment-neutral';
            }

            // Add to list
            const li = document.createElement('li');
            li.className = 'list-group-item d-flex justify-content-between align-items-center';
            li.innerHTML = `
                <div>
                    <span class="model-badge ${modelBadgeClass}">${modelDisplayName}</span>
                </div>
                <div class="${colorClass}">
                    ${sentiment.charAt(0).toUpperCase() + sentiment.slice(1)}
                    <i class="fas ${iconClass} ms-2"></i>
                </div>
            `;
            modelResultsList.appendChild(li);

            // Collect data for chart
            modelLabels.push(modelDisplayName);
            sentiments.push(sentiment);
            
            // Set color based on sentiment
            if (sentiment === 'positive') {
                colors.push('#28a745');
            } else if (sentiment === 'negative') {
                colors.push('#dc3545');
            } else {
                colors.push('#6c757d');
            }
        }

        // Update comparison chart
        updateComparisonChart(modelLabels, sentiments, colors);
    }

    function updateSentimentScoresChart(scores) {
        // Destroy previous chart if exists
        if (sentimentScoresChart) {
            sentimentScoresChart.destroy();
        }

        // Create new chart
        const ctx = document.getElementById('sentiment-scores-chart').getContext('2d');
        sentimentScoresChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Positive', 'Neutral', 'Negative', 'Compound'],
                datasets: [{
                    label: 'Sentiment Scores',
                    data: [scores.positive, scores.neutral, scores.negative, scores.compound],
                    backgroundColor: [
                        'rgba(40, 167, 69, 0.7)',  // Positive - green
                        'rgba(108, 117, 125, 0.7)', // Neutral - gray
                        'rgba(220, 53, 69, 0.7)',  // Negative - red
                        'rgba(0, 123, 255, 0.7)'   // Compound - blue
                    ],
                    borderColor: [
                        'rgba(40, 167, 69, 1)',
                        'rgba(108, 117, 125, 1)',
                        'rgba(220, 53, 69, 1)',
                        'rgba(0, 123, 255, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Score: ${context.raw.toFixed(3)}`;
                            }
                        }
                    }
                }
            }
        });
    }

    function updateComparisonChart(labels, sentiments, colors) {
        // Prepare data - convert sentiments to numeric values for visualization
        const numericValues = sentiments.map(sentiment => {
            if (sentiment === 'positive') return 1;
            if (sentiment === 'negative') return -1;
            return 0; // neutral
        });

        // Destroy previous chart if exists
        if (comparisonChart) {
            comparisonChart.destroy();
        }

        // Create new chart
        const ctx = document.getElementById('comparison-chart').getContext('2d');
        comparisonChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Sentiment',
                    data: numericValues,
                    backgroundColor: colors,
                    borderColor: colors.map(color => color.replace('0.7', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                scales: {
                    x: {
                        min: -1,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                if (value === -1) return 'Negative';
                                if (value === 0) return 'Neutral';
                                if (value === 1) return 'Positive';
                                return '';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const sentiment = sentiments[context.dataIndex];
                                return `${sentiment.charAt(0).toUpperCase() + sentiment.slice(1)}`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Initialize by fetching available models
    async function fetchModels() {
        try {
            const response = await fetch('/models');
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            updateModelSelect(data.models);
        } catch (error) {
            console.error('Error fetching models:', error);
        }
    }

    function updateModelSelect(models) {
        // Clear existing options
        modelSelect.innerHTML = '';

        // Add models to select
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.name;
            modelSelect.appendChild(option);
        });
    }

    // Fetch models on load
    fetchModels();
}); 