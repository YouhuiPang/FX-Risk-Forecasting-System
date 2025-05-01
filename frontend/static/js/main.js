document.addEventListener('DOMContentLoaded', function() {
  // Fetch data from the API
  fetch('/data')
    .then(response => response.json())
    .then(data => {
      // Display risk score and information
      updateRiskDisplay(data);
      
      // Setup risk forecast chart
      setupForecastChart(data.recent_predictions);
      
      // Setup sentiment analysis
      setupSentimentAnalysis(data.sentiment_score);
    })
    .catch(error => console.error('Error fetching data:', error));

  // Setup the sentiment analysis button
  document.getElementById('analyzeSentiment').addEventListener('click', function() {
    const sentimentScore = -0.42; // Mock sentiment score for demo
    updateSentimentDisplay(sentimentScore);
  });
});

// Function to update the risk display
function updateRiskDisplay(data) {
  // Risk gauge
  setupRiskGauge(data.risk_index);
  
  // Risk level text
  const riskColors = {
    'LOW': '#10b981',
    'MEDIUM': '#f59e0b',
    'HIGH': '#ef4444'
  };
  
  const riskExplanations = {
    "Low": "Current conditions suggest stability. No meaningful downside pressure is expected over the next few days.",

    "Medium": "Risk signals have increased. While conditions are not critical, some short-term weakness may develop and should be monitored.",

    "High": "Several risk indicators point to a potential sharp decline. Active caution is advised, and defensive positioning may be appropriate."
  };


  // Update risk gauge value
  document.getElementById('riskScoreDisplay').textContent = data.risk_index;
  document.getElementById('riskScoreDisplay').style.color = riskColors[data.risk_label] || '#3b82f6';
  
  // Update risk text labels
  document.getElementById('riskTextLabel').textContent = data.risk_label + ' RISK';
  document.getElementById('riskTextExplain').textContent = riskExplanations[data.risk_label] || 'No description available.';
  
  // Update prediction card
  document.getElementById('prediction_rate').textContent = data.risk_index;

  document.getElementById('volatility_label').textContent = data.risk_label;
  document.getElementById('risk_percentage').textContent = data.risk_index;
  document.getElementById('prediction_date').textContent = data.prediction_date;
  document.getElementById('key_factors').textContent =
  Array.isArray(data.key_factors) && data.key_factors.length > 0
    ? data.key_factors.join(', ')
    : 'No dominant factors detected';

  const preds = data.recent_predictions;
  if (preds.length >= 2) {
    const today = preds[preds.length - 1];
    const yesterday = preds[preds.length - 2];

    const today_score = today.risk_prob_low * 20 + today.risk_prob_medium * 60 + today.risk_prob_high * 100;
    const yest_score = yesterday.risk_prob_low * 20 + yesterday.risk_prob_medium * 60 + yesterday.risk_prob_high * 100;

    const trendIcon = document.getElementById('riskTrendArrow');
    const labelText = document.getElementById('volatility_label');
    if (today_score > yest_score + 1) {
      trendIcon.className = 'fas fa-arrow-up h-5 w-5 mr-1 text-red-500';
      labelText.className = 'text-red-500';
    } else if (today_score < yest_score - 1) {
      trendIcon.className = 'fas fa-arrow-down h-5 w-5 mr-1 text-green-500';
      labelText.className = 'text-green-500';
    } else {
      trendIcon.className = 'fas fa-arrows-alt-h h-5 w-5 mr-1 text-yellow-500';
      labelText.className = 'text-yellow-500';
    }
  }
}

// Setup the risk gauge
function setupRiskGauge(score) {
  const canvas = document.getElementById('riskGauge');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  const outerRadius = Math.min(canvas.width, canvas.height) * 0.4;
  const innerRadius = outerRadius * 0.6;

  // Draw background (gray semicircle)
  ctx.beginPath();
  ctx.arc(centerX, centerY, outerRadius, Math.PI, 0, false);
  ctx.arc(centerX, centerY, innerRadius, 0, Math.PI, true);
  ctx.closePath();
  ctx.fillStyle = '#27272a';
  ctx.fill();

  // Compute sweep angle (proportional to score)
  const sweepAngle = (score / 100) * Math.PI;

  // Determine risk color
  let color = '#10b981'; // Low
  if (score >= 60) color = '#ef4444'; // High
  else if (score >= 30) color = '#f59e0b'; // Medium

  // Draw filled portion
  ctx.beginPath();
  ctx.arc(centerX, centerY, outerRadius, Math.PI, Math.PI + sweepAngle, false);
  ctx.arc(centerX, centerY, innerRadius, Math.PI + sweepAngle, Math.PI, true);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}


// Setup the forecast chart
function setupForecastChart(predictions) {
  // Format the data
  const chartData = predictions.map(item => {
    const date = new Date(item.date);
    const formattedDate = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

    const riskScore = Math.round(
      item.risk_prob_low * 0 +
      item.risk_prob_medium * 50 +
      item.risk_prob_high * 100
    );

    return { date: formattedDate, riskScore };
  });

  // Prepare chart data
  const labels = chartData.map(item => item.date);
  const values = chartData.map(item => item.riskScore);

  // Set colors based on risk score
  const colors = values.map(score => {
    if (score < 30) return '#10b981'; // Low
    if (score < 60) return '#f59e0b'; // Medium
    return '#ef4444'; // High
  });

  // Create chart
  const ctx = document.getElementById('riskForecastChart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Risk',
        data: values,
        borderColor: '#4169e1',
        backgroundColor: 'rgba(65,105,225,0.2)',
        pointBackgroundColor: colors,
        pointRadius: 5,
        tension: 0.2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function(context) {
              return 'Risk: ' + context.parsed.y;
            }
          }
        }
      },
      scales: {
        x: {
          ticks: { color: '#ccc' },
          grid: { color: '#2b2e44' }
        },
        y: {
          beginAtZero: true,
          min: 0,
          max: 100,
          ticks: { stepSize: 20, color: '#ccc' },
          grid: { color: '#2b2e44' }
        }
      }
    }
  });
}

// Setup and update sentiment analysis display
function setupSentimentAnalysis(score) {
  updateSentimentDisplay(score);
}

function updateSentimentDisplay(score) {
  const sentimentElement = document.getElementById('sentimentScore');
  
  // Format score to show + sign for positive values
  const formattedScore = score > 0 ? `+${score.toFixed(2)}` : score.toFixed(2);
  
  // Determine sentiment category
  let category, colorClass;
  if (score < -0.3) {
    category = 'Negative';
    colorClass = 'sentiment-negative';
  } else if (score > 0.3) {
    category = 'Positive';
    colorClass = 'sentiment-positive';
  } else {
    category = 'Neutral';
    colorClass = 'sentiment-neutral';
  }
  
  // Update display
  sentimentElement.textContent = `${formattedScore} (${category})`;
  sentimentElement.className = `font-bold mb-2 ${colorClass}`;
}