document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultCard = document.getElementById('result-card');
    const predictBtn = document.getElementById('predict-btn');
    const closeBtn = document.querySelector('.close-btn');

    // Close Result Card
    closeBtn.addEventListener('click', () => {
        resultCard.classList.remove('visible');
        setTimeout(() => {
            resultCard.classList.add('hidden');
        }, 500);
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show Loading State
        predictBtn.classList.add('loading');
        predictBtn.disabled = true;

        // Gather Data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        // Handle Checkbox Manually (checkboxes are not included in entries if unchecked)
        data.has_video = document.getElementById('has_video').checked;

        // Convert types
        data.goal = parseFloat(data.goal);
        data.duration_days = parseInt(data.duration_days);

        try {
            // Send Request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const result = await response.json();

            // Render Result
            displayResult(result);

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while fetching the prediction. Please try again.');
        } finally {
            // Reset Loading State
            predictBtn.classList.remove('loading');
            predictBtn.disabled = false;
        }
    });

    function displayResult(result) {
        const percentageEl = document.querySelector('.percentage');
        const circle = document.querySelector('.circle');
        const statusTitle = document.getElementById('prediction-status');
        const statusMessage = document.getElementById('prediction-message');
        const resultBody = document.querySelector('.result-body');

        // Reset previous classes
        resultBody.classList.remove('status-success', 'status-fail');

        // Update Text
        percentageEl.textContent = result.percent;

        // Update Ring
        const percent = result.success_probability * 100;
        // Stroke-dasharray: value, 100
        // We want to animate to 'percent, 100'
        // Reset to 0 first to animate? CSS transition handles it if we change the attr.
        // SVG stroke-dasharray is often set via attribute.
        circle.setAttribute('stroke-dasharray', `${percent}, 100`);

        // Update Status & Colors
        if (result.prediction === 'SUCCESS') {
            statusTitle.textContent = 'Likely Success';
            statusMessage.textContent = `Great news! This project has a high probability of reaching its goal (${result.percent}).`;
            resultBody.classList.add('status-success');
        } else {
            statusTitle.textContent = 'High Risk';
            statusMessage.textContent = `This project shows signs of risk. The estimated success probability is ${result.percent}. Consider adjusting the goal or duration.`;
            resultBody.classList.add('status-fail');
        }

        // Show Card
        resultCard.classList.remove('hidden');
        // Small delay to allow display:block to apply before opacity transition
        setTimeout(() => {
            resultCard.classList.add('visible');
            // Scroll to result
            resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 10);
    }
});
