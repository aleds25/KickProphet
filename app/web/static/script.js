document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultCard = document.getElementById('result-card');
    const predictBtn = document.getElementById('predict-btn');
    const closeBtn = document.querySelector('.close-btn');
    const categorySelect = document.getElementById('category');
    const subCategorySelect = document.getElementById('sub_category');
    let categoryData = {};

    // Fetch Categories
    fetch('/static/categories.json')
        .then(response => response.json())
        .then(data => {
            categoryData = data;
            // Populate Main Categories
            for (const category in categoryData) {
                const option = document.createElement('option');
                option.value = category;
                option.textContent = category;
                categorySelect.appendChild(option);
            }
        })
        .catch(error => console.error('Error fetching categories:', error));

    // Handle Category Change
    categorySelect.addEventListener('change', function () {
        const selectedCategory = this.value;
        const subCategories = categoryData[selectedCategory] || [];

        // Clear existing options
        subCategorySelect.innerHTML = '<option value="" disabled selected>Select Sub-Category</option>';

        // Enable Select
        subCategorySelect.disabled = false;

        // Populate Sub-Categories
        subCategories.forEach(subCat => {
            const option = document.createElement('option');
            option.value = subCat;
            option.textContent = subCat;
            subCategorySelect.appendChild(option);
        });
    });

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
        data.prelaunch_activated = document.getElementById('prelaunch_activated').checked;

        // Convert types
        data.goal = parseFloat(data.goal);
        data.duration_days = parseInt(data.duration_days);
        data.preparation_days = parseFloat(data.preparation_days);

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
        // Elements
        const resultCard = document.getElementById('result-card');
        const predictionText = document.getElementById('prediction-text');
        const progressBar = document.getElementById('progress-bar');
        const progressPercent = document.getElementById('progress-percent');
        const closeBtn = document.querySelector('.close-btn');

        resultCard.classList.remove('hidden');
        // Trigger reflow
        void resultCard.offsetWidth;
        resultCard.classList.add('visible');

        // Parse percentage
        const percent = Math.round(result.success_probability * 100);

        // Update Text Logic
        let messageText = "";
        const prob = result.success_probability;

        if (prob < 0.1) messageText = "It's a disaster";
        else if (prob < 0.2) messageText = "Miracle needed";
        else if (prob < 0.3) messageText = "Uphill battle";
        else if (prob < 0.4) messageText = "High risk";
        else if (prob < 0.5) messageText = "Uncertain";
        else if (prob < 0.6) messageText = "Potential";
        else if (prob < 0.7) messageText = "Promising";
        else if (prob < 0.8) messageText = "Strong contender";
        else if (prob < 0.9) messageText = "Very likely";
        else messageText = "Bank on it!";

        // Update Content
        predictionText.textContent = messageText;
        progressPercent.textContent = result.percent;

        // Reset styles
        progressBar.style.width = '0%';
        predictionText.className = '';

        // Set Colors based on result (optional, but good for visual feedback)
        if (result.prediction === 'SUCCESS') {
            predictionText.classList.add('text-success');
        } else {
            predictionText.classList.add('text-fail');
        }

        // Animate Bar
        setTimeout(() => {
            progressBar.style.width = `${percent}%`;
        }, 100);

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
