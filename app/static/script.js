const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const resultsTable = document.getElementById('resultsTable');
const polarityChart = document.getElementById('polarityChart').getContext('2d');

let entryCount = 0;

let chartLabels = [];
let chartData = [];

let chart = new Chart(polarityChart, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Polarity * Intensity',
            data: [],
            backgroundColor: 'rgba(75, 192, 192, 0.6)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
            fill: false,
        }]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});

userInput.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        sendMessage();
    }
});

function sendMessage() {
    const message = userInput.value.trim();
    if (message === '') return;
    appendMessage(message, 'user-message');
    userInput.value = '';

    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    })
    .then(response => response.json())
    .then(data => {
        appendMessage(data.response, 'bot-message');
        updateTableAndChart(data.latest_data);
    });
}

function appendMessage(text, className) {
    const messageElement = document.createElement('div');
    messageElement.className = `message ${className}`;

    if (className === 'bot-message') {
        const lines = text.split(', ');
        lines.forEach(line => {
            const lineElement = document.createElement('div');
            lineElement.textContent = line;
            messageElement.appendChild(lineElement);
        });
    } else {
        messageElement.textContent = text;
    }

    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function updateTableAndChart(latestData) {
    entryCount++;
    const row = resultsTable.querySelector('tbody').insertRow(0);
    row.insertCell(0).textContent = entryCount;
    row.insertCell(1).textContent = latestData.polarity;
    row.insertCell(2).textContent = latestData.concern;
    row.insertCell(3).textContent = latestData.category;
    row.insertCell(4).textContent = latestData.intensity;

    const polarityIntensity = latestData.polarity * latestData.intensity;

    chartLabels.push(`Entry ${entryCount}`);
    chartData.push(polarityIntensity);

    chart.data.labels = chartLabels;
    chart.data.datasets[0].data = chartData;
    chart.update();

    if (resultsTable.querySelector('tbody').rows.length > 4) {
        resultsTable.parentElement.style.overflowY = 'auto';
    }
}

function startNewChat() {
    chatBox.innerHTML = '';
    chart.data.labels = [];
    chart.data.datasets[0].data = [];
    chart.update();
    entryCount = 0;
    while (resultsTable.rows.length > 1) {
        resultsTable.deleteRow(1);
    }
}
