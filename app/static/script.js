const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');

// Listen for Enter key press
userInput.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Prevent form submission
        sendMessage(); // Call sendMessage() function
    }
});

function sendMessage() {
    const message = userInput.value.trim();
    if (message === '') return;

    // Display user message
    appendMessage(message, 'user-message');

    // Clear the input field
    userInput.value = '';

    // Send message to server
    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    })
    .then(response => response.json())
    .then(data => {
        // Display bot response
        appendMessage(data.response, 'bot-message');
    });
}

function appendMessage(text, className) {
    const messageElement = document.createElement('div');
    messageElement.className = `message ${className}`;
    messageElement.textContent = text;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to the latest message
}


document.addEventListener('DOMContentLoaded', () => {
    // Display user info
    const userInfo = document.getElementById('user-info');
    fetch('/history')
        .then(response => response.json())
        .then(data => {
            if (data.history) {
                const historyList = document.getElementById('history-list');
                data.history.forEach(chat => {
                    const listItem = document.createElement('li');
                    listItem.textContent = chat.message;
                    historyList.appendChild(listItem);
                });
            }
        });
});

document.addEventListener('DOMContentLoaded', () => {
    // Fetch chat history only if user is logged in
    fetch('/history')
        .then(response => response.json())
        .then(data => {
            if (data.history && data.history.length > 0) {
                const historyList = document.getElementById('history-list');
                historyList.innerHTML = '';
                data.history.forEach(chat => {
                    const listItem = document.createElement('li');
                    listItem.textContent = chat.message;
                    listItem.onclick = () => loadChat(chat.id);  // Load specific chat
                    historyList.appendChild(listItem);
                });
            }
        });
});

function startNewChat() {
    // Clear chat box for new chat session
    document.getElementById('chat-box').innerHTML = '';
}

function loadChat(chatId) {
    // Fetch and display messages from a specific chat history
    fetch(`/load_chat/${chatId}`)
        .then(response => response.json())
        .then(data => {
            const chatBox = document.getElementById('chat-box');
            chatBox.innerHTML = '';  // Clear previous messages
            data.messages.forEach(message => {
                appendMessage(message.text, message.sender === 'user' ? 'user-message' : 'bot-message');
            });
        });
}


