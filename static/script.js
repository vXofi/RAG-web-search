document.addEventListener('DOMContentLoaded', () => {
    const chatLog = document.getElementById('chat-log');
    const queryInput = document.getElementById('query-input');
    const sendButton = document.getElementById('send-button');
    const loadingIndicator = document.getElementById('loading-indicator');
    let eventSource = null;

    sendButton.addEventListener('click', sendMessage);
    queryInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    async function sendMessage() {
        const query = queryInput.value.trim();
        if (!query) return;

        addUserMessage(query);
        queryInput.value = '';

        loadingIndicator.style.display = 'block';
        chatLog.scrollTop = chatLog.scrollHeight;

        let assistantMessageDiv = null;

        if (eventSource) {
            eventSource.close();
        }

        eventSource = new EventSource(`/rag?query=${encodeURIComponent(query)}`);

        eventSource.onopen = () => {
            loadingIndicator.style.display = 'none';
        };

        eventSource.onerror = (error) => {
            console.warn("SSE connection closed by server (normal).", error);
            loadingIndicator.style.display = 'none';
            // Optionally, remove the system message about the error
            // or replace it with a more informative message if needed.
            // addSystemMessage("Response complete.");
            if (eventSource) {
                eventSource.close();
            }
        };

        eventSource.onmessage = (event) => {
            const token = event.data;
            if (!assistantMessageDiv) {
                assistantMessageDiv = document.createElement('div');
                assistantMessageDiv.classList.add('message', 'assistant-message');
                chatLog.appendChild(assistantMessageDiv);
            }
            assistantMessageDiv.textContent += token;
            chatLog.scrollTop = chatLog.scrollHeight;
        };

        eventSource.addEventListener('close', () => {
            console.log("SSE stream closed by server.");
            loadingIndicator.style.display = 'none';
        });
    }

    function addUserMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'user-message');
        messageDiv.textContent = message;
        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    function addSystemMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'system-message');
        messageDiv.textContent = message;
        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight;
    }
});