marked.setOptions({
    breaks: true,
    gfm: true,
    mangle: false,
    headerIds: false,
    smartLists: true,
    smartypants: false
});

const typingAnimation = () => {
    const cursor = document.createElement('span');
    cursor.className = 'typing-cursor';
    return cursor;
};

function preserveListStructure(text) {
    return text
        .replace(/(\d+)\. /g, '\n$1. ')
        .replace(/(\n\d+\. )/g, '\n\n$1')
        .replace(/(\n)- /g, '\n\n- ');
}

document.addEventListener('DOMContentLoaded', () => {
    const chatLog = document.getElementById('chat-log');
    const queryInput = document.getElementById('query-input');
    const sendButton = document.getElementById('send-button');
    const loadingIndicator = document.getElementById('loading-indicator');
    
    let eventSource = null;
    let currentAssistantMessage = "";
    let assistantMessageDiv = null;
    let contentDiv = null;
    let streamingText = null;

    const appendCursor = () => {
        const existingCursors = contentDiv.getElementsByClassName('typing-cursor');
        if (existingCursors.length === 0) {
            contentDiv.appendChild(typingAnimation());
        }
    };

    const removeCursor = () => {
        const cursors = contentDiv.getElementsByClassName('typing-cursor');
        while(cursors.length > 0) {
            cursors[0].remove();
        }
    };

    const addUserMessage = (message) => {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'user-message');
        messageDiv.textContent = message;
        chatLog.appendChild(messageDiv);
    };

    sendButton.addEventListener('click', () => sendMessage());
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    function sendMessage() {
        const query = queryInput.value.trim();
        if (!query) return;

        if (eventSource) eventSource.close();
        queryInput.value = '';
        currentAssistantMessage = '';
        addUserMessage(query);

        assistantMessageDiv = document.createElement('div');
        assistantMessageDiv.classList.add('message', 'assistant-message');
        contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        streamingText = document.createElement('pre');
        streamingText.style.whiteSpace = 'pre-wrap';
        
        contentDiv.appendChild(streamingText);
        assistantMessageDiv.appendChild(contentDiv);
        chatLog.appendChild(assistantMessageDiv);
        loadingIndicator.style.display = 'block';
        
        eventSource = new EventSource(`/rag?query=${encodeURIComponent(query)}`);

        eventSource.onopen = () => {
            loadingIndicator.style.display = 'none';
            appendCursor();
        };

        eventSource.onmessage = (event) => {
            const token = event.data;
            currentAssistantMessage += token;
            streamingText.textContent += token;
            removeCursor();
            appendCursor();
            
            const scrollThreshold = 100;
            const nearBottom = chatLog.scrollHeight - chatLog.clientHeight <= chatLog.scrollTop + scrollThreshold;
            if (nearBottom) {
                chatLog.scrollTop = chatLog.scrollHeight;
            }
        };

        eventSource.onerror = (error) => {
            console.error('SSE error:', error);
            loadingIndicator.style.display = 'none';
            removeCursor();
            eventSource.close();
        };

        eventSource.addEventListener('close', () => {
            const formatted = marked.parse(preserveListStructure(currentAssistantMessage));
            contentDiv.removeChild(streamingText);
            contentDiv.innerHTML = formatted;
            removeCursor();
            eventSource.close();
            
            requestAnimationFrame(() => {
                chatLog.scrollTop = chatLog.scrollHeight;
            });
        });
    }
});