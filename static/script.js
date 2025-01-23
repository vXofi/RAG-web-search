marked.setOptions({
    breaks: true,
    gfm: true,
    mangle: false,
    headerIds: false,
    langPrefix: 'language-',
    pedantic: false,
    smartLists: true,
    smartypants: false
});

const typingAnimation = () => {
    const cursor = document.createElement('span');
    cursor.className = 'typing-cursor';
    return cursor;
};

const debounce = (func, wait) => {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
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
    let renderQueue = [];
    let isRendering = false;

    sendButton.addEventListener('click', sendMessage);
    queryInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    function appendCursor() {
        const walker = document.createTreeWalker(
            contentDiv, 
            NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT,
            { acceptNode: (node) => 
                node.nodeType === Node.ELEMENT_NODE || 
                (node.nodeType === Node.TEXT_NODE && node.textContent.trim())
                ? NodeFilter.FILTER_ACCEPT 
                : NodeFilter.FILTER_REJECT 
            }
        );

        let lastNode = null;
        while (walker.nextNode()) lastNode = walker.currentNode;

        if (lastNode) {
            if (lastNode.nodeType === Node.TEXT_NODE) {
                const wrapper = document.createElement('span');
                wrapper.style.display = 'inline-block';
                wrapper.appendChild(document.createTextNode(lastNode.textContent));
                lastNode.parentNode.replaceChild(wrapper, lastNode);
                lastNode = wrapper;
            }
            lastNode.after(typingAnimation());
        } else {
            contentDiv.appendChild(typingAnimation());
        }
    }

    function removeCursor() {
        if (assistantMessageDiv) {
            const cursors = assistantMessageDiv.getElementsByClassName('typing-cursor');
            while(cursors.length > 0) {
                cursors[0].classList.add('hidden');
                setTimeout(() => cursors[0].remove(), 200);
            }
        }
    }

    const processTokens = debounce(() => {
        if (!isRendering) {
            isRendering = true;
            requestAnimationFrame(() => {
                removeCursor();
                
                if (eventSource && eventSource.readyState !== EventSource.CLOSED) {
                    let formatted = currentAssistantMessage
                        // newlines before list items
                        .replace(/(\S)(\n\d+\.)/g, '$1\n$2')
                        // double newlines between paragraphs
                        .replace(/(\n\n)(?=\S)/g, '\n\n\n')
                        // protect code blocks
                        .replace(/(```[\w]*\n)([^\n`]+)/g, '$1$2\n');
        
                    contentDiv.innerHTML = marked.parse(formatted, {
                        breaks: false,
                        gfm: true,
                        mangle: false
                    });
        
                    contentDiv.querySelectorAll('li').forEach(li => {
                        if (!li.textContent.includes('\n')) {
                            li.innerHTML = li.innerHTML.replace(/(\S)(<br>|$)/, '$1\n$2');
                        }
                    });
                    appendCursor();
                }
    
                isRendering = false;
                renderQueue = [];
            });
        }
    }, 100);

    eventSource.addEventListener('close', () => {
        console.log("SSE stream closed by server.");
        loadingIndicator.style.display = 'none';
        
        // Force a final render and remove cursor
        processTokens(); // Process any remaining tokens
        removeCursor();  // Explicitly remove cursor
        eventSource.close();
        
        // Add slight delay for final animation
        setTimeout(removeCursor, 100);
    });

    function addUserMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'user-message');
        messageDiv.textContent = message;
        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    async function sendMessage() {
        const query = queryInput.value.trim();
        if (!query) return;

        addUserMessage(query);
        queryInput.value = '';

        loadingIndicator.style.display = 'block';
        chatLog.scrollTop = chatLog.scrollHeight;

        assistantMessageDiv = document.createElement('div');
        assistantMessageDiv.classList.add('message', 'assistant-message');
        contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        assistantMessageDiv.appendChild(contentDiv);
        chatLog.appendChild(assistantMessageDiv);
        currentAssistantMessage = "";

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
            removeCursor();
            if (eventSource) {
                eventSource.close();
            }
        };

        eventSource.onmessage = (event) => {
            if (event.data === '<|endoftext|>') {
                eventSource.close();
                return;
            }
            
            const token = event.data;
            currentAssistantMessage += token;
            renderQueue.push(token);
            processTokens();
            chatLog.scrollTop = chatLog.scrollHeight;
        };

        eventSource.addEventListener('close', () => {
            console.log("SSE stream closed by server.");
            loadingIndicator.style.display = 'none';
            removeCursor();
            eventSource.close();
        });
    }

    function addSystemMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'system-message');
        messageDiv.innerHTML = marked.parse(message);
        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    function isStreaming() {
        return eventSource && eventSource.readyState === EventSource.OPEN;
    }

});