document.addEventListener('DOMContentLoaded', function() {
const chatWindow = document.getElementById('chat-window');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatHistory = document.getElementById('chat-history');
const newChatBtn = document.getElementById('new-chat');

let lastUserProblem = '';

// Modal and chat history logic
const modal = document.getElementById('save-delete-modal');
const saveChatBtn = document.getElementById('save-chat-btn');
const deleteChatBtn = document.getElementById('delete-chat-btn');
const cancelModalBtn = document.getElementById('cancel-modal-btn');
const chatNameInput = document.getElementById('chat-name-input');
const chatNameInputContainer = document.querySelector('.chat-name-input');

let currentChat = [];
let chatHistoryList = [];
let activeHistoryIndex = null;

const imageUpload = document.getElementById('image-upload');
const imageUploadLabel = document.getElementById('image-upload-label');
let lastImageDiagnosis = '';
let usingImageDiagnosis = false;

const fileUpload = document.getElementById('file-upload');
const fileUploadLabel = document.getElementById('file-upload-label');

// Model selector functionality
const modelToggle = document.getElementById('model-toggle');
const modelDropdown = document.getElementById('model-dropdown');
const currentModelDisplay = document.getElementById('current-model');
const modelOptions = document.querySelectorAll('.model-option');

// Available models with their icons and display names
const availableModels = {
  'llama3:latest': { icon: '🦙', name: 'Llama 3', size: '4.7GB' },
  'mistral:latest': { icon: '🌪️', name: 'Mistral', size: '4.1GB' },
  'llava:latest': { icon: '👁️', name: 'LLaVA', size: '4.7GB' },
  'llama3.2:3b': { icon: '🦙', name: 'Llama 3.2 3B', size: '2.0GB' },
  'phi:latest': { icon: 'φ', name: 'Phi', size: '1.6GB' }
};

// Current selected model (default to llama3:latest)
let currentModel = localStorage.getItem('selected_model') || 'llama3:latest';

// Document management
let uploadedDocuments = [];
let currentDocumentId = null; // Track the currently active document

// Load uploaded documents
async function loadUploadedDocuments() {
  try {
    const response = await fetch('http://localhost:8080/documents');
    if (response.ok) {
      const data = await response.json();
      uploadedDocuments = data.documents || [];
      updateDocumentList();
    }
  } catch (err) {
    console.error('Error loading documents:', err);
  }
}

function updateDocumentList() {
  // You can add a document list UI here if needed
  console.log('Uploaded documents:', uploadedDocuments);
  
  // Create document selector if there are multiple documents
  const chatWindow = document.getElementById('chat-window');
  const existingSelector = document.querySelector('.document-selector');
  if (existingSelector) {
    existingSelector.remove();
  }
  
  if (uploadedDocuments.length > 0) {
    const selector = document.createElement('div');
    selector.className = 'document-selector';
    selector.innerHTML = `
      <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 12px; margin: 8px 0;">
        <div style="font-weight: bold; margin-bottom: 8px; color: #495057;">📚 Available Documents:</div>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
          ${uploadedDocuments.map((doc, index) => `
            <button class="doc-select-btn" data-doc-id="${doc.id}" style="
              background: ${currentDocumentId === doc.id ? '#007bff' : '#e9ecef'}; 
              color: ${currentDocumentId === doc.id ? 'white' : '#495057'};
              border: 1px solid #dee2e6; border-radius: 4px; padding: 4px 8px; 
              font-size: 12px; cursor: pointer; transition: all 0.2s;
            ">${doc.filename || `Document ${index + 1}`}</button>
          `).join('')}
        </div>
      </div>
    `;
    
    // Insert after welcome message
    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) {
      welcomeMessage.parentNode.insertBefore(selector, welcomeMessage.nextSibling);
    } else {
      chatWindow.insertBefore(selector, chatWindow.firstChild);
    }
    
    // Add event listeners to document selection buttons
    const docButtons = selector.querySelectorAll('.doc-select-btn');
    docButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const docId = btn.dataset.docId;
        currentDocumentId = docId;
        
        // Update button styles
        docButtons.forEach(b => {
          b.style.background = b.dataset.docId === docId ? '#007bff' : '#e9ecef';
          b.style.color = b.dataset.docId === docId ? 'white' : '#495057';
        });
        
        // Update indicator
        updateActiveDocumentIndicator();
        
        // Show confirmation
        appendMessage(`📄 Now using document: ${btn.textContent}`, 'bot');
      });
    });
  }
}

function updateActiveDocumentIndicator() {
  const chatWindow = document.getElementById('chat-window');
  
  // Remove any existing document indicator
  const existingIndicator = document.querySelector('.document-indicator');
  if (existingIndicator) {
    existingIndicator.remove();
  }
  
  // Add indicator if a document is active
  if (currentDocumentId) {
    const indicator = document.createElement('div');
    indicator.className = 'document-indicator';
    indicator.innerHTML = `
      <div style="background: #e0f2fe; border: 1px solid #0288d1; border-radius: 8px; padding: 8px 12px; margin: 8px 0; font-size: 14px; color: #0277bd;">
        📄 Document Active - Your questions will be answered based on your uploaded document
      </div>
    `;
    
    // Insert after welcome message or at the top
    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) {
      welcomeMessage.parentNode.insertBefore(indicator, welcomeMessage.nextSibling);
    } else {
      chatWindow.insertBefore(indicator, chatWindow.firstChild);
    }
  }
}

// Handle file upload
fileUpload.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  // Show upload status
  appendMessage(`📄 Uploading ${file.name}...`, 'bot');
  
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('http://localhost:8080/upload', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.status}`);
    }

    const data = await response.json();
    
    // Remove upload status message
    chatWindow.removeChild(chatWindow.lastChild);
    
    // Show success message with summary
    appendMessage(`✅ Document uploaded successfully!\n\n📋 Summary:\n${data.summary}`, 'bot');
    
    // Set this as the current active document
    currentDocumentId = data.doc_id;
    
    // Reload document list
    await loadUploadedDocuments();
    updateActiveDocumentIndicator(); // Update indicator after successful upload
    
  } catch (err) {
    console.error('Upload error:', err);
    chatWindow.removeChild(chatWindow.lastChild);
    
    // More detailed error message
    let errorMessage = '❌ Failed to upload document. ';
    if (err.message.includes('fetch')) {
      errorMessage += 'Server is not running. Please start the server first.';
    } else if (err.message.includes('413')) {
      errorMessage += 'File is too large. Please try a smaller file.';
    } else if (err.message.includes('415')) {
      errorMessage += 'Unsupported file type. Please use PDF, DOCX, or TXT files.';
    } else {
      errorMessage += `Error: ${err.message}`;
    }
    
    appendMessage(errorMessage, 'bot');
  }
  
  // Clear file input
  event.target.value = '';
});

function loadHistoryFromStorage() {
  const data = localStorage.getItem('nutrimed_chat_history');
  chatHistoryList = data ? JSON.parse(data) : [];
}

function saveHistoryToStorage() {
  localStorage.setItem('nutrimed_chat_history', JSON.stringify(chatHistoryList));
}

function renderChatHistory() {
  chatHistory.innerHTML = '';
  chatHistoryList.forEach((chat, idx) => {
    const item = document.createElement('div');
    item.className = 'chat-history-item' + (idx === activeHistoryIndex ? ' active' : '');
    
    const titleSpan = document.createElement('span');
    titleSpan.textContent = chat.title || ('Chat ' + (idx+1));
    titleSpan.title = chat.title || '';
    
    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'delete-btn';
    deleteBtn.textContent = '×';
    deleteBtn.title = 'Delete chat';
    deleteBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteChatFromHistory(idx);
    });
    
    item.appendChild(titleSpan);
    item.appendChild(deleteBtn);
    
    item.addEventListener('click', () => {
      loadChatFromHistory(idx);
    });
    chatHistory.appendChild(item);
  });
}

function deleteChatFromHistory(idx) {
  chatHistoryList.splice(idx, 1);
  saveHistoryToStorage();
  if (activeHistoryIndex === idx) {
    activeHistoryIndex = null;
    clearChatWindow();
    resetChatState();
  } else if (activeHistoryIndex > idx) {
    activeHistoryIndex--;
  }
  renderChatHistory();
}

function loadChatFromHistory(idx) {
  activeHistoryIndex = idx;
  const chat = chatHistoryList[idx];
  chatWindow.innerHTML = '';
  chat.messages.forEach(msg => {
    appendMessage(msg.text, msg.sender);
  });
  renderChatHistory();
  optionButtons.forEach(btn => btn.disabled = true);
  userHasEnteredInput = chat.messages.some(m => m.sender === 'user');
  lastUserProblem = '';
  for (let i = chat.messages.length - 1; i >= 0; i--) {
    if (chat.messages[i].sender === 'user') {
      lastUserProblem = chat.messages[i].text;
      break;
    }
  }
  currentChat = chat.messages.slice();
}

function clearChatWindow() {
  chatWindow.innerHTML = `
    <div class="welcome-message">
      <h3>🤖 Welcome to Timeless Tonic!</h3>
      <p>💡 I help with symptoms, causes, medicines, nutrition, and physical wellness — all in a detailed, readable format.</p>
      <p>💬 How can I help you today? (Example: 'I have nausea', 'I'm feeling weak', 'joint pain', etc.)</p>
      <div class="suggestion-buttons">
        <button class="suggestion-btn" onclick="sendPredefinedMessage('I have a headache', null)">Headache</button>
        <button class="suggestion-btn" onclick="sendPredefinedMessage('I have joint pain', null)">Joint Pain</button>
        <button class="suggestion-btn" onclick="sendPredefinedMessage('I feel tired', null)">Fatigue</button>
        <button class="suggestion-btn" onclick="sendPredefinedMessage('I have stomach pain', null)">Stomach Pain</button>
      </div>
    </div>
  `;
}

function resetChatState() {
  currentChat = [];
  lastUserProblem = '';
  lastImageDiagnosis = '';
  usingImageDiagnosis = false;
  currentDocumentId = null; // Clear the current document when starting new chat
  userHasEnteredInput = false;
  optionButtons.forEach(btn => btn.disabled = true);
  
  // Remove document indicator and selector
  const existingIndicator = document.querySelector('.document-indicator');
  if (existingIndicator) {
    existingIndicator.remove();
  }
  
  const existingSelector = document.querySelector('.document-selector');
  if (existingSelector) {
    existingSelector.remove();
  }
}

function showModal() {
  modal.style.display = 'flex';
  chatNameInputContainer.style.display = 'none';
  chatNameInput.value = '';
}
function hideModal() {
  modal.style.display = 'none';
  chatNameInputContainer.style.display = 'none';
  chatNameInput.value = '';
}

function isChatEmpty() {
  return currentChat.length === 0 || (currentChat.length === 1 && currentChat[0].sender === 'bot');
}

function appendMessage(text, sender) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${sender}`;
  
  if (sender === 'bot') {
    const logoImg = document.createElement('img');
    logoImg.src = '/static/assets/logo1.png';
    logoImg.className = 'bot-logo';
    messageDiv.appendChild(logoImg);
  }
  
  const bubbleDiv = document.createElement('div');
  bubbleDiv.className = 'bubble';
  
  // Enhanced message formatting for better readability
  if (sender === 'bot') {
    // Format bot messages with better styling
    const formattedText = formatBotMessage(text);
    bubbleDiv.innerHTML = formattedText;
  } else {
    bubbleDiv.textContent = text;
  }
  
  messageDiv.appendChild(bubbleDiv);
  chatWindow.appendChild(messageDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  
  if (typeof currentChat !== 'undefined') {
    currentChat.push({ text, sender });
  }
}

function formatBotMessage(text) {
  // Enhanced formatting for bot messages
  let formattedText = text;
  
  // Format section headers (🏥, ⚠️, 💊, 🥗, 🏃‍♂️, 🔍)
  formattedText = formattedText.replace(
    /(🏥|⚠️|💊|🥗|🏃‍♂️|🔍|📋|💡)\s*\*\*(.*?)\*\*/g,
    '<div class="section-header"><span class="section-icon">$1</span><span class="section-title">$2</span></div>'
  );
  
  // Format bullet points
  formattedText = formattedText.replace(
    /^•\s*(.*)$/gm,
    '<div class="bullet-point">• $1</div>'
  );
  
  // Format numbered points
  formattedText = formattedText.replace(
    /^(\d+)\.\s*(.*)$/gm,
    '<div class="numbered-point"><span class="number">$1.</span> $2</div>'
  );
  
  // Format bold text
  formattedText = formattedText.replace(
    /\*\*(.*?)\*\*/g,
    '<strong>$1</strong>'
  );
  
  // Add line breaks for better spacing
  formattedText = formattedText.replace(/\n/g, '<br>');
  
  return formattedText;
}

async function sendPredefinedMessage(message, btn) {
  appendMessage(message, 'user');
  appendMessage('Timeless Tonic is typing...', 'bot');
  
  try {
    const response = await fetch('http://localhost:8080/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        message: message,
        model: currentModel,
        doc_id: currentDocumentId // Include the current document ID
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    chatWindow.removeChild(chatWindow.lastChild); // Remove typing message
    
    if (Array.isArray(data.responses)) {
      // Display each response as a separate message for better readability
      for (const msg of data.responses) {
        if (msg.trim()) {
          appendMessage(msg.trim(), 'bot');
        }
      }
    } else if (data.response) {
      appendMessage(data.response, 'bot');
    } else {
      appendMessage('Sorry, I received an unexpected response format.', 'bot');
    }
    
    if (btn) btn.disabled = true;
    
  } catch (err) {
    console.error('Error:', err);
    chatWindow.removeChild(chatWindow.lastChild); // Remove typing message
    appendMessage('Sorry, there was an error connecting to Timeless Tonic. Please check your connection and try again.', 'bot');
  }
}

// Initially disable the topic buttons
const optionButtons = document.querySelectorAll('.option-btn');
optionButtons.forEach(btn => btn.disabled = true);

let userHasEnteredInput = false;

// Initialize the application
async function initializeApp() {
  loadHistoryFromStorage();
  renderChatHistory();
  await loadUploadedDocuments();
  initializeModelSelector();
}

// Model selector functionality
function initializeModelSelector() {
  // Set initial model display
  updateModelDisplay();
  
  // Toggle dropdown
  modelToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    const isOpen = modelDropdown.style.display === 'block';
    modelDropdown.style.display = isOpen ? 'none' : 'block';
    modelToggle.classList.toggle('active', !isOpen);
  });
  
  // Handle model selection
  modelOptions.forEach(option => {
    option.addEventListener('click', () => {
      clearErrorMessages();
      const selectedModel = option.dataset.model;
      currentModel = selectedModel;
      localStorage.setItem('selected_model', selectedModel);
      
      // Update display
      updateModelDisplay();
      
      // Update active state
      modelOptions.forEach(opt => opt.classList.remove('active'));
      option.classList.add('active');
      
      // Close dropdown
      modelDropdown.style.display = 'none';
      modelToggle.classList.remove('active');
      
      // Show model change notification
      const modelInfo = availableModels[selectedModel];
      appendMessage(`🤖 Switched to ${modelInfo.name} model`, 'bot');
    });
  });
  
  // Close dropdown when clicking outside
  document.addEventListener('click', (e) => {
    if (!modelToggle.contains(e.target) && !modelDropdown.contains(e.target)) {
      modelDropdown.style.display = 'none';
      modelToggle.classList.remove('active');
    }
  });
  
  // Set initial active state
  const currentOption = document.querySelector(`[data-model="${currentModel}"]`);
  if (currentOption) {
    currentOption.classList.add('active');
  }
}

function updateModelDisplay() {
  const modelInfo = availableModels[currentModel];
  if (modelInfo) {
    currentModelDisplay.textContent = modelInfo.icon;
    modelToggle.title = `Current: ${modelInfo.name} (${modelInfo.size})`;
  }
}

// Start the app
initializeApp();

function clearErrorMessages() {
  // Remove all error messages from the chat window
  const errorMessages = chatWindow.querySelectorAll('.message.bot');
  errorMessages.forEach(msg => {
    if (msg.textContent && msg.textContent.includes('Sorry, there was an error processing your request.')) {
      msg.remove();
    }
  });
}

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  clearErrorMessages();
  const symptom = userInput.value.trim();
  if (!symptom) return;
  
  appendMessage(symptom, 'user');
  userInput.value = '';
  lastUserProblem = symptom; // Store the last user problem
  userHasEnteredInput = true;
  
  appendMessage('Timeless Tonic is typing...', 'bot');
  
  try {
    const response = await fetch('http://localhost:8080/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        message: symptom,
        model: currentModel,
        doc_id: currentDocumentId // Include the current document ID
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    chatWindow.removeChild(chatWindow.lastChild); // Remove typing message
    
    if (Array.isArray(data.responses)) {
      // Display each response as a separate message for better readability
      for (const msg of data.responses) {
        if (msg.trim()) {
          appendMessage(msg.trim(), 'bot');
        }
      }
    } else if (data.response) {
      appendMessage(data.response, 'bot');
    } else {
      appendMessage('Sorry, I received an unexpected response format.', 'bot');
    }
    
    // Enable topic buttons after receiving response
    optionButtons.forEach(btn => btn.disabled = false);
    
  } catch (err) {
    console.error('Error:', err);
    chatWindow.removeChild(chatWindow.lastChild); // Remove typing message
    appendMessage('Sorry, there was an error connecting to Timeless Tonic. Please check your connection and try again.', 'bot');
  }
});

// Make '+ New Chat' button clear the chat and show welcome message, and reset topic buttons
newChatBtn.addEventListener('click', async () => {
  if (!isChatEmpty()) {
    showModal();
  } else {
    // Reset session on server
    try {
      await fetch('http://localhost:8080/reset-session', {
        method: 'POST'
      });
    } catch (err) {
      console.error('Failed to reset session:', err);
    }
    
    clearChatWindow();
    resetChatState();
  }
});

// Make the topic buttons send focused, detailed prompts about the last user problem only once per session
const predefinedMessages = [
  (problem) => `List ONLY the symptoms of "${problem}". Start the first message with the heading 'Symptoms:', then send each symptom as a separate message in numbered order (1., 2., 3., etc.), without repeating the heading. Do not include impacts, medications, or exercises.`,
  (problem) => `List ONLY the impacts of "${problem}". Start the first message with the heading 'Impacts:', then send each impact as a separate message in numbered order (1., 2., 3., etc.), without repeating the heading. Do not include symptoms, medications, or exercises.`,
  (problem) => `List ONLY the medications and their effects for "${problem}". Start the first message with the heading 'Medications:', then send each medication and its effect as a separate message in numbered order (1., 2., 3., etc.), without repeating the heading. Do not include symptoms, impacts, or exercises.`,
  (problem) => `List ONLY the exercises and physical activities for "${problem}". Start the first message with the heading 'Exercises:', then send each exercise or step as a separate message in numbered order (1., 2., 3., etc.), without repeating the heading. Do not include symptoms, impacts, or medications. List exercises only once, step by step.`,
  (problem) => `List ONLY the nutrients, vitamins, and dietary recommendations for "${problem}". Start the first message with the heading 'Nutrients:', then send each nutrient, vitamin, or dietary recommendation as a separate message in numbered order (1., 2., 3., etc.), without repeating the heading. Do not include symptoms, impacts, medications, or exercises. Focus on nutritional support and dietary changes that can help with this condition.`
];
optionButtons.forEach((btn, idx) => {
  btn.addEventListener('click', () => {
    let problem = lastUserProblem;
    if (usingImageDiagnosis && lastImageDiagnosis) {
      problem = lastImageDiagnosis;
    }
    if (!btn.disabled && problem) {
      appendMessage(btn.textContent, 'user');
      appendMessage('Timeless Tonic is typing...', 'bot');
      fetch('http://localhost:8080/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: predefinedMessages[idx](problem),
          model: currentModel,
          doc_id: currentDocumentId // Include the current document ID
        })
      })
      .then(response => response.json())
      .then(data => {
        chatWindow.removeChild(chatWindow.lastChild);
        if (Array.isArray(data.responses)) {
          for (const msg of data.responses) {
            appendMessage(msg, 'bot');
          }
        } else if (data.response) {
          appendMessage(data.response, 'bot');
        }
        btn.disabled = true;
      })
      .catch(err => {
        chatWindow.removeChild(chatWindow.lastChild);
        appendMessage('Sorry, there was an error connecting to Timeless Tonic.', 'bot');
      });
    }
  });
});

// Intercept New Chat button
newChatBtn.addEventListener('click', () => {
  if (!isChatEmpty()) {
    showModal();
  } else {
    clearChatWindow();
    resetChatState();
  }
});

saveChatBtn.addEventListener('click', () => {
  if (chatNameInputContainer.style.display === 'none') {
    // First click - show name input
    chatNameInputContainer.style.display = 'block';
    chatNameInput.focus();
    saveChatBtn.textContent = 'Confirm Save';
  } else {
    // Second click - actually save
    const chatName = chatNameInput.value.trim();
    if (!chatName) {
      alert('Please enter a chat name');
      return;
    }
    
    if (!isChatEmpty()) {
      chatHistoryList.push({
        title: chatName,
        messages: currentChat,
        timestamp: Date.now()
      });
      saveHistoryToStorage();
      renderChatHistory();
    }
    clearChatWindow();
    resetChatState();
    hideModal();
    saveChatBtn.textContent = 'Save'; // Reset button text
  }
});

deleteChatBtn.addEventListener('click', async () => {
  // Reset session on server
  try {
    await fetch('http://localhost:8080/reset-session', {
      method: 'POST'
    });
  } catch (err) {
    console.error('Failed to reset session:', err);
  }
  
  clearChatWindow();
  resetChatState();
  hideModal();
  saveChatBtn.textContent = 'Save'; // Reset button text
});

cancelModalBtn.addEventListener('click', () => {
  hideModal();
  saveChatBtn.textContent = 'Save'; // Reset button text
});

// Handle Enter key in chat name input
chatNameInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') {
    saveChatBtn.click();
  }
});

// On page load, load history
loadHistoryFromStorage();
renderChatHistory();

// When user sends a text message, reset image diagnosis mode
chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const symptom = userInput.value.trim();
  if (!symptom) return;
  
  appendMessage(symptom, 'user');
  userInput.value = '';
  lastUserProblem = symptom; // Store the last user problem
  usingImageDiagnosis = false; // Reset image diagnosis mode
  lastImageDiagnosis = '';
  
  appendMessage('Timeless Tonic is typing...', 'bot');
  try {
    const response = await fetch('http://localhost:8080/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: symptom })
    });
    if (!response.ok) throw new Error("Network response was not ok");
    const data = await response.json();
    chatWindow.removeChild(chatWindow.lastChild);
    if (Array.isArray(data.responses)) {
      for (const msg of data.responses) {
        appendMessage(msg, 'bot');
      }
    } else if (data.response) {
      appendMessage(data.response, 'bot');
    }
    if (!userHasEnteredInput) {
      optionButtons.forEach(btn => btn.disabled = false);
      userHasEnteredInput = true;
    }
  } catch (err) {
    chatWindow.removeChild(chatWindow.lastChild);
    appendMessage('Sorry, there was an error connecting to Timeless Tonic.', 'bot');
  }
});

imageUpload.addEventListener('change', async function() {
  const file = this.files[0];
  if (!file) return;

  // Show image preview in chat as user message
  const reader = new FileReader();
  reader.onload = function(evt) {
    const img = document.createElement('img');
    img.src = evt.target.result;
    img.className = 'uploaded-image-preview';
    appendMessage('Uploaded image:', 'user');
    chatWindow.appendChild(img);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  };
  reader.readAsDataURL(file);

  appendMessage('Analyzing image...', 'bot');

  // Send image to backend
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('http://localhost:8080/upload-image', {
      method: 'POST',
      body: formData
    });
    if (!response.ok) throw new Error('Server error');
    const data = await response.json();
    chatWindow.removeChild(chatWindow.lastChild); // Remove "Analyzing image..." message
    if (data.llama3_response) {
      appendMessage(data.llama3_response, 'bot');
      lastImageDiagnosis = data.llama3_response;
      usingImageDiagnosis = true;
      lastUserProblem = data.llama3_response;
      optionButtons.forEach(btn => btn.disabled = false);
    } else {
      appendMessage('Sorry, I could not analyze the image.', 'bot');
      lastImageDiagnosis = '';
      usingImageDiagnosis = false;
    }
  } catch (err) {
    chatWindow.removeChild(chatWindow.lastChild);
    appendMessage('Sorry, there was an error analyzing the image.', 'bot');
    lastImageDiagnosis = '';
    usingImageDiagnosis = false;
  }
  imageUpload.value = '';
});

// Option buttons use image diagnosis if available
optionButtons.forEach((btn, idx) => {
  btn.addEventListener('click', () => {
    let problem = lastUserProblem;
    if (usingImageDiagnosis && lastImageDiagnosis) {
      problem = lastImageDiagnosis;
    }
    if (!btn.disabled && problem) {
      appendMessage(btn.textContent, 'user');
      appendMessage('Timeless Tonic is typing...', 'bot');
      fetch('http://localhost:8080/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: predefinedMessages[idx](problem) })
      })
      .then(response => response.json())
      .then(data => {
        chatWindow.removeChild(chatWindow.lastChild);
        if (Array.isArray(data.responses)) {
          for (const msg of data.responses) {
            appendMessage(msg, 'bot');
          }
        } else if (data.response) {
          appendMessage(data.response, 'bot');
        }
        btn.disabled = true;
      })
      .catch(err => {
        chatWindow.removeChild(chatWindow.lastChild);
        appendMessage('Sorry, there was an error connecting to Timeless Tonic.', 'bot');
      });
    }
  });
});

fileUpload.addEventListener('change', async function() {
  const file = this.files[0];
  if (!file) return;

  // Show file name in chat as user message
  appendMessage(`Uploaded file: ${file.name}`, 'user');

  // Prepare form data
  const formData = new FormData();
  formData.append('file', file);

  // Show uploading message
  appendMessage('Uploading and processing document...', 'bot');

  try {
    const response = await fetch('http://localhost:8080/upload', {
      method: 'POST',
      body: formData
    });
    const data = await response.json();
    chatWindow.removeChild(chatWindow.lastChild); // Remove uploading message
    if (data.summary && data.extracted_text) {
      // Show extracted text (short preview if long)
      let preview = data.extracted_text;
      if (preview.length > 600) {
        preview = preview.slice(0, 600) + '...';
      }
      appendMessage('Extracted document text (preview):<br><div style="max-height:120px;overflow:auto;background:#23252b;padding:8px;border-radius:6px;font-size:0.95em;">' + preview.replace(/\n/g, '<br>') + '</div>', 'bot');
      // Show summary
      appendMessage('<strong>Summary:</strong><br>' + data.summary.replace(/\n/g, '<br>'), 'bot');
    } else if (data.error) {
      appendMessage('Error: ' + data.error, 'bot');
    } else {
      appendMessage('Failed to upload or process document.', 'bot');
    }
  } catch (err) {
    chatWindow.removeChild(chatWindow.lastChild);
    appendMessage('Sorry, there was an error uploading the document.', 'bot');
  }
  fileUpload.value = '';
});

// === THEME TOGGLE ===
const themeToggle = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');
const THEME_KEY = 'nutrimed_theme';
const themeOrder = ['light', 'dark'];
const themeIcons = {
  'light': '🌞',
  'dark': '🌙'
};

console.log('Theme toggle element:', themeToggle);
console.log('Theme icon element:', themeIcon);

// Test if we can find the button
setTimeout(() => {
  const testButton = document.getElementById('theme-toggle');
  console.log('Delayed test - Theme toggle element:', testButton);
  if (testButton) {
    console.log('Button found, adding test click');
    testButton.onclick = function() {
      alert('Button clicked!');
      const current = getSavedTheme();
      const next = getNextTheme(current);
      setTheme(next);
    };
  }
}, 1000);

function setTheme(theme) {
  console.log('Setting theme to:', theme);
  document.body.classList.remove('theme-light', 'theme-dark');
  if (theme === 'light') {
    // Remove dark theme styles and apply light theme
    document.body.style.background = '#f5f7fa';
    document.body.style.color = '#22223b';
    
    // Reset sidebar to light theme
    const sidebar = document.querySelector('.sidebar');
    if (sidebar) {
      sidebar.style.background = '#fff';
      sidebar.style.boxShadow = '2px 0 12px 0 #e0e7ff';
      sidebar.style.borderRight = 'none';
    }
    
    // Reset main header to light theme
    const mainHeader = document.querySelector('.main-header');
    if (mainHeader) {
      mainHeader.style.background = '#fff';
      mainHeader.style.boxShadow = 'none';
      mainHeader.style.borderBottom = 'none';
    }
    
    // Reset chat form to light theme
    const chatForm = document.getElementById('chat-form');
    if (chatForm) {
      chatForm.style.background = '#fff';
      chatForm.style.boxShadow = '0 -2px 12px #818cf822';
      chatForm.style.borderTop = 'none';
    }
    
    // Reset input field to light theme
    const userInput = document.getElementById('user-input');
    if (userInput) {
      userInput.style.background = '#f5f7fa';
      userInput.style.color = '#3730a3';
      userInput.style.border = '1px solid #818cf822';
    }
    
    // Reset welcome message to light theme
    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) {
      welcomeMessage.style.background = '#fff';
      welcomeMessage.style.color = '#3730a3';
      welcomeMessage.style.boxShadow = '0 2px 16px #818cf822';
      welcomeMessage.style.border = 'none';
    }
    
    // Reset chat history items to light theme
    const chatHistoryItems = document.querySelectorAll('.chat-history-item');
    chatHistoryItems.forEach(item => {
      item.style.background = '#f5f7fa';
      item.style.color = '#3730a3';
      item.style.boxShadow = '0 1px 4px #818cf811';
      item.style.border = 'none';
    });
    
    // Reset bubbles to light theme
    const bubbles = document.querySelectorAll('.bubble');
    bubbles.forEach(bubble => {
      bubble.style.background = '#f5f7fa';
      bubble.style.color = '#3730a3';
      bubble.style.border = 'none';
    });
    
    // Reset user bubbles to light theme
    const userBubbles = document.querySelectorAll('.message.user .bubble');
    userBubbles.forEach(bubble => {
      bubble.style.background = '#818cf8';
      bubble.style.color = '#fff';
      bubble.style.borderColor = 'transparent';
    });
    
    // Reset bot bubbles to light theme
    const botBubbles = document.querySelectorAll('.message.bot .bubble');
    botBubbles.forEach(bubble => {
      bubble.style.background = '#f5f7fa';
      bubble.style.color = '#3730a3';
      bubble.style.border = '1.5px solid #d1d5db';
      bubble.style.boxShadow = '0 4px 16px #818cf822, 0 1.5px 6px #d1d5db22';
    });
    
    // Reset buttons to light theme
    const buttons = document.querySelectorAll('#new-chat, .option-btn, #chat-form button[type="submit"], #file-upload-label, #model-toggle, #save-chat-btn, #delete-chat-btn, #cancel-modal-btn');
    buttons.forEach(btn => {
      btn.style.background = '#818cf8';
      btn.style.color = '#fff';
    });
    
    // Reset logos to light theme
    const logos = document.querySelectorAll('.sidebar-logo, .main-logo, .bot-logo');
    logos.forEach(logo => {
      logo.style.background = '#fff';
      logo.style.border = 'none';
    });
    
    // Reset headers to light theme
    const headers = document.querySelectorAll('.sidebar-header span, .main-header h1');
    headers.forEach(header => {
      header.style.color = '#3730a3';
    });
    
    // Reset dropdowns to light theme
    const dropdowns = document.querySelectorAll('.model-dropdown, .theme-dropdown');
    dropdowns.forEach(dropdown => {
      dropdown.style.background = '#fff';
      dropdown.style.boxShadow = '0 4px 24px #818cf822';
      dropdown.style.border = 'none';
    });
    
    // Reset modal to light theme
    const modalDialog = document.querySelector('.modal-dialog');
    if (modalDialog) {
      modalDialog.style.background = '#fff';
      modalDialog.style.boxShadow = '0 4px 32px #818cf855';
      modalDialog.style.border = 'none';
    }
    
    // Reset input fields to light theme
    const inputs = document.querySelectorAll('#chat-name-input');
    inputs.forEach(input => {
      input.style.background = '#f5f7fa';
      input.style.color = '#3730a3';
      input.style.border = '1.5px solid #818cf8';
    });
    
  } else {
    // Apply dark theme (current theme is already applied via CSS)
    document.body.style.background = '#0f0f23';
    document.body.style.color = '#e0e7ff';
  }
  
  if (themeIcon) themeIcon.textContent = themeIcons[theme] || '🎨';
  localStorage.setItem(THEME_KEY, theme);
}

function getSavedTheme() {
  return localStorage.getItem(THEME_KEY) || 'dark';
}

function getNextTheme(current) {
  const idx = themeOrder.indexOf(current);
  return themeOrder[(idx + 1) % themeOrder.length];
}

// On click, toggle theme
if (themeToggle) {
  console.log('Adding click event listener to theme toggle');
  themeToggle.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    console.log('Theme toggle clicked!');
    const current = getSavedTheme();
    const next = getNextTheme(current);
    console.log('Current theme:', current, 'Next theme:', next);
    setTheme(next);
  });
  
  // Also add a simple test click
  themeToggle.onclick = function(e) {
    console.log('Direct onclick triggered!');
    const current = getSavedTheme();
    const next = getNextTheme(current);
    setTheme(next);
  };
} else {
  console.error('Theme toggle element not found!');
}

// On load, set theme
setTheme(getSavedTheme());
}); 