<template>
  <div class="chat-window">
    <header class="chat-header">
      <span>智能客服</span>
      <span class="status-dot online"></span>
    </header>

    <main class="chat-body" ref="chatBodyRef">
      <p v-if="messages.length === 0" class="welcome">
        你好！我是智能客服，有任何问题都可以问我。
      </p>

      <div
        v-for="(msg, i) in messages"
        :key="i"
        :class="['bubble', msg.role]"
      >
        <div class="content">{{ msg.content }}</div>
        <div class="time">{{ msg.time }}</div>
      </div>

      <!-- 流式打字过程中的临时应答气泡 -->
      <div v-if="streaming && streamingContent" class="bubble assistant streaming">
        <div class="content">
          {{ streamingContent }}
          <span class="cursor">|</span>
        </div>
      </div>

      <div v-if="streaming && !streamingContent" class="bubble assistant thinking">
        <div class="content typing-indicator">
          <span></span><span></span><span></span>
        </div>
      </div>
    </main>

    <footer class="chat-footer">
      <input
        v-model="input"
        placeholder="请输入您的问题..."
        :disabled="streaming"
        @keydown.enter="send"
      />
      <button :disabled="!input.trim() || streaming" @click="send">发送</button>
    </footer>
  </div>
</template>

<script setup>
import { ref, nextTick, onMounted } from 'vue';

const API_BASE = '/api';

const input = ref('');
const messages = ref([]);
const chatBodyRef = ref(null);

// 流式状态
const streaming = ref(false);
const streamingContent = ref('');

// 生成会话 ID（模拟用户身份）
const sessionId = ref(localStorage.getItem('chat_session_id') || '');
if (!sessionId.value) {
  sessionId.value = 'session_' + Math.random().toString(36).slice(2, 10) + '_' + Date.now();
  localStorage.setItem('chat_session_id', sessionId.value);
}

// 进度消息收集
let progressMessages = [];

onMounted(() => {
  scrollToBottom();
});

function scrollToBottom() {
  nextTick(() => {
    if (chatBodyRef.value) {
      chatBodyRef.value.scrollTop = chatBodyRef.value.scrollHeight;
    }
  });
}

function getTime() {
  const d = new Date();
  return String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
}

/**
 * 发送消息 - 使用 SSE 流式接收
 */
async function sendMessage(text) {
  if (!text.trim() || streaming.value) return;

  // 添加用户消息
  const userTime = getTime();
  messages.value.push({ role: 'user', content: text, time: userTime });
  scrollToBottom();

  // 启动流式输出
  streaming.value = true;
  streamingContent.value = '';
  progressMessages = [];

  try {
    const response = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        session_id: sessionId.value,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      // 保留最后一个不完整的行
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));

            switch (data.type) {
              case 'start':
                // 开始处理
                break;

              case 'progress':
                // 处理进度
                progressMessages.push(data.content);
                break;

              case 'chunk':
                // 逐字输出
                streamingContent.value += data.content;
                scrollToBottom();
                break;

              case 'done':
                // 完成 - 将完整内容写入消息列表
                if (data.content) {
                  const finalContent = data.content;
                  // 将进度消息作为"思考过程"折叠或直接跳过，只保留最终回复
                  messages.value.push({
                    role: 'assistant',
                    content: finalContent,
                    time: getTime(),
                    progress: progressMessages.length > 0 ? progressMessages : undefined,
                  });
                  progressMessages = [];
                }
                break;

              case 'error':
                console.error('流式错误:', data.content);
                messages.value.push({
                  role: 'assistant',
                  content: `抱歉，系统出错了: ${data.content}`,
                  time: getTime(),
                });
                break;
            }
          } catch (e) {
            // 解析失败的行，忽略
            console.warn('SSE 解析失败:', line, e);
          }
        }
      }
    }
  } catch (err) {
    console.error('流式请求失败:', err);
    messages.value.push({
      role: 'assistant',
      content: '抱歉，网络连接失败，请稍后重试。',
      time: getTime(),
    });
  } finally {
    streaming.value = false;
    streamingContent.value = '';
    scrollToBottom();
  }
}

function send() {
  sendMessage(input.value);
  input.value = '';
}
</script>

<style scoped>
/* 与之前相同的样式 */
.chat-window {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
  overflow: hidden;
}

.chat-header {
  padding: 14px 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #fff;
  font-weight: 600;
  font-size: 15px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}
.status-dot.online { background: #4ade80; }

.chat-body {
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  background: #f8f9fc;
}

.welcome {
  text-align: center;
  color: #999;
  margin-top: 60px;
  font-size: 14px;
}

.bubble {
  max-width: 82%;
  padding: 10px 14px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.6;
  word-break: break-word;
  white-space: pre-wrap;
}

.bubble.user {
  align-self: flex-end;
  background: #667eea;
  color: #fff;
  border-bottom-right-radius: 4px;
}

.bubble.assistant {
  align-self: flex-start;
  background: #fff;
  color: #333;
  border: 1px solid #e8e8e8;
  border-bottom-left-radius: 4px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.bubble.streaming {
  border-color: #667eea;
}

.bubble.thinking {
  padding: 16px;
}

.typing-indicator {
  display: flex;
  gap: 4px;
  align-items: center;
  justify-content: center;
}
.typing-indicator span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #bbb;
  animation: typing 1.4s ease-in-out infinite;
}
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing {
  0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
  30% { transform: translateY(-6px); opacity: 1; }
}

.content { margin-bottom: 2px; }

.time {
  font-size: 11px;
  opacity: 0.5;
  text-align: right;
  margin-top: 4px;
}

.cursor {
  display: inline-block;
  animation: blink 0.8s step-end infinite;
  color: #667eea;
  font-weight: bold;
}
@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

.bubble.user .cursor { color: rgba(255,255,255,0.7); }

.chat-footer {
  display: flex;
  gap: 8px;
  padding: 12px 16px;
  border-top: 1px solid #eee;
  background: #fff;
}

.chat-footer input {
  flex: 1;
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 10px 14px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
}

.chat-footer input:focus {
  border-color: #667eea;
}

.chat-footer input:disabled {
  background: #f5f5f5;
  cursor: not-allowed;
}

.chat-footer button {
  padding: 10px 22px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #fff;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  cursor: pointer;
  transition: opacity 0.2s;
}

.chat-footer button:hover:not(:disabled) {
  opacity: 0.9;
}

.chat-footer button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>