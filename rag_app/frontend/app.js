/**
 * SmartRAG — Frontend Application v2
 * Auth + Chat History Sidebar + mem0 Memory Panel
 */

class SmartRAG {
  constructor() {
    this.API_BASE = window.location.port === '3000'
      ? `${window.location.protocol}//${window.location.hostname}:8003`
      : window.location.origin;

    this.token = localStorage.getItem('token');
    this.user = JSON.parse(localStorage.getItem('user') || 'null');
    this.sessionId = localStorage.getItem('session_id');
    this.conversationId = null;
    this.isGenerating = false;

    // Auth guard
    if (!this.token || !this.user) {
      window.location.href = '/login';
      return;
    }

    this._init();
  }

  _authHeaders() {
    return { 'Content-Type': 'application/json', 'Authorization': `Bearer ${this.token}` };
  }

  async _init() {
    this._renderUserInfo();
    this._bindEvents();
    this._setupDrop();
    this._autoResizeTextarea();
    this._checkHealth();
    await this._loadConversations();
  }

  // ── User info ──────────────────────────────────────────────

  _renderUserInfo() {
    const name = this.user.full_name || this.user.username;
    document.getElementById('user-name').textContent = name;
    document.getElementById('user-avatar').textContent = name.charAt(0).toUpperCase();
  }

  // ── Event binding ──────────────────────────────────────────

  _bindEvents() {
    document.getElementById('chat-form').addEventListener('submit', e => this._sendMessage(e));
    document.getElementById('upload-form').addEventListener('submit', e => this._uploadFile(e));
    document.getElementById('file-input').addEventListener('change', e => this._onFileSelected(e));
    document.getElementById('browse-btn').addEventListener('click', () => document.getElementById('file-input').click());
    document.getElementById('drop-zone').addEventListener('click', () => document.getElementById('file-input').click());
    document.getElementById('new-chat-btn').addEventListener('click', () => this._newChat());
    document.getElementById('logout-btn').addEventListener('click', () => this._logout());
    document.getElementById('memory-btn').addEventListener('click', () => this._openMemoryPanel());
    document.getElementById('memory-close-btn').addEventListener('click', () => this._closeMemoryPanel());
    document.getElementById('memory-overlay').addEventListener('click', () => this._closeMemoryPanel());
    document.getElementById('clear-all-mem-btn').addEventListener('click', () => this._clearAllMemories());

    const ta = document.getElementById('question-input');
    ta.addEventListener('input', () => {
      this._autoResizeTextarea();
      document.getElementById('send-btn').disabled = !ta.value.trim() || this.isGenerating;
    });
    ta.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!this.isGenerating && ta.value.trim()) this._sendMessage(e);
      }
    });

    document.querySelectorAll('.qp-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        ta.value = btn.dataset.text;
        ta.focus();
        document.getElementById('send-btn').disabled = false;
        this._autoResizeTextarea();
      });
    });
  }

  _autoResizeTextarea() {
    const ta = document.getElementById('question-input');
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 120) + 'px';
  }

  // ── Health check ───────────────────────────────────────────

  async _checkHealth() {
    try {
      const r = await fetch(`${this.API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
      const dot = document.getElementById('status-dot');
      const label = document.getElementById('status-label');
      if (r.ok) { dot.classList.remove('error'); label.textContent = 'System Ready'; }
      else { dot.classList.add('error'); label.textContent = 'Backend Error'; }
    } catch {
      document.getElementById('status-dot').classList.add('error');
      document.getElementById('status-label').textContent = 'Backend Offline';
    }
  }

  // ── Conversations ──────────────────────────────────────────

  async _loadConversations() {
    try {
      const resp = await fetch(`${this.API_BASE}/conversations`, { headers: this._authHeaders() });
      if (resp.status === 401) { this._handleUnauth(); return; }
      if (!resp.ok) return;
      const data = await resp.json();
      this._renderConversationList(data.conversations || []);
    } catch (e) {
      console.error('Failed to load conversations', e);
    }
  }

  _renderConversationList(conversations) {
    const list = document.getElementById('conv-list');
    const empty = document.getElementById('conv-empty');

    if (!conversations.length) {
      list.innerHTML = '<div class="conv-empty" id="conv-empty">No conversations yet</div>';
      return;
    }

    // Group by date
    const today = new Date(); today.setHours(0,0,0,0);
    const yesterday = new Date(today); yesterday.setDate(yesterday.getDate() - 1);

    const groups = { Today: [], Yesterday: [], Earlier: [] };
    conversations.forEach(c => {
      const d = new Date(c.updated_at); d.setHours(0,0,0,0);
      if (d >= today) groups.Today.push(c);
      else if (d >= yesterday) groups.Yesterday.push(c);
      else groups.Earlier.push(c);
    });

    let html = '';
    for (const [label, items] of Object.entries(groups)) {
      if (!items.length) continue;
      html += `<div class="conv-group-label">${label}</div>`;
      items.forEach(c => {
        const active = c.id === this.conversationId ? ' active' : '';
        const title = this._escHtml(c.title || `Conversation #${c.id}`);
        html += `
          <div class="conv-item${active}" data-id="${c.id}">
            <div class="conv-item-title">${title}</div>
            <button class="conv-delete-btn" data-id="${c.id}" title="Delete">✕</button>
          </div>`;
      });
    }
    list.innerHTML = html;

    list.querySelectorAll('.conv-item').forEach(el => {
      el.addEventListener('click', e => {
        if (e.target.classList.contains('conv-delete-btn')) return;
        this._loadConversation(parseInt(el.dataset.id));
      });
    });
    list.querySelectorAll('.conv-delete-btn').forEach(btn => {
      btn.addEventListener('click', e => {
        e.stopPropagation();
        this._deleteConversation(parseInt(btn.dataset.id));
      });
    });
  }

  async _loadConversation(convId) {
    try {
      const resp = await fetch(`${this.API_BASE}/conversations/${convId}/messages`, {
        headers: this._authHeaders()
      });
      if (!resp.ok) { this._toast('Failed to load conversation', 'error'); return; }
      const data = await resp.json();

      this.conversationId = convId;
      const area = document.getElementById('messages-area');
      const ws = document.getElementById('welcome-screen');
      if (ws) ws.remove();
      area.innerHTML = '';

      data.messages.forEach(msg => this._appendMessage({ role: msg.role, content: msg.content }));

      const title = data.title || `Conversation #${convId}`;
      document.getElementById('topbar-title').textContent = title;
      document.getElementById('conv-sub').textContent = `${data.total_messages} messages`;

      // Highlight active
      document.querySelectorAll('.conv-item').forEach(el => {
        el.classList.toggle('active', parseInt(el.dataset.id) === convId);
      });
    } catch (e) {
      this._toast('Error loading conversation', 'error');
    }
  }

  async _deleteConversation(convId) {
    try {
      const resp = await fetch(`${this.API_BASE}/conversations/${convId}`, {
        method: 'DELETE',
        headers: this._authHeaders()
      });
      if (resp.ok) {
        if (this.conversationId === convId) this._newChat();
        await this._loadConversations();
        this._toast('Conversation deleted', 'info');
      }
    } catch { this._toast('Failed to delete', 'error'); }
  }

  _newChat() {
    this.conversationId = null;
    const area = document.getElementById('messages-area');
    area.innerHTML = `
      <div class="welcome-screen" id="welcome-screen">
        <div class="welcome-glow"></div>
        <div class="welcome-icon-wrap">
          <svg viewBox="0 0 64 64" fill="none" width="64">
            <circle cx="32" cy="32" r="32" fill="url(#wg3)"/>
            <path d="M22 42L32 20L42 42" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="32" cy="36" r="4.5" fill="#fff"/>
            <defs><linearGradient id="wg3" x1="0" y1="0" x2="64" y2="64">
              <stop stop-color="#7C3AED"/><stop offset="1" stop-color="#4F46E5"/>
            </linearGradient></defs>
          </svg>
        </div>
        <h2 class="welcome-title">New Conversation</h2>
        <p class="welcome-sub">Ask anything about your documents or say "Remember that…" to save a memory.</p>
      </div>`;
    document.getElementById('topbar-title').textContent = 'Chat Assistant';
    document.getElementById('conv-sub').textContent = 'Start a new conversation';
    document.getElementById('question-input').value = '';
    document.getElementById('send-btn').disabled = true;
    document.querySelectorAll('.conv-item').forEach(el => el.classList.remove('active'));
  }

  // ── Send message ───────────────────────────────────────────

  async _sendMessage(e) {
    e.preventDefault();
    if (this.isGenerating) return;

    const ta = document.getElementById('question-input');
    const question = ta.value.trim();
    if (!question) return;

    ta.value = '';
    this._autoResizeTextarea();
    document.getElementById('send-btn').disabled = true;

    const ws = document.getElementById('welcome-screen');
    if (ws) ws.remove();

    this._appendMessage({ role: 'user', content: question });

    const typingId = this._showTyping();
    this.isGenerating = true;
    this._setGenerating(true);

    try {
      const resp = await fetch(`${this.API_BASE}/chat`, {
        method: 'POST',
        headers: this._authHeaders(),
        body: JSON.stringify({ question, conversation_id: this.conversationId }),
      });

      this._removeTyping(typingId);

      if (resp.status === 401) { this._handleUnauth(); return; }

      if (resp.ok) {
        const data = await resp.json();
        this.conversationId = data.conversation_id;

        if (data.status === 'awaiting_human_approval') {
          this._appendHITLMessage(data);
        } else {
          this._appendMessage({
            role: 'assistant',
            content: data.answer,
            status: data.status,
            responseType: data.response_type,
            sources: data.sources,
            stats: data.retrieval_stats,
            toolResult: data.tool_result,
          });
        }
        // Refresh sidebar
        await this._loadConversations();
        document.querySelectorAll('.conv-item').forEach(el => {
          if (parseInt(el.dataset.id) === this.conversationId) {
            const t = el.querySelector('.conv-item-title').textContent;
            document.getElementById('topbar-title').textContent = t;
          }
        });
      } else {
        const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
        this._appendMessage({ role: 'assistant', content: `Error: ${err.detail}`, responseType: 'error', isError: true });
        this._toast(err.detail || 'Request failed', 'error');
      }
    } catch {
      this._removeTyping(typingId);
      this._appendMessage({ role: 'assistant', content: 'Connection error. Please check the server.', responseType: 'error', isError: true });
      this._toast('Connection error', 'error');
    } finally {
      this.isGenerating = false;
      this._setGenerating(false);
      const t = document.getElementById('question-input');
      if (t.value.trim()) document.getElementById('send-btn').disabled = false;
    }
  }

  _setGenerating(on) {
    const btn = document.getElementById('send-btn');
    const icon = document.getElementById('send-icon');
    const spinner = document.getElementById('send-spinner');
    const ta = document.getElementById('question-input');
    if (on) {
      btn.disabled = true; icon.classList.add('hidden'); spinner.classList.remove('hidden');
      ta.disabled = true; ta.placeholder = 'Processing through 9-stage RAG pipeline…';
    } else {
      icon.classList.remove('hidden'); spinner.classList.add('hidden');
      ta.disabled = false; ta.placeholder = 'Ask anything — documents, general knowledge, or say "Remember that…"';
      ta.focus();
    }
  }

  // ── HITL approval message ──────────────────────────────────

  _appendHITLMessage(data) {
    const area = document.getElementById('messages-area');
    const div = document.createElement('div');
    div.className = 'msg assistant hitl-msg';
    div.innerHTML = `
      <div class="msg-avatar">🤖</div>
      <div class="msg-body">
        <div class="msg-bubble">
          <div class="msg-text">${this._formatText(data.answer)}</div>
          <div class="hitl-actions">
            <button class="hitl-btn hitl-web" data-conv="${data.conversation_id}" data-action="search_web">
              🌐 Search the Web
            </button>
            <button class="hitl-btn hitl-docs" data-conv="${data.conversation_id}" data-action="documents_only">
              📄 Stay with Documents
            </button>
          </div>
        </div>
        <div class="msg-meta">
          <span class="response-badge badge-hitl">⏳ Awaiting Your Decision</span>
          <span>${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
      </div>`;
    div.querySelectorAll('.hitl-btn').forEach(btn => {
      btn.addEventListener('click', () => this._handleHITLAction(
        parseInt(btn.dataset.conv), btn.dataset.action, div
      ));
    });
    area.appendChild(div);
    area.scrollTop = area.scrollHeight;
  }

  async _handleHITLAction(conversationId, action, msgDiv) {
    msgDiv.querySelectorAll('.hitl-btn').forEach(b => b.disabled = true);
    const typingId = this._showTyping();
    this._setGenerating(true);
    this.isGenerating = true;
    try {
      const resp = await fetch(`${this.API_BASE}/chat/hitl-action`, {
        method: 'POST',
        headers: this._authHeaders(),
        body: JSON.stringify({ conversation_id: conversationId, action }),
      });
      this._removeTyping(typingId);
      if (resp.status === 401) { this._handleUnauth(); return; }
      if (resp.ok) {
        const data = await resp.json();
        this._appendMessage({
          role: 'assistant',
          content: data.answer,
          status: data.status,
          responseType: data.response_type,
          sources: data.sources,
          stats: data.retrieval_stats,
        });
        await this._loadConversations();
      } else {
        const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
        this._appendMessage({ role: 'assistant', content: `Error: ${err.detail}`, responseType: 'error', isError: true });
        this._toast(err.detail || 'Action failed', 'error');
      }
    } catch {
      this._removeTyping(typingId);
      this._appendMessage({ role: 'assistant', content: 'Connection error.', responseType: 'error', isError: true });
      this._toast('Connection error', 'error');
    } finally {
      this.isGenerating = false;
      this._setGenerating(false);
    }
  }

  // ── Message rendering ──────────────────────────────────────

  _appendMessage(msg) {
    const area = document.getElementById('messages-area');
    const div = document.createElement('div');
    div.className = `msg ${msg.role}${msg.isError ? ' error' : ''}`;

    const avatar = msg.role === 'user' ? '👤' : '🤖';
    div.innerHTML = `
      <div class="msg-avatar">${avatar}</div>
      <div class="msg-body">
        ${this._renderBubbleContent(msg)}
        <div class="msg-meta">
          ${this._badgeHtml(msg.responseType)}
          <span>${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
      </div>`;

    const statsToggle = div.querySelector('.stats-toggle');
    if (statsToggle) {
      const statsPanel = div.querySelector('.stats-panel');
      statsToggle.addEventListener('click', () => {
        statsPanel.classList.toggle('open');
        statsToggle.querySelector('.toggle-arrow').textContent = statsPanel.classList.contains('open') ? '▲' : '▼';
      });
    }

    area.appendChild(div);
    area.scrollTop = area.scrollHeight;
  }

  _renderBubbleContent(msg) {
    const text = `<div class="msg-bubble"><div class="msg-text">${this._formatText(msg.content)}</div></div>`;
    const sources = msg.sources && msg.sources.length ? this._sourcesHtml(msg.sources) : '';
    const confidence = msg.stats && msg.stats.confidence != null ? this._confidenceHtml(msg.stats.confidence) : '';
    const stats = msg.stats ? this._statsHtml(msg.stats) : '';
    return text + confidence + sources + stats;
  }

  _formatText(raw) {
    return raw
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/^#{1,3}\s+(.*)/gm, '<strong>$1</strong>')
      .replace(/^[-*]\s+(.*)/gm, '<li>$1</li>')
      .replace(/(<li>.*<\/li>\n?)+/g, m => `<ul>${m}</ul>`)
      .replace(/\n{2,}/g, '</p><p>')
      .replace(/\n/g, '<br>');
  }

  _badgeHtml(type) {
    const map = {
      document_based:           ['badge-doc',  '📄 Document'],
      general_ai:               ['badge-ai',   '🤖 AI'],
      tool_result:              ['badge-tool', '🔧 Tool'],
      error:                    ['badge-err',  '❌ Error'],
      web_fallback:             ['badge-ai',   '🌐 Fallback'],
      web_search:               ['badge-web',  '🌐 Web Search'],
      memory:                   ['badge-mem',  '🧠 Memory'],
      memory_saved:             ['badge-mem',  '🧠 Memory'],
      hitl_pending:             ['badge-hitl', '⏳ Awaiting Approval'],
      documents_only:           ['badge-doc',  '📄 Documents Only'],
      web_no_results:           ['badge-ai',   '🌐 No Results'],
    };
    const [cls, label] = map[type] || ['badge-ai', '💬 Response'];
    return `<span class="response-badge ${cls}">${label}</span>`;
  }

  _sourcesHtml(sources) {
    const rows = sources.map((s, i) => `
      <div class="source-item">
        <span class="source-name">${s.filename} · chunk ${s.chunk_index + 1}</span>
        <span class="source-rank">#${i + 1}</span>
        <div class="source-scores">
          ${s.rrf_score ? `<span class="score-tag">RRF ${s.rrf_score}</span>` : ''}
          ${s.similarity > 0 ? `<span class="score-tag">Sim ${s.similarity}</span>` : ''}
        </div>
      </div>`).join('');
    return `<div class="msg-sources"><div class="sources-title">📚 Sources</div>${rows}</div>`;
  }

  _confidenceHtml(conf) {
    const pct = Math.round(conf * 100);
    return `<div class="confidence-bar-wrap">
      <div class="confidence-bar-bg"><div class="confidence-bar-fill" style="width:${pct}%"></div></div>
      <span class="confidence-pct">${pct}%</span></div>`;
  }

  _statsHtml(stats) {
    const rows = [
      ['Query type', stats.query_type || '—'],
      ['Variants', stats.query_variants || '—'],
      ['Vector candidates', stats.vector_candidates ?? '—'],
      ['BM25 candidates', stats.bm25_candidates ?? '—'],
      ['After RRF fusion', stats.after_rrf ?? '—'],
      ['After MMR (top 8)', stats.after_mmr ?? '—'],
      ['After rerank (top 5)', stats.after_rerank ?? '—'],
      ['Context chunks', stats.context_chunks ?? '—'],
      ['Best score', stats.best_score ?? '—'],
      ['Passed gate', stats.passed_gate != null ? (stats.passed_gate ? 'Yes ✓' : 'No (fallback)') : '—'],
    ].map(([k, v]) => `<div class="stat-row"><span class="stat-key">${k}</span><span class="stat-val">${v}</span></div>`).join('');
    return `<div class="msg-stats">
      <button class="stats-toggle"><span class="toggle-arrow">▼</span> Pipeline stats</button>
      <div class="stats-panel">${rows}</div></div>`;
  }

  // ── Typing indicator ───────────────────────────────────────

  _showTyping() {
    const id = 'typing-' + Date.now();
    const area = document.getElementById('messages-area');
    const div = document.createElement('div');
    div.className = 'msg assistant typing-msg';
    div.id = id;
    div.innerHTML = `
      <div class="msg-avatar">🤖</div>
      <div class="msg-body"><div class="msg-bubble">
        <div class="typing-label" id="${id}-label">Normalizing query…</div>
        <div class="typing-dots"><span></span><span></span><span></span></div>
      </div></div>`;
    area.appendChild(div);
    area.scrollTop = area.scrollHeight;

    const stages = ['Normalizing query…','Rewriting variants…','Hybrid retrieval…','RRF fusion…','MMR diversity…','Cross-encoder rerank…','Neighbor expansion…','Context compression…','LLM generation…'];
    let i = 0;
    const interval = setInterval(() => {
      const label = document.getElementById(`${id}-label`);
      if (label && i < stages.length - 1) { i++; label.textContent = stages[i]; }
      else clearInterval(interval);
    }, 1800);
    div._interval = interval;
    return id;
  }

  _removeTyping(id) {
    const el = document.getElementById(id);
    if (el) { if (el._interval) clearInterval(el._interval); el.remove(); }
  }

  // ── Memory Panel ───────────────────────────────────────────

  async _openMemoryPanel() {
    document.getElementById('memory-panel').classList.remove('hidden');
    document.getElementById('memory-overlay').classList.remove('hidden');
    await this._loadMemories();
  }

  _closeMemoryPanel() {
    document.getElementById('memory-panel').classList.add('hidden');
    document.getElementById('memory-overlay').classList.add('hidden');
  }

  async _loadMemories() {
    const list = document.getElementById('memory-list');
    list.innerHTML = '<div class="memory-loading">Loading memories…</div>';
    try {
      const resp = await fetch(`${this.API_BASE}/memory/`, { headers: this._authHeaders() });
      if (!resp.ok) { list.innerHTML = '<div class="memory-empty">Failed to load memories.</div>'; return; }
      const data = await resp.json();
      const memories = data.memories || [];
      if (!memories.length) {
        list.innerHTML = '<div class="memory-empty">No memories saved yet.<br><small>Memories are captured automatically from your conversations.</small></div>';
        return;
      }
      list.innerHTML = memories.map(m => {
        const text = this._escHtml(m.memory || m.text || JSON.stringify(m));
        const id = m.id || '';
        return `<div class="memory-item" data-id="${id}">
          <div class="memory-text">${text}</div>
          <button class="memory-delete-btn" data-id="${id}" title="Delete memory">🗑</button>
        </div>`;
      }).join('');

      list.querySelectorAll('.memory-delete-btn').forEach(btn => {
        btn.addEventListener('click', () => this._deleteMemory(btn.dataset.id));
      });
    } catch { list.innerHTML = '<div class="memory-empty">Error loading memories.</div>'; }
  }

  async _deleteMemory(memoryId) {
    try {
      const resp = await fetch(`${this.API_BASE}/memory/${memoryId}`, {
        method: 'DELETE',
        headers: this._authHeaders()
      });
      if (resp.ok) { this._toast('Memory deleted', 'info'); await this._loadMemories(); }
      else this._toast('Failed to delete memory', 'error');
    } catch { this._toast('Error deleting memory', 'error'); }
  }

  async _clearAllMemories() {
    if (!confirm('Delete all your memories? This cannot be undone.')) return;
    try {
      const resp = await fetch(`${this.API_BASE}/memory/`, {
        method: 'DELETE',
        headers: this._authHeaders()
      });
      if (resp.ok) { this._toast('All memories cleared', 'info'); await this._loadMemories(); }
      else this._toast('Failed to clear memories', 'error');
    } catch { this._toast('Error clearing memories', 'error'); }
  }

  // ── Auth ───────────────────────────────────────────────────

  async _logout() {
    try {
      await fetch(`${this.API_BASE}/auth/logout?session_id=${this.sessionId}`, {
        method: 'POST',
        headers: this._authHeaders()
      });
    } catch {}
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    localStorage.removeItem('session_id');
    window.location.href = '/login';
  }

  _handleUnauth() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    localStorage.removeItem('session_id');
    window.location.href = '/login';
  }

  // ── File upload ────────────────────────────────────────────

  _setupDrop() {
    const zone = document.getElementById('drop-zone');
    ['dragenter','dragover','dragleave','drop'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); }));
    ['dragenter','dragover'].forEach(ev => zone.addEventListener(ev, () => zone.classList.add('drag-over')));
    ['dragleave','drop'].forEach(ev => zone.addEventListener(ev, () => zone.classList.remove('drag-over')));
    zone.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (f) this._setFile(f); });
  }

  _onFileSelected(e) { const f = e.target.files[0]; if (f) this._setFile(f); }

  _setFile(file) {
    if (file.type !== 'application/pdf') { this._toast('Please select a PDF file', 'error'); return; }
    if (file.size > 52428800) { this._toast('File too large (max 50 MB)', 'error'); return; }
    document.getElementById('drop-primary').textContent = file.name;
    document.getElementById('upload-btn').disabled = false;
    const dt = new DataTransfer(); dt.items.add(file);
    document.getElementById('file-input').files = dt.files;
  }

  async _uploadFile(e) {
    e.preventDefault();
    const fi = document.getElementById('file-input');
    if (!fi.files.length) { this._toast('Select a PDF first', 'warning'); return; }

    const file = fi.files[0];
    const formData = new FormData();
    formData.append('file', file);

    const status = document.getElementById('upload-status');
    const btn = document.getElementById('upload-btn');
    const spinner = document.getElementById('upload-spinner');
    const btnText = document.getElementById('upload-btn-text');

    btn.disabled = true;
    spinner.classList.remove('hidden');
    btnText.textContent = 'Processing…';
    status.className = 'upload-status processing';
    status.textContent = `Uploading ${file.name}…`;
    status.classList.remove('hidden');

    try {
      const headers = { 'Authorization': `Bearer ${this.token}` };
      const resp = await fetch(`${this.API_BASE}/upload`, { method: 'POST', headers, body: formData });
      spinner.classList.add('hidden');
      btnText.textContent = 'Upload & Process';

      if (resp.ok) {
        const result = await resp.json();
        status.className = 'upload-status success';
        status.innerHTML = `✅ <strong>${file.name}</strong> — ${result.chunks || '?'} chunks indexed`;
        fi.value = '';
        document.getElementById('drop-primary').textContent = 'Drop PDF here';
        this._toast('Document processed!', 'success');
        const ws = document.getElementById('welcome-screen');
        if (ws) ws.remove();
        this._appendMessage({ role: 'assistant', content: `✅ **${file.name}** uploaded and indexed. You can now ask questions about it!`, responseType: 'general_ai' });
      } else {
        const err = await resp.json().catch(() => ({ detail: 'Upload failed' }));
        status.className = 'upload-status error';
        status.textContent = `❌ ${err.detail || 'Upload failed'}`;
        this._toast(err.detail || 'Upload failed', 'error');
      }
    } catch {
      spinner.classList.add('hidden');
      btnText.textContent = 'Upload & Process';
      status.className = 'upload-status error';
      status.textContent = '❌ Connection error during upload';
      this._toast('Upload failed — connection error', 'error');
    } finally { btn.disabled = false; }
  }

  // ── Helpers ────────────────────────────────────────────────

  _escHtml(str) {
    return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  _toast(msg, type = 'info') {
    const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };
    const container = document.getElementById('toasts');
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.innerHTML = `<span>${icons[type]}</span><span>${msg}</span>`;
    container.appendChild(t);
    t.addEventListener('click', () => t.remove());
    setTimeout(() => { if (t.parentNode) t.remove(); }, 4000);
  }
}

document.addEventListener('DOMContentLoaded', () => { window.app = new SmartRAG(); });
