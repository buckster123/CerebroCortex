/* Threads Panel */
const threadsPanel = {
  async load() {
    const el = document.getElementById('threads-body');
    el.innerHTML = '<div class="empty-state"><div class="spinner"></div></div>';
    try {
      const data = await API.listThreads();
      if (!data.threads.length) {
        el.innerHTML = '<div class="empty-state"><div class="icon">&#x265b;</div><p>No threads found.</p></div>';
        return;
      }
      el.innerHTML = `<table class="data-table">
        <thead><tr><th>Thread ID</th><th>Memories</th><th></th></tr></thead>
        <tbody>${data.threads.map(t => this.renderRow(t)).join('')}</tbody>
      </table>`;
    } catch (e) {
      el.innerHTML = `<div class="empty-state">Error: ${e.message}</div>`;
    }
  },

  renderRow(t) {
    return `<tr>
      <td style="font-family:var(--font-mono); font-size:12px;">${this.escape(t.thread_id || t.id || 'N/A')}</td>
      <td>${t.memory_count ?? '-'}</td>
      <td>
        <button class="btn small" onclick="threadsPanel.view('${this.escape(t.thread_id || t.id || '')}')">View</button>
        <button class="btn danger small" onclick="threadsPanel.prune('${this.escape(t.thread_id || t.id || '')}')">Prune</button>
      </td>
    </tr>`;
  },

  async view(id) {
    try {
      const data = await API.getThreadMemories(id);
      app.showModal('Thread: ' + id, `<div style="max-height:400px; overflow-y:auto;">${data.memories.map(m => `<div style="padding:8px 0; border-bottom:1px solid var(--border);"><span class="badge badge-${m.type}">${m.type}</span> <span style="font-size:12px;">${this.escape(m.content.substring(0, 120))}${m.content.length > 120 ? '...' : ''}</span></div>`).join('')}</div>`);
    } catch (e) { app.toast(e.message, 'error'); }
  },

  async prune(id) {
    if (!confirm('Soft-delete all memories in this thread?')) return;
    try { await API.pruneThread(id); app.toast('Thread pruned', 'info'); this.load(); app.loadStats(); }
    catch (e) { app.toast(e.message, 'error'); }
  },

  escape(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; },
};
