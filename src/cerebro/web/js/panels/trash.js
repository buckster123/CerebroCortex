/* Trash Panel */
const trashPanel = {
  async load() {
    const el = document.getElementById('trash-body');
    el.innerHTML = '<div class="empty-state"><div class="spinner"></div></div>';
    try {
      const data = await API.listTrash();
      if (!data.memories.length) {
        el.innerHTML = '<div class="empty-state"><div class="icon">☽</div><p>The ash heap is empty.</p></div>';
        return;
      }
      el.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
          <span style="font-size:12px; color:var(--text-muted);">${data.count} memory(s) in trash</span>
          <button class="btn danger small" onclick="trashPanel.purgeAll()">Purge All</button>
        </div>
        <table class="data-table">
          <thead><tr><th>Content</th><th>Type</th><th>Deleted</th><th></th></tr></thead>
          <tbody>${data.memories.map(m => this.renderRow(m)).join('')}</tbody>
        </table>
      `;
    } catch (e) {
      el.innerHTML = `<div class="empty-state">Error: ${e.message}</div>`;
    }
  },

  renderRow(m) {
    return `<tr>
      <td><div style="font-size:12px; max-width:400px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${this.escape(m.content)}</div></td>
      <td><span class="badge badge-${m.type}">${m.type}</span></td>
      <td style="font-size:11px; color:var(--text-muted);">${m.deleted_at ? new Date(m.deleted_at).toLocaleString() : '-'}</td>
      <td>
        <button class="btn small" onclick="trashPanel.restore('${m.id}')">Restore</button>
        <button class="btn danger small" onclick="trashPanel.purge('${m.id}')">Purge</button>
      </td>
    </tr>`;
  },

  async restore(id) {
    try { await API.restoreTrash(id); app.toast('Restored', 'success'); this.load(); app.loadStats(); }
    catch (e) { app.toast(e.message, 'error'); }
  },

  async purge(id) {
    if (!confirm('Permanently erase this memory?')) return;
    try { await API.purgeTrash(id); app.toast('Purged', 'info'); this.load(); app.loadStats(); }
    catch (e) { app.toast(e.message, 'error'); }
  },

  async purgeAll() {
    if (!confirm('Purge ALL trashed memories?')) return;
    try { const res = await API.purgeAllTrash(); app.toast(`Purged ${res.purged} memories`, 'info'); this.load(); app.loadStats(); }
    catch (e) { app.toast(e.message, 'error'); }
  },

  escape(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; },
};