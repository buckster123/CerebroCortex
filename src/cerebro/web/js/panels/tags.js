/* Tags Panel */
const tagsPanel = {
  async load() {
    const el = document.getElementById('tags-body');
    el.innerHTML = '<div class="empty-state"><div class="spinner"></div></div>';
    try {
      const data = await API.listTags();
      if (!data.tags.length) {
        el.innerHTML = '<div class="empty-state"><div class="icon">&#x2735;</div><p>No sigils found.</p></div>';
        return;
      }
      el.innerHTML = `
        <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:20px;">
          ${data.tags.map(t => `<span class="tag" style="cursor:pointer; font-size:13px; padding:6px 14px;" onclick="tagsPanel.actions('${this.escape(t.name)}')">${this.escape(t.name)} <span style="opacity:0.5; font-size:11px;">${t.count}</span></span>`).join('')}
        </div>
      `;
    } catch (e) {
      el.innerHTML = `<div class="empty-state">Error: ${e.message}</div>`;
    }
  },

  actions(tag) {
    app.showModal('Sigil: ' + tag, '', `
      <button class="btn" onclick="app.closeModal()">Cancel</button>
      <button class="btn" onclick="tagsPanel.renamePrompt('${tag}')">Rename</button>
      <button class="btn danger" onclick="tagsPanel.delete('${tag}')">Delete</button>
    `);
  },

  async renamePrompt(oldTag) {
    const newTag = prompt('Rename "' + oldTag + '" to:', oldTag);
    if (!newTag || newTag === oldTag) { app.closeModal(); return; }
    try { await API.renameTag({ old_tag: oldTag, new_tag: newTag }); app.closeModal(); app.toast('Renamed', 'success'); this.load(); }
    catch (e) { app.toast(e.message, 'error'); }
  },

  async delete(tag) {
    if (!confirm('Remove tag "' + tag + '" from all memories?')) return;
    try { await API.deleteTag(tag); app.closeModal(); app.toast('Deleted', 'success'); this.load(); }
    catch (e) { app.toast(e.message, 'error'); }
  },

  escape(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; },
};
