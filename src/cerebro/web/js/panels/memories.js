/* Memories Panel */
const memoriesPanel = {
  selected: new Set(),

  async load() {
    const el = document.getElementById('memories-body');
    el.innerHTML = `
      <div class="search-bar">
        <input type="text" id="mem-search" placeholder="Search memories..." onkeydown="if(event.key==='Enter')memoriesPanel.search()">
        <button class="btn primary" onclick="memoriesPanel.search()">☉ Search</button>
      </div>
      <div id="mem-results"></div>
    `;
    await this.search();
  },

  async search() {
    const query = document.getElementById('mem-search')?.value || '';
    const el = document.getElementById('mem-results');
    el.innerHTML = '<div class="empty-state"><div class="spinner"></div></div>';
    try {
      const results = await API.recall({ query, top_k: 50, explain: false });
      if (!results.length) {
        el.innerHTML = '<div class="empty-state"><div class="icon">☽</div><p>No memories found.</p></div>';
        return;
      }
      el.innerHTML = `<table class="data-table">
        <thead><tr>
          <th style="width:30px"><input type="checkbox" onchange="memoriesPanel.toggleAll(this)"></th>
          <th>Content</th><th>Type</th><th>Salience</th><th>Tags</th><th></th>
        </tr></thead>
        <tbody>${results.map(([n, s]) => this.renderRow(n, s)).join('')}</tbody>
      </table>
      ${this.selected.size > 0 ? `<div style="margin-top:12px; display:flex; gap:8px;">
        <button class="btn danger small" onclick="memoriesPanel.bulkDelete()">Delete Selected</button>
        <button class="btn small" onclick="memoriesPanel.bulkShare('private')">Make Private</button>
        <button class="btn small" onclick="memoriesPanel.bulkShare('shared')">Make Shared</button>
      </div>` : ''}`;
    } catch (e) {
      el.innerHTML = `<div class="empty-state">Error: ${e.message}</div>`;
    }
  },

  renderRow(node, score) {
    const type = node.metadata.memory_type;
    const badge = `badge-${type}`;
    const checked = this.selected.has(node.id) ? 'checked' : '';
    return `<tr>
      <td><input type="checkbox" ${checked} onchange="memoriesPanel.toggle('${node.id}')"></td>
      <td><div style="font-size:12px; line-height:1.4; max-width:400px; overflow:hidden; text-overflow:ellipsis; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical;">${this.escape(node.content)}</div></td>
      <td><span class="badge ${badge}">${type}</span></td>
      <td>${node.metadata.salience.toFixed(2)}</td>
      <td>${node.metadata.tags.map(t => `<span class="tag">${this.escape(t)}</span>`).join(' ')}</td>
      <td><button class="btn small" onclick="memoriesPanel.view('${node.id}')">☼</button></td>
    </tr>`;
  },

  toggle(id) {
    if (this.selected.has(id)) this.selected.delete(id);
    else this.selected.add(id);
    this.search();
  },

  toggleAll(cb) {
    if (cb.checked) document.querySelectorAll('#mem-results input[type=checkbox]').forEach(c => { if (c !== cb) { const id = c.getAttribute('onchange')?.match(/'([^']+)'/)?.[1]; if (id) this.selected.add(id); } });
    else this.selected.clear();
    this.search();
  },

  async view(id) {
    try {
      const m = await API.getMemory(id);
      app.showModal('Memory', `<pre style="white-space:pre-wrap; font-size:12px; line-height:1.5;">${this.escape(m.content)}</pre>
        <div style="margin-top:12px; display:flex; gap:8px; flex-wrap:wrap;">
          <span class="badge badge-${m.type}">${m.type}</span>
          <span class="tag">salience: ${m.salience}</span>
          ${m.tags.map(t => `<span class="tag">${this.escape(t)}</span>`).join('')}
        </div>`,
        `<button class="btn" onclick="app.closeModal()">Close</button>
        <button class="btn danger" onclick="memoriesPanel.delete('${id}')">Delete</button>`
      );
    } catch (e) { app.toast(e.message, 'error'); }
  },

  async delete(id) {
    try { await API.deleteMemory(id); app.closeModal(); app.toast('Memory deleted', 'success'); this.search(); }
    catch (e) { app.toast(e.message, 'error'); }
  },

  async bulkDelete() {
    if (!confirm('Soft-delete selected memories?')) return;
    try { await API.bulkDelete({ memory_ids: [...this.selected], soft: true }); this.selected.clear(); app.toast('Deleted', 'success'); this.search(); }
    catch (e) { app.toast(e.message, 'error'); }
  },

  async bulkShare(vis) {
    try { await API.bulkVisibility({ memory_ids: [...this.selected], visibility: vis }); app.toast(`Made ${vis}`, 'success'); this.search(); }
    catch (e) { app.toast(e.message, 'error'); }
  },

  escape(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  },
};
