/* Store Memory Panel */
const storePanel = {
  load() {
    document.getElementById('store-body').innerHTML = `
      <div style="max-width:600px;">
        <div style="margin-bottom:16px;">
          <label style="display:block; font-size:11px; color:var(--text-muted); margin-bottom:6px; text-transform:uppercase; letter-spacing:1px;">Content</label>
          <textarea id="store-content" rows="6" placeholder="What shall be remembered?"></textarea>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:16px;">
          <div>
            <label style="display:block; font-size:11px; color:var(--text-muted); margin-bottom:6px; text-transform:uppercase; letter-spacing:1px;">Type</label>
            <select id="store-type">
              <option value="semantic">Semantic</option>
              <option value="episodic">Episodic</option>
              <option value="procedural">Procedural</option>
              <option value="affective">Affective</option>
              <option value="prospective">Prospective</option>
              <option value="schematic">Schematic</option>
            </select>
          </div>
          <div>
            <label style="display:block; font-size:11px; color:var(--text-muted); margin-bottom:6px; text-transform:uppercase; letter-spacing:1px;">Visibility</label>
            <select id="store-visibility">
              <option value="shared">Shared</option>
              <option value="private">Private</option>
              <option value="thread">Thread</option>
            </select>
          </div>
        </div>
        <div style="margin-bottom:16px;">
          <label style="display:block; font-size:11px; color:var(--text-muted); margin-bottom:6px; text-transform:uppercase; letter-spacing:1px;">Tags (comma-separated)</label>
          <input type="text" id="store-tags" placeholder="e.g. project, research">
        </div>
        <div style="margin-bottom:16px;">
          <label style="display:block; font-size:11px; color:var(--text-muted); margin-bottom:6px; text-transform:uppercase; letter-spacing:1px;">Salience (0–1)</label>
          <input type="number" id="store-salience" min="0" max="1" step="0.1" value="0.5">
        </div>
        <button class="btn primary" onclick="storePanel.submit()">⚛ Store Memory</button>
      </div>
    `;
  },

  async submit() {
    const content = document.getElementById('store-content').value.trim();
    if (!content) { app.toast('Content is required', 'error'); return; }
    const tags = document.getElementById('store-tags').value.split(',').map(t => t.trim()).filter(Boolean);
    try {
      await API.remember({
        content,
        memory_type: document.getElementById('store-type').value,
        visibility: document.getElementById('store-visibility').value,
        tags,
        salience: parseFloat(document.getElementById('store-salience').value),
      });
      app.toast('Memory stored', 'success');
      app.nav('memories');
    } catch (e) { app.toast(e.message, 'error'); }
  },
};
