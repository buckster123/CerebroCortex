/* Settings Panel */
const settingsPanel = {
  settings: null,

  async load() {
    const el = document.getElementById('settings-body');
    el.innerHTML = '<div class="empty-state"><div class="spinner"></div></div>';
    try {
      this.settings = await API.getSettings();
      el.innerHTML = `
        <div style="max-width:600px;">
          ${this.renderSection('LLM', this.settings.llm, 'llm')}
          ${this.renderSection('Dream Engine', this.settings.dream, 'dream')}
          ${this.renderWatchSection()}
          <div style="margin-top:24px; display:flex; gap:10px;">
            <button class="btn primary" onclick="settingsPanel.save()">Save Settings</button>
            <button class="btn danger" onclick="settingsPanel.reset()">Reset to Defaults</button>
          </div>
        </div>
      `;
    } catch (e) {
      el.innerHTML = `<div class="empty-state">Error: ${e.message}</div>`;
    }
  },

  renderSection(title, data, prefix) {
    if (!data) return '';
    return `<div style="margin-bottom:24px;">
      <h3 style="font-family:var(--font-display); font-size:14px; color:var(--text-muted); letter-spacing:2px; margin-bottom:12px; text-transform:uppercase;">${title}</h3>
      <div style="display:flex; flex-direction:column; gap:10px;">
        ${Object.entries(data).map(([k, v]) => {
          const id = `${prefix}-${k}`;
          const isBool = typeof v === 'boolean';
          const isNum = typeof v === 'number';
          const input = isBool
            ? `<input type="checkbox" id="${id}" ${v ? 'checked' : ''} style="width:auto;">`
            : `<input type="${isNum ? 'number' : 'text'}" id="${id}" value="${v ?? ''}" ${isNum ? 'step="any"' : ''}>`;
          return `<div>
            <label style="display:block; font-size:11px; color:var(--text-muted); margin-bottom:4px; text-transform:uppercase; letter-spacing:1px;">${k}</label>
            ${input}
          </div>`;
        }).join('')}
      </div>
    </div>`;
  },

  renderWatchSection() {
    const data = this.settings.watch || {};
    const dirs = Array.isArray(data.dirs) ? data.dirs : (data.dirs ? [data.dirs] : []);
    const dirValue = dirs.join(', ') || '';
    const hasDefault = !dirValue;
    return `<div style="margin-bottom:24px;">
      <h3 style="font-family:var(--font-display); font-size:14px; color:var(--text-muted); letter-spacing:2px; margin-bottom:12px; text-transform:uppercase;">File Watcher</h3>
      <div style="display:flex; flex-direction:column; gap:10px;">
        <div>
          <label style="display:block; font-size:11px; color:var(--text-muted); margin-bottom:4px; text-transform:uppercase; letter-spacing:1px;">enabled</label>
          <input type="checkbox" id="watch-enabled" ${data.enabled ? 'checked' : ''} style="width:auto;">
        </div>
        <div>
          <label style="display:block; font-size:11px; color:var(--text-muted); margin-bottom:4px; text-transform:uppercase; letter-spacing:1px;">watch directory</label>
          <div style="display:flex; gap:8px;">
            <input type="text" id="watch-dirs" value="${dirValue}" placeholder="~/cerebro-watch (default)" style="flex:1;">
            <button class="btn small" onclick="settingsPanel.openDirPicker()">Browse</button>
          </div>
          ${hasDefault ? '<div style="font-size:11px; color:var(--text-muted); margin-top:4px;">Default: ~/cerebro-watch (auto-created when watcher starts)</div>' : ''}
        </div>
        <div>
          <label style="display:block; font-size:11px; color:var(--text-muted); margin-bottom:4px; text-transform:uppercase; letter-spacing:1px;">agent_id</label>
          <input type="text" id="watch-agent_id" value="${data.agent_id || ''}">
        </div>
        <div>
          <label style="display:block; font-size:11px; color:var(--text-muted); margin-bottom:4px; text-transform:uppercase; letter-spacing:1px;">tags</label>
          <input type="text" id="watch-tags" value="${Array.isArray(data.tags) ? data.tags.join(', ') : (data.tags || '')}">
        </div>
        <div>
          <label style="display:block; font-size:11px; color:var(--text-muted); margin-bottom:4px; text-transform:uppercase; letter-spacing:1px;">patterns</label>
          <input type="text" id="watch-patterns" value="${Array.isArray(data.patterns) ? data.patterns.join(', ') : (data.patterns || '')}">
        </div>
      </div>
    </div>`;
  },

  async openDirPicker() {
    const modal = document.getElementById('modal-overlay');
    modal.innerHTML = '<div class="modal" style="max-width:500px; max-height:70vh; overflow:auto;"><h3>Select Watch Directory</h3><div id="dir-browser" style="margin:12px 0;"><div class="spinner"></div></div><div style="display:flex; gap:10px; justify-content:flex-end; margin-top:12px;"><button class="btn" onclick="app.closeModal()">Cancel</button><button class="btn primary" id="dir-select-btn" disabled onclick="settingsPanel.confirmDir()">Select</button></div></div>';
    modal.style.display = 'flex';
    this._browsePath = null;
    this._selectedPath = null;
    await this._renderBrowser();
  },

  async _renderBrowser(path) {
    const el = document.getElementById('dir-browser');
    try {
      const data = await API.browse(path);
      this._browsePath = data.current;
      let html = '<div style="font-size:12px; color:var(--text-muted); margin-bottom:8px; word-break:break-all;">' + data.current + '</div>';
      html += '<div style="display:flex; flex-direction:column; gap:2px;">';
      if (data.parent) {
        html += `<div class="dir-entry" onclick="settingsPanel._renderBrowser('${data.parent}')" style="cursor:pointer; padding:6px; border-radius:4px; display:flex; align-items:center; gap:8px;">&#8593; ..</div>`;
      }
      for (const entry of data.entries) {
        if (entry.type === 'directory') {
          const selectedClass = (this._selectedPath === entry.path) ? ' style="background:rgba(212,175,55,0.15);"' : '';
          html += `<div class="dir-entry" onclick="settingsPanel._selectDir('${entry.path}')" ondblclick="settingsPanel._renderBrowser('${entry.path}')" style="cursor:pointer; padding:6px; border-radius:4px; display:flex; align-items:center; gap:8px;"${selectedClass}>
            <span style="color:var(--gold);">&#128193;</span> ${entry.name}
          </div>`;
        }
      }
      html += '</div>';
      el.innerHTML = html;
    } catch (e) {
      el.innerHTML = `<div class="empty-state">Error: ${e.message}</div>`;
    }
  },

  _selectDir(path) {
    this._selectedPath = path;
    document.getElementById('dir-select-btn').disabled = false;
    this._renderBrowser(this._browsePath);
  },

  confirmDir() {
    if (this._selectedPath) {
      document.getElementById('watch-dirs').value = this._selectedPath;
    }
    app.closeModal();
  },

  async save() {
    const updates = { llm: {}, dream: {}, watch: {} };
    const collect = (prefix, obj) => {
      if (!obj) return;
      for (const k of Object.keys(obj)) {
        const el = document.getElementById(`${prefix}-${k}`);
        if (!el) continue;
        let val = el.type === 'checkbox' ? el.checked : el.value;
        if (typeof obj[k] === 'number') val = parseFloat(val);
        if (val !== obj[k]) updates[prefix][k] = val;
      }
    };
    collect('llm', this.settings.llm);
    collect('dream', this.settings.dream);

    // Handle watch section manually
    const watchEnabled = document.getElementById('watch-enabled');
    if (watchEnabled) updates.watch.enabled = watchEnabled.checked;
    const watchDirs = document.getElementById('watch-dirs');
    if (watchDirs) {
      const raw = watchDirs.value.trim();
      updates.watch.dirs = raw ? raw.split(/,\s*/).filter(Boolean) : [];
    }
    const watchAgent = document.getElementById('watch-agent_id');
    if (watchAgent) updates.watch.agent_id = watchAgent.value || null;
    const watchTags = document.getElementById('watch-tags');
    if (watchTags) {
      const raw = watchTags.value.trim();
      updates.watch.tags = raw ? raw.split(/,\s*/).filter(Boolean) : [];
    }
    const watchPatterns = document.getElementById('watch-patterns');
    if (watchPatterns) {
      const raw = watchPatterns.value.trim();
      updates.watch.patterns = raw ? raw.split(/,\s*/).filter(Boolean) : [];
    }

    try {
      await API.updateSettings(updates);
      app.toast('Settings saved', 'success');
      this.load();
    } catch (e) { app.toast(e.message, 'error'); }
  },

  async reset() {
    if (!confirm('Reset all settings to defaults?')) return;
    try { await API.resetSettings(); app.toast('Settings reset', 'info'); this.load(); }
    catch (e) { app.toast(e.message, 'error'); }
  },
};
