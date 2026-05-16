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
          ${this.renderSection('File Watcher', this.settings.watch || {}, 'watch')}
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
    collect('watch', this.settings.watch || {});

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
