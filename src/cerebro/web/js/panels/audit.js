/* Audit Panel */
const auditPanel = {
  load() {
    const el = document.getElementById('audit-body');
    el.innerHTML = `
      <div class="panel-header">
        <h2>✶ Audit Log</h2>
        <div style="display:flex; gap:8px; align-items:center;">
          <select id="audit-filter-type" onchange="auditPanel.loadEntries()">
            <option value="">All Events</option>
            <option value="access_denial">Access Denials</option>
            <option value="visibility_changed">Visibility Changes</option>
            <option value="content_edited">Content Edits</option>
            <option value="memory_deleted">Deletions</option>
            <option value="link_created">Link Creation</option>
          </select>
          <button class="btn small" onclick="auditPanel.loadEntries()">↻ Refresh</button>
        </div>
      </div>
      <div style="padding: 16px;">
        <div id="audit-summary" style="margin-bottom: 16px; font-size: 12px; color: var(--text-muted);"></div>
        <div id="audit-entries"></div>
      </div>
    `;
    this.loadSummary();
    this.loadEntries();
  },

  async loadSummary() {
    try {
      const data = await API.auditSummary();
      const el = document.getElementById('audit-summary');
      if (!data.by_type.length) {
        el.innerHTML = 'No audit events recorded yet.';
        return;
      }
      el.innerHTML = `<b>${data.total_events}</b> total events — ` +
        data.by_type.map(t => `${t.event_type}: ${t.count}`).join(' · ');
    } catch (e) {
      console.warn('audit summary', e);
    }
  },

  async loadEntries() {
    const typeFilter = document.getElementById('audit-filter-type')?.value || '';
    const el = document.getElementById('audit-entries');
    el.innerHTML = '<div style="font-size:12px; color:var(--text-muted);">Loading...</div>';

    try {
      const body = { limit: 50 };
      if (typeFilter) body.event_type = typeFilter;
      const data = await API.queryAudit(body);

      if (!data.entries.length) {
        el.innerHTML = '<p style="font-size: 12px; color: var(--text-muted);">No audit entries match the filter.</p>';
        return;
      }

      el.innerHTML = data.entries.map(e => {
        const color = this._eventColor(e.event_type);
        return `
          <div style="padding: 8px; border-bottom: 1px solid var(--border); font-size: 12px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <span style="font-size: 10px; text-transform: uppercase; color: ${color}; font-weight: bold;">${e.event_type}</span>
              <span style="font-size: 10px; color: var(--text-muted);">${e.timestamp}</span>
            </div>
            <div style="margin-top: 4px; line-height: 1.4;">
              ${e.actor_agent_id ? `<b>${e.actor_agent_id}</b> → ` : ''}
              ${e.target_memory_id ? `memory <code>${e.target_memory_id.slice(0, 8)}...</code>` : ''}
            </div>
            ${e.old_value ? `<div style="margin-top: 2px; color: var(--text-muted); font-size: 11px;">old: ${e.old_value}</div>` : ''}
            ${e.new_value ? `<div style="margin-top: 2px; color: var(--text-muted); font-size: 11px;">new: ${e.new_value}</div>` : ''}
          </div>
        `;
      }).join('');
    } catch (e) {
      el.innerHTML = `<p style="font-size: 12px; color: #ff4444;">Error: ${e.message}</p>`;
    }
  },

  _eventColor(type) {
    const colors = {
      access_denial: '#ff4444',
      visibility_changed: '#d4af37',
      content_edited: '#44aaff',
      memory_deleted: '#ff8844',
      link_created: '#44ff88',
    };
    return colors[type] || '#aaa';
  },
};
