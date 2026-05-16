/* Cortex Dashboard Panel */
const cortexPanel = {
  async load() {
    const el = document.getElementById('cortex-body');
    try {
      const s = await API.stats();
      el.innerHTML = `
        <div class="card-grid">
          <div class="card"><div class="label">Memories</div><div class="value">${s.nodes}</div></div>
          <div class="card"><div class="label">Links</div><div class="value cyan">${s.links}</div></div>
          <div class="card"><div class="label">Episodes</div><div class="value purple">${s.episodes}</div></div>
          <div class="card"><div class="label">Schemas</div><div class="value magenta">${s.schemas}</div></div>
        </div>
        <h3 style="margin: 24px 0 12px; font-family: var(--font-display); font-size: 14px; color: var(--text-muted); letter-spacing: 2px;">MEMORY TYPES</h3>
        <div class="card-grid">
          ${this.renderTypeBars(s.memory_types)}
        </div>
        <h3 style="margin: 24px 0 12px; font-family: var(--font-display); font-size: 14px; color: var(--text-muted); letter-spacing: 2px;">LAYERS</h3>
        <div class="card-grid">
          ${this.renderTypeBars(s.layers)}
        </div>
      `;
    } catch (e) {
      el.innerHTML = `<div class="empty-state">Failed to load stats: ${e.message}</div>`;
    }
  },

  renderTypeBars(data) {
    if (!data) return '';
    const max = Math.max(...Object.values(data));
    return Object.entries(data).map(([k, v]) => {
      const pct = max > 0 ? (v / max) * 100 : 0;
      const color = this.typeColor(k);
      return `<div class="card" style="padding: 14px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <span style="font-size:12px; text-transform:capitalize;">${k}</span>
          <span style="font-size:14px; font-weight:600; color:${color};">${v}</span>
        </div>
        <div style="height:4px; background:var(--bg-deep); border-radius:2px; overflow:hidden;">
          <div style="width:${pct}%; height:100%; background:${color}; border-radius:2px; transition:width 0.8s ease;"></div>
        </div>
      </div>`;
    }).join('');
  },

  typeColor(type) {
    const map = { episodic: '#FF8C00', semantic: '#00d4ff', procedural: '#D4AF37', affective: '#ff0055', prospective: '#9d4edd', schematic: '#FF6B35', sensory: '#888', working: '#aaa', long_term: '#ccc', cortex: '#fff' };
    return map[type] || '#D4AF37';
  },
};
