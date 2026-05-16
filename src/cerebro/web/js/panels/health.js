/* Health Panel */
const healthPanel = {
  async load() {
    const el = document.getElementById('health-body');
    el.innerHTML = '<div class="empty-state"><div class="spinner"></div></div>';
    try {
      const data = await API.health();
      el.innerHTML = `
        <div class="card-grid">
          <div class="card"><div class="label">Total Memories</div><div class="value">${data.total_memories}</div></div>
          <div class="card"><div class="label">Total Links</div><div class="value cyan">${data.total_links}</div></div>
          <div class="card"><div class="label">Episodes</div><div class="value purple">${data.total_episodes}</div></div>
          <div class="card"><div class="label">Agents</div><div class="value magenta">${data.total_agents}</div></div>
        </div>
        ${data.emotional_summary ? `<h3 style="margin:24px 0 12px; font-family:var(--font-display); font-size:14px; color:var(--text-muted); letter-spacing:2px;">EMOTIONAL SUMMARY</h3>
        <div class="card-grid">${Object.entries(data.emotional_summary).map(([k,v]) => `<div class="card"><div class="label">${k}</div><div class="value">${v}</div></div>`).join('')}</div>` : ''}
      `;
    } catch (e) {
      el.innerHTML = `<div class="empty-state">Error: ${e.message}</div>`;
    }
  },
};
