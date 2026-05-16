/* Dream Panel */
const dreamPanel = {
  async load() {
    const el = document.getElementById('dream-body');
    try {
      const status = await API.dreamStatus();
      const isRunning = status.status === 'running';
      el.innerHTML = `
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:24px;">
          <div style="width:12px; height:12px; border-radius:50%; background:${isRunning ? 'var(--gold)' : 'var(--text-muted)'}; box-shadow:${isRunning ? '0 0 12px var(--gold)' : 'none'};"></div>
          <span style="font-size:14px; color:${isRunning ? 'var(--gold)' : 'var(--text-muted)'};">${isRunning ? 'Oneiros is wandering...' : 'Oneiros sleeps.'}</span>
        </div>
        <button class="btn primary" ${isRunning ? 'disabled' : ''} onclick="dreamPanel.run()">&#x263d; Run Dream Cycle</button>
        ${status.last_report ? this.renderReport(status.last_report) : ''}
      `;
    } catch (e) {
      el.innerHTML = `<div class="empty-state">Error: ${e.message}</div>`;
    }
  },

  async run() {
    try { await API.dreamRun(); app.toast('Dream cycle started', 'success'); this.load(); }
    catch (e) { app.toast(e.message, 'error'); }
  },

  renderReport(r) {
    return `<div style="margin-top:24px; padding:16px; background:var(--bg-surface); border:1px solid var(--border); border-radius:var(--radius);">
      <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">Last Report</div>
      <div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(140px,1fr)); gap:12px;">
        <div><div style="font-size:10px; color:var(--text-muted);">Memories</div><div style="font-size:18px; color:var(--gold);">${r.memories_processed ?? '-'}</div></div>
        <div><div style="font-size:10px; color:var(--text-muted);">Links Created</div><div style="font-size:18px; color:var(--cyan);">${r.links_created ?? '-'}</div></div>
        <div><div style="font-size:10px; color:var(--text-muted);">Pruned</div><div style="font-size:18px; color:var(--magenta);">${r.memories_pruned ?? '-'}</div></div>
        <div><div style="font-size:10px; color:var(--text-muted);">Schemas</div><div style="font-size:18px; color:var(--purple);">${r.schemas_extracted ?? '-'}</div></div>
        <div><div style="font-size:10px; color:var(--text-muted);">Duration</div><div style="font-size:18px; color:var(--text-bright);">${r.duration_seconds ? r.duration_seconds + 's' : '-'}</div></div>
      </div>
    </div>`;
  },
};