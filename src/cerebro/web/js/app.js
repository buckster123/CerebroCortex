/* CerebroCortex Dashboard — Main App */

const app = {
  currentTab: 'cortex',
  ws: null,
  wsConnected: false,
  watchEnabled: false,

  init() {
    this.initParticles();
    this.initWebSocket();
    this.loadStats();
    this.loadWatchStatus();
    this.nav('cortex');
    setInterval(() => this.loadStats(), 30000);
  },

  nav(tab) {
    this.currentTab = tab;
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.getElementById('panel-' + tab)?.classList.add('active');
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`.nav-btn[data-tab="${tab}"]`)?.classList.add('active');

    switch (tab) {
      case 'cortex': cortexPanel.load(); break;
      case 'memories': memoriesPanel.load(); break;
      case 'store': storePanel.load(); break;
      case 'ingest': ingestPanel.load(); break;
      case 'graph': graphPanel.load(); break;
      case 'trash': trashPanel.load(); break;
      case 'tags': tagsPanel.load(); break;
      case 'threads': threadsPanel.load(); break;
      case 'dream': dreamPanel.load(); break;
      case 'decay': decayPanel.load(); break;
      case 'health': healthPanel.load(); break;
      case 'audit': auditPanel.load(); break;
      case 'settings': settingsPanel.load(); break;
    }
  },

  async loadStats() {
    try {
      const s = await API.stats();
      document.getElementById('stat-memories').textContent = s.nodes ?? '-';
      document.getElementById('stat-links').textContent = s.links ?? '-';
      document.getElementById('stat-episodes').textContent = s.episodes ?? '-';
    } catch (e) { console.warn('stats', e); }
  },

  async loadWatchStatus() {
    try {
      const s = await API.watchStatus();
      this.watchEnabled = s.running;
      this.updateWatchToggle();
    } catch (e) { console.warn('watch status', e); }
  },

  async toggleWatch() {
    try {
      const enable = !this.watchEnabled;
      await API.watchToggle({ enabled: enable });
      this.watchEnabled = enable;
      this.updateWatchToggle();
      app.toast(enable ? 'Watcher activated' : 'Watcher deactivated', enable ? 'success' : 'info');
    } catch (e) {
      app.toast('Watch toggle failed: ' + e.message, 'error');
    }
  },

  updateWatchToggle() {
    const el = document.getElementById('watch-toggle');
    el.classList.toggle('active', this.watchEnabled);
    document.getElementById('watch-label').textContent = this.watchEnabled ? 'Watcher On' : 'Watcher Off';
  },

  toast(message, type = 'info') {
    const c = document.getElementById('toast-container');
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.textContent = message;
    c.appendChild(t);
    setTimeout(() => t.remove(), 4000);
  },

  showModal(title, contentHtml, actionsHtml = '') {
    const overlay = document.getElementById('modal-overlay');
    overlay.innerHTML = `<div class="modal" onclick="event.stopPropagation()">
      <h3>${title}</h3>
      <div class="modal-content">${contentHtml}</div>
      ${actionsHtml ? `<div class="modal-actions">${actionsHtml}</div>` : ''}
    </div>`;
    overlay.classList.add('show');
  },

  closeModal() {
    document.getElementById('modal-overlay').classList.remove('show');
  },

  initWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${location.host}/ws`;
    this.ws = new WebSocket(url);
    this.ws.onopen = () => {
      this.wsConnected = true;
      document.getElementById('ws-status').classList.remove('disconnected');
    };
    this.ws.onclose = () => {
      this.wsConnected = false;
      document.getElementById('ws-status').classList.add('disconnected');
      setTimeout(() => this.initWebSocket(), 5000);
    };
    this.ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        this.handleWsMessage(msg);
      } catch (_) {}
    };
  },

  handleWsMessage(msg) {
    if (msg.type === 'stats:refresh') {
      this.loadStats();
      if (this.currentTab === 'cortex') cortexPanel.load();
    }
    if (msg.type === 'memory:created' || msg.type === 'memory:deleted') {
      if (this.currentTab === 'memories') memoriesPanel.load();
      if (this.currentTab === 'trash') trashPanel.load();
    }
    if (msg.type === 'dream:phase_complete' || msg.type === 'dream:complete') {
      if (this.currentTab === 'dream') dreamPanel.load();
    }
  },

  initParticles() {
    const canvas = document.getElementById('particle-canvas');
    const ctx = canvas.getContext('2d');
    let w, h, particles = [];

    const resize = () => {
      w = canvas.width = window.innerWidth;
      h = canvas.height = window.innerHeight;
    };
    window.addEventListener('resize', resize);
    resize();

    for (let i = 0; i < 60; i++) {
      particles.push({
        x: Math.random() * w, y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        r: Math.random() * 1.5 + 0.5,
        alpha: Math.random() * 0.3 + 0.1,
      });
    }

    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      for (const p of particles) {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0) p.x = w; if (p.x > w) p.x = 0;
        if (p.y < 0) p.y = h; if (p.y > h) p.y = 0;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(212, 175, 55, ${p.alpha})`;
        ctx.fill();
      }
      // Draw connections
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(212, 175, 55, ${0.04 * (1 - dist / 120)})`;
            ctx.stroke();
          }
        }
      }
      requestAnimationFrame(draw);
    };
    draw();
  },
};

document.addEventListener('DOMContentLoaded', () => app.init());
