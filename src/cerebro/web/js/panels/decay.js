/* Decay Panel — Temporal activation visualization */
const decayPanel = {
  load() {
    const el = document.getElementById('decay-body');
    el.innerHTML = `
      <div class="panel-header">
        <h2>☠ Decay</h2>
        <div style="display:flex; gap:8px; align-items:center;">
          <select id="decay-layer-filter" onchange="decayPanel.loadHeatmap()">
            <option value="">All Layers</option>
            <option value="sensory">Sensory</option>
            <option value="working">Working</option>
            <option value="long_term">Long-term</option>
            <option value="cortex">Cortex</option>
          </select>
          <button class="btn small" onclick="decayPanel.loadHeatmap()">↻ Refresh</button>
        </div>
      </div>
      <div style="padding: 16px;">
        <div style="margin-bottom: 16px;">
          <h3 style="font-size: 14px; margin-bottom: 8px;">Activation Scatter</h3>
          <p style="font-size: 11px; color: var(--text-muted); margin-bottom: 8px;">
            X = age (hours), Y = ACT-R activation, color = layer, size = salience
          </p>
          <canvas id="decay-scatter" width="800" height="320" style="width:100%; height:320px; background: var(--panel-bg); border: 1px solid var(--border); border-radius: 6px;"></canvas>
        </div>
        <div style="margin-bottom: 16px;">
          <h3 style="font-size: 14px; margin-bottom: 8px;">At Risk — Fading Memories</h3>
          <p style="font-size: 11px; color: var(--text-muted); margin-bottom: 8px;">
            Memories in sensory/working layers not accessed in >24h
          </p>
          <div id="decay-at-risk"></div>
        </div>
        <div>
          <h3 style="font-size: 14px; margin-bottom: 8px;">Sample Decay Curve</h3>
          <p style="font-size: 11px; color: var(--text-muted); margin-bottom: 8px;">
            Theoretical ACT-R decay for a memory with no further accesses
          </p>
          <canvas id="decay-curve" width="800" height="200" style="width:100%; height:200px; background: var(--panel-bg); border: 1px solid var(--border); border-radius: 6px;"></canvas>
        </div>
      </div>
    `;
    this.loadHeatmap();
    this.loadAtRisk();
    this.drawSampleCurve();
  },

  async loadHeatmap() {
    const layer = document.getElementById('decay-layer-filter')?.value || '';
    const params = layer ? `layer=${layer}` : '';
    try {
      const data = await API.activationHeatmap(params);
      this.drawScatter(data.points);
    } catch (e) {
      console.warn('heatmap', e);
    }
  },

  async loadAtRisk() {
    try {
      const data = await API.activationAtRisk();
      const el = document.getElementById('decay-at-risk');
      if (!data.memories.length) {
        el.innerHTML = '<p style="font-size: 12px; color: var(--green);">No memories at risk — all are well-accessed.</p>';
        return;
      }
      el.innerHTML = data.memories.map(m => `
        <div style="padding: 8px; border-bottom: 1px solid var(--border); font-size: 12px;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color: var(--text-muted); font-size: 10px; text-transform: uppercase;">${m.layer} • ${m.hours_since_access}h idle</span>
            <span style="font-size: 10px; color: ${m.retrievability < 0.3 ? '#ff4444' : m.retrievability < 0.6 ? 'var(--amber)' : 'var(--green)'}">
              R=${m.retrievability}
            </span>
          </div>
          <div style="margin-top: 4px; line-height: 1.4;">${m.content_preview}</div>
          <div style="margin-top: 4px;">
            <button class="btn small" onclick="decayPanel.revive('${m.id}')">Revive</button>
            <button class="btn small" onclick="decayPanel.showCurve('${m.id}')">Curve</button>
          </div>
        </div>
      `).join('');
    } catch (e) {
      console.warn('at-risk', e);
    }
  },

  async revive(id) {
    try {
      await API.recall({ query: id, top_k: 1 });
      app.toast('Memory revived — activation boosted', 'success');
      this.loadAtRisk();
      this.loadHeatmap();
    } catch (e) {
      app.toast('Revive failed: ' + e.message, 'error');
    }
  },

  async showCurve(id) {
    try {
      const data = await API.activationCurve(id, 30);
      this.drawCurve(data.curve, `Decay curve for ${id.slice(0, 8)}...`);
    } catch (e) {
      app.toast('Curve fetch failed: ' + e.message, 'error');
    }
  },

  drawScatter(points) {
    const canvas = document.getElementById('decay-scatter');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (!points.length) {
      ctx.fillStyle = 'var(--text-muted)';
      ctx.font = '14px sans-serif';
      ctx.fillText('No data', w / 2 - 30, h / 2);
      return;
    }

    const pad = 40;
    const maxAge = Math.max(...points.map(p => p.age_hours), 1);
    const minAct = Math.min(...points.map(p => p.activation), -5);
    const maxAct = Math.max(...points.map(p => p.activation), 1);

    const layerColors = {
      sensory: '#ff4444',
      working: '#d4af37',
      long_term: '#44ff88',
      cortex: '#8888ff',
    };

    // Axes
    ctx.strokeStyle = 'var(--border)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, pad);
    ctx.lineTo(pad, h - pad);
    ctx.lineTo(w - pad, h - pad);
    ctx.stroke();

    // Labels
    ctx.fillStyle = 'var(--text-muted)';
    ctx.font = '10px sans-serif';
    ctx.fillText('Age (hours) →', w - 80, h - 10);
    ctx.save();
    ctx.translate(12, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Activation →', 0, 0);
    ctx.restore();

    // Points
    for (const p of points) {
      const x = pad + (p.age_hours / maxAge) * (w - pad * 2);
      const y = h - pad - ((p.activation - minAct) / (maxAct - minAct || 1)) * (h - pad * 2);
      const r = 2 + p.salience * 4;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = layerColors[p.layer] || '#aaa';
      ctx.globalAlpha = 0.6;
      ctx.fill();
      ctx.globalAlpha = 1.0;
    }

    // Legend
    let lx = w - 120;
    for (const [layer, color] of Object.entries(layerColors)) {
      ctx.fillStyle = color;
      ctx.fillRect(lx, 12, 10, 10);
      ctx.fillStyle = 'var(--text-muted)';
      ctx.fillText(layer, lx + 14, 20);
      lx -= 70;
    }
  },

  drawSampleCurve() {
    // Draw a theoretical curve without needing an ID
    const days = 30;
    const curve = [];
    for (let d = 0; d <= days; d++) {
      const activation = Math.log(1 + 10 * Math.pow(d + 1, -0.5));
      const retrievability = Math.pow(1 + d / 9, -1);
      curve.push({ day: d, activation, retrievability });
    }
    this.drawCurve(curve, 'Theoretical decay (no further accesses)');
  },

  drawCurve(curve, label) {
    const canvas = document.getElementById('decay-curve');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (!curve.length) return;

    const pad = 30;
    const maxDay = curve[curve.length - 1].day;
    const minAct = Math.min(...curve.map(c => c.activation), -2);
    const maxAct = Math.max(...curve.map(c => c.activation), 2);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = pad + (i / 5) * (h - pad * 2);
      ctx.beginPath();
      ctx.moveTo(pad, y);
      ctx.lineTo(w - pad, y);
      ctx.stroke();
    }

    // Activation line
    ctx.strokeStyle = '#d4af37';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < curve.length; i++) {
      const c = curve[i];
      const x = pad + (c.day / maxDay) * (w - pad * 2);
      const y = h - pad - ((c.activation - minAct) / (maxAct - minAct || 1)) * (h - pad * 2);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Retrievability line
    ctx.strokeStyle = '#44aaff';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    for (let i = 0; i < curve.length; i++) {
      const c = curve[i];
      const x = pad + (c.day / maxDay) * (w - pad * 2);
      const y = h - pad - c.retrievability * (h - pad * 2);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Legend
    ctx.fillStyle = '#d4af37';
    ctx.fillRect(pad, 10, 12, 3);
    ctx.fillStyle = 'var(--text-muted)';
    ctx.font = '10px sans-serif';
    ctx.fillText('Activation', pad + 16, 14);
    ctx.fillStyle = '#44aaff';
    ctx.fillRect(pad + 80, 10, 12, 3);
    ctx.fillStyle = 'var(--text-muted)';
    ctx.fillText('Retrievability', pad + 96, 14);

    // Label
    ctx.fillStyle = 'var(--text-muted)';
    ctx.fillText(label, pad, h - 6);
  },
};
