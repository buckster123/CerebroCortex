/* Ingest Panel */
const ingestPanel = {
  load() {
    const el = document.getElementById('ingest-body');
    el.innerHTML = `
      <div id="drop-zone" class="drop-zone" ondrop="ingestPanel.drop(event)" ondragover="ingestPanel.dragOver(event)" ondragleave="ingestPanel.dragLeave(event)" onclick="document.getElementById('file-input').click()">
        <div class="icon">⚡</div>
        <p>Drop files here or click to browse</p>
        <p style="font-size:11px; margin-top:8px; opacity:0.5;">Supports: md, txt, json, pdf, html, csv, png, jpg, py, js, ts, go, rs...</p>
        <input type="file" id="file-input" style="display:none" multiple onchange="ingestPanel.selectFiles(event)">
      </div>
      <div style="margin-top:16px;">
        <label style="font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:1px;">Tags</label>
        <input type="text" id="ingest-tags" placeholder="uploaded, research" style="margin-top:6px;">
      </div>
      <div id="ingest-progress" style="margin-top:16px;"></div>
    `;
  },

  dragOver(e) { e.preventDefault(); document.getElementById('drop-zone').classList.add('dragover'); },
  dragLeave(e) { e.preventDefault(); document.getElementById('drop-zone').classList.remove('dragover'); },

  drop(e) {
    e.preventDefault();
    document.getElementById('drop-zone').classList.remove('dragover');
    this.uploadFiles(e.dataTransfer.files);
  },

  selectFiles(e) {
    this.uploadFiles(e.target.files);
  },

  async uploadFiles(files) {
    const tags = document.getElementById('ingest-tags').value;
    const progress = document.getElementById('ingest-progress');
    const arr = Array.from(files);
    progress.innerHTML = `<div style="font-size:12px; color:var(--text-muted);">Uploading ${arr.length} file(s)...</div>`;

    for (const file of arr) {
      const fd = new FormData();
      fd.append('file', file);
      if (tags) fd.append('tags', tags);
      try {
        const res = await API.uploadFile(fd);
        let msg = `✓ ${file.name} → ${res.memories_created} created`;
        if (res.memories_skipped > 0) {
          msg += `, ${res.memories_skipped} skipped (duplicates/short)`;
        }
        progress.innerHTML += `<div style="font-size:12px; color:var(--green); margin-top:4px;">${msg}</div>`;
      } catch (e) {
        progress.innerHTML += `<div style="font-size:12px; color:#ff4444; margin-top:4px;">✗ ${file.name}: ${e.message}</div>`;
      }
    }
    app.loadStats();
  },
};
