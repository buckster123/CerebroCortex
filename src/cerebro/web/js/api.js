/* CerebroCortex Dashboard — API Client */

const API = (() => {
  const BASE = '';

  async function request(method, path, body, opts = {}) {
    const url = `${BASE}${path}`;
    const cfg = { method, headers: {} };
    if (body && !opts.formData) {
      cfg.headers['Content-Type'] = 'application/json';
      cfg.body = JSON.stringify(body);
    } else if (opts.formData) {
      cfg.body = body;
    }
    const res = await fetch(url, cfg);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  }

  return {
    get: (path) => request('GET', path),
    post: (path, body) => request('POST', path, body),
    put: (path, body) => request('PUT', path, body),
    delete: (path) => request('DELETE', path),
    upload: (path, formData) => request('POST', path, formData, { formData: true }),

    // Core
    stats: () => API.get('/stats'),
    health: () => API.get('/health'),
    remember: (body) => API.post('/remember', body),
    recall: (body) => API.post('/recall', body),
    getMemory: (id) => API.get(`/memory/${id}`),
    deleteMemory: (id) => API.delete(`/memory/${id}`),
    updateMemory: (id, body) => API.patch?.call ? fetch(`${BASE}/memory/${id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(r => r.json()) : API.post(`/memory/${id}`, body),

    // Ingestion
    uploadFile: (formData) => API.upload('/ingest/upload', formData),

    // Trash
    listTrash: () => API.get('/trash'),
    restoreTrash: (id) => API.post(`/trash/${id}/restore`),
    purgeTrash: (id) => API.delete(`/trash/${id}`),
    purgeAllTrash: (days = 0) => API.post(`/trash/purge-all?older_than_days=${days}`),

    // Versions
    getVersions: (id) => API.get(`/memory/${id}/versions`),
    restoreVersion: (id, vid) => API.post(`/memory/${id}/versions/${vid}/restore`),

    // Tags
    listTags: () => API.get('/tags'),
    renameTag: (body) => API.post('/tags/rename', body),
    mergeTags: (body) => API.post('/tags/merge', body),
    deleteTag: (tag) => API.delete(`/tags/${encodeURIComponent(tag)}`),

    // Threads
    listThreads: () => API.get('/threads'),
    getThreadMemories: (id) => API.get(`/threads/${id}/memories`),
    pruneThread: (id) => API.delete(`/threads/${id}`),

    // Bulk
    bulkDelete: (body) => API.post('/bulk/delete', body),
    bulkVisibility: (body) => API.post('/bulk/visibility', body),
    export: (body) => API.post('/export', body),

    // Graph
    graphData: () => API.get('/graph/data'),
    graphStats: () => API.get('/graph/stats'),
    neighbors: (id) => API.get(`/graph/neighbors/${id}`),

    // Dream
    dreamRun: () => API.post('/dream/run'),
    dreamStatus: () => API.get('/dream/status'),

    // Settings
    getSettings: () => API.get('/settings'),
    updateSettings: (body) => API.put('/settings', body),
    resetSettings: () => API.post('/settings/reset'),

    // Watch
    watchStatus: () => API.get('/watch/status'),
    watchToggle: (body) => API.post('/watch/toggle', body),

    // Agents
    listAgents: () => API.get('/agents'),
  };
})();
