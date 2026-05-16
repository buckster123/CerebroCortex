/* Graph Panel */
const graphPanel = {
  instance: null,
  mode3d: true,
  data: null,

  async load() {
    const el = document.getElementById('graph-body');
    if (!this.data) {
      try {
        this.data = await API.graphData();
      } catch (e) {
        el.innerHTML = `<div class="empty-state">Error loading graph: ${e.message}</div>`;
        return;
      }
    }
    this.render();
  },

  render() {
    const el = document.getElementById('graph-body');
    el.innerHTML = '';
    if (!this.data || !this.data.nodes.length) {
      el.innerHTML = '<div class="empty-state"><div class="icon">&#x2609;</div><p>Graph is empty.</p></div>';
      return;
    }

    const g = { nodes: this.data.nodes.map(n => ({ id: n.id, val: n.val || 1, type: n.type, content: n.content })), links: this.data.links };
    const container = document.createElement('div');
    container.style.width = '100%';
    container.style.height = '100%';
    el.appendChild(container);

    if (this.mode3d) {
      this.instance = ForceGraph3D()(container)
        .graphData(g)
        .nodeRelSize(6)
        .nodeColor(n => this.nodeColor(n.type))
        .nodeLabel(n => `${n.id.substring(0, 8)}... ${n.type}`)
        .linkColor(() => 'rgba(212,175,55,0.2)')
        .linkWidth(1)
        .backgroundColor('#0c0c1a')
        .showNavInfo(false);
    } else {
      this.instance = ForceGraph()(container)
        .graphData(g)
        .nodeRelSize(6)
        .nodeColor(n => this.nodeColor(n.type))
        .nodeLabel(n => `${n.id.substring(0, 8)}... ${n.type}`)
        .linkColor(() => 'rgba(212,175,55,0.3)')
        .linkWidth(1)
        .backgroundColor('#0c0c1a');
    }
  },

  toggleRenderMode() {
    this.mode3d = !this.mode3d;
    this.render();
  },

  nodeColor(type) {
    const map = { episodic: '#FF8C00', semantic: '#00d4ff', procedural: '#D4AF37', affective: '#ff0055', prospective: '#9d4edd', schematic: '#FF6B35' };
    return map[type] || '#D4AF37';
  },
};
