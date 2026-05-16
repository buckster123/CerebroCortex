"""File watcher for auto-ingesting documents into CerebroCortex.

Usage::

    from cerebro.watch import FileWatcher
    watcher = FileWatcher(cortex)
    watcher.start()
    watcher.add_directory("/path/to/watch")
    # ...
    watcher.stop()
"""

from cerebro.watch.watcher import FileWatcher

__all__ = ["FileWatcher"]
