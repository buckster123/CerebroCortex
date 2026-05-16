# Module: cerebro-index
## Purpose
Navigational index for all Cerebro operational sub-modules. Use this to find the right sub-module for a specific memory operation.

## Activation Signals
(always loaded)

## Module Map
| Sub-Module | Use When | Key Tools |
|------------|----------|-----------|
| cerebro-ops | Storing, searching, linking memories | remember, recall, associate, search_vision |
| cerebro-session | Tracking session flow, checkpoints | episode_start, episode_add_step, episode_end, session_save |
| cerebro-intentions | TODOs, reminders, deferred tasks | store_intention, list_intentions, resolve_intention |
| cerebro-agents | Talking to other agents | send_message, check_inbox |
| cerebro-meta | Deciding WHAT to persist and WHERE | 90/10 rule, auto-remember triggers |

## Navigation Protocol
1. For any memory operation, first consult this index.
2. Load the relevant sub-module before executing the operation.
3. Sub-modules link to each other via "Related Modules" — follow the graph.

## Inhibitions
- Do not execute complex memory workflows without loading the relevant sub-module.
- Do not guess at MCP tool schemas — load cerebro-ops for the exact signatures.

## Related Modules
- module-cerebro-ops (storage and retrieval)
- module-cerebro-session (session tracking)
- module-cerebro-meta (persistence strategy)

## Version
0.1.0
