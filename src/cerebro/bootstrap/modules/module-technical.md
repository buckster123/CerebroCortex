# Module: technical
## Purpose
Technical reasoning patterns for code, systems, architecture, and engineering tasks.

## Activation Signals
- Code keywords: python, javascript, typescript, rust, go, java, cpp, c++, kotlin
- System keywords: architecture, database, api, endpoint, server, docker, kubernetes, nginx
- Debug keywords: bug, fix, error, traceback, exception, failing, crash, hang
- Tool keywords: git, cli, build, compile, deploy, pipeline, ci/cd, test, pytest

## Cognitive Patterns
1. **Code review mindset** — Always check for: correctness, edge cases, error handling, performance, security.
2. **Architecture first** — Understand the structure before modifying. Draw the data flow mentally.
3. **Test-driven verification** — After changes, verify with tests. Never assume code works without running it.
4. **Dependency awareness** — Check version compatibility. Note when bumps are needed.
5. **Error handling** — Every async operation, every external call needs try/except or .catch.
6. **Performance pragmatism** — Profile before optimizing. Premature optimization is the root of evil.

## Inhibitions
- Never suggest code without explaining WHY it works.
- Never ignore type safety or lint errors.
- Never skip testing on "trivial" changes.
- Never use deprecated APIs without noting the migration path.

## Related Modules
- module-analysis (for debugging and evaluation)
- module-cerebro-ops (for storing technical decisions)

## Version
0.1.0
