# Contributing

Thank you for your interest in contributing!

## Quick Start

1. Fork the repository
2. Create a branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `uv run pytest tests/ -v`
5. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/FreeTheEnd.git
cd FreeTheEnd

# Set up Python 3.12 environment
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# Build C++ extension
cd cpp/build && cmake .. && cmake --build . && cd ../..
cp cpp/build/mc189_core.cpython-*-darwin.so python/minecraft_sim/

# Run tests
uv run pytest tests/ -v
```

## Code Style

- Python: Follow PEP 8, use `ruff` for linting
- C++: Follow existing style in `cpp/src/`
- Shaders: GLSL with consistent naming

## What to Contribute

**Good first issues:**
- Documentation improvements
- Test coverage
- Bug fixes with clear reproduction steps

**Larger contributions:**
- New curriculum stages
- Performance optimizations
- CPU backend implementation

## Submitting Changes

1. Ensure all tests pass
2. Update documentation if needed
3. Write a clear PR description
4. Reference any related issues

## Questions?

Open an issue for discussion before starting large changes.
