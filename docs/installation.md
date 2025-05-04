# Installation

SmartSurge can be installed via pip:

```bash
pip install smartsurge
```

## Requirements

SmartSurge requires Python 3.8 or newer and has the following dependencies:

- requests (>=2.25.0)
- aiohttp (>=3.7.0)
- pydantic (>=2.0.0)
- scipy (>=1.7.0)

## Development Installation

If you want to contribute to SmartSurge, you can install the development version with additional dependencies:

```bash
git clone https://github.com/example/smartsurge.git
cd smartsurge
pip install -e ".[dev]"
```

## Verification

To verify that SmartSurge is installed correctly, you can run:

```python
import smartsurge
print(smartsurge.__version__)
```
