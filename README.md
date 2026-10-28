# IPython plugin for DataLinter

[![License](https://img.shields.io/pypi/l/datalintermagic)](https://pypi.org/project/datalintermagic/)
[![Tests](https://github.com/zgornel/Ipython-datalinter/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/zgornel/Ipython-datalinter/actions/workflows/tests.yml?query=branch%3Amain)
[![PyPI version](https://img.shields.io/pypi/v/datalintermagic)](https://pypi.org/project/datalintermagic/)
[![Python versions](https://img.shields.io/pypi/pyversions/datalintermagic)](https://pypi.org/project/datalintermagic/)
![til](./gifs/jupyter.gif)

This is a Ipython magic that allows one to run [DataLinter](https://github.com/zgornel/DataLinter) in Jupyter notebooks.

## Installation

Make sure you have up-to-date stable versions of [IPython](https://ipython.org/) and [Jupyter](https://jupyter.org/). In order for the linter to work, both the Jupyter plugin and DataLinter (server in a Docker image) need to be installed in the system.

### Jupyter plugin package

The Jupyter plugin can be installed with `pip`:
```
pip install datalintermagic
```

### DataLinter

[DataLinter](https://github.com/zgornel/DataLinter) installation is done with:
```bash
docker pull ghcr.io/zgornel/datalinter-compiled:latest
```
The command to start the linting server is:
```bash
docker run -d --rm -p10000:10000 \
    ghcr.io/zgornel/datalinter-compiled:latest \
        /datalinterserver/bin/datalinterserver \
            -i 0.0.0.0 \
            --config-path /datalinter/config/python_modelling_config.toml \
            --log-level info
```
If the server starts correctly, it should display something like:
```
[ Info: • Data linting server online @0.0.0.0:10000...
```

### Quick start
```python
%reload_ext datalintermagic
import numpy as np
X = np.random.random((10,100))

%add --tracked-variable X  --data-header --data-delim ','
%lintline X --show-stats
```

Further examples of linter usage in Jupyter are found in this [notebook](https://github.com/zgornel/Ipython-datalinter/blob/main/notebooks/workflow_example.ipynb).
## License

This code has an GNU GPLv3 license.

## Contributing

To report a bug or request a feature, please [file an issue](https://github.com/zgornel/Ipython-datalinter/issues/new).

Recent changes can be found in [CHANGELOG.md](CHANGELOG.md).

## References

[1] https://en.wikipedia.org/wiki/Lint_(software)
