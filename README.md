# IPython plugin for DataLinter

[![License](https://img.shields.io/pypi/l/datalintermagic)](https://pypi.org/project/datalintermagic/)
[![tests](https://github.com/zgornel/Ipython-datalinter/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/zgornel/Ipython-datalinter/actions/workflows/tests.yml?query=branch%3Amain)
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
```
docker pull ghcr.io/zgornel/datalinter-compiled:latest
```
The command to start the linting server is:
```
docker run -it --rm -p10000:10000 \
    ghcr.io/zgornel/datalinter-compiled:latest \
        /datalinterserver/bin/datalinterserver \
            -i 0.0.0.0 \
            --config-path /datalinter/config/default.toml \
            --log-level debug
```
If the server starts correctly, it should display something like:
```
Warning: KB file not correctly specified, defaults will be used.
└ @ datalinterserver /DataLinter/apps/datalinterserver/src/datalinterserver.jl:83
[ Info: • Data linting server online @0.0.0.0:10000...
[ Info: Listening on: 0.0.0.0:10000, thread id: 1
```

## License

This code has an GNU GPLv3 license.

## Contributing

To report a bug or request a feature, please [file an issue](https://github.com/zgornel/Ipython-datalinter/issues/new).

Recent changes can be found in [CHANGELOG.md](CHANGELOG.md).

## References

[1] https://en.wikipedia.org/wiki/Lint_(software)
