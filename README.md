# IPython plugin for DataLinter

This is a Ipython magic that allows one to run [DataLinter](https://github.com/zgornel/DataLinter) in Jupyter notebooks.

[![License](http://img.shields.io/badge/license-GPL-brightgreen.svg?style=flat)](LICENSE)

![til](./gifs/jupyter.gif)

## Installation

Make sure you have up-to-date stable versions of [IPython](https://ipython.org/) and [Jupyter](https://jupyter.org/).

### The Jupyter magic
The Jupyter magic can be installed with
```
pip install datalintermagic
```

### DataLinter

Install [DataLinter](https://github.com/zgornel/DataLinter) by pulling the Docker image:
```
docker pull ghcr.io/zgornel/datalinter-compiled:latest
```
and start it with:
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
