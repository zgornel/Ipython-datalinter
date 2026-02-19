# IPython plugin for DataLinter

> Warning: This work is still very experimental.

This is a Ipython magic that allows one to run [DataLinter](https://github.com/zgornel/DataLinter) in Jupyter notebooks.

[![License](http://img.shields.io/badge/license-GPL-brightgreen.svg?style=flat)](LICENSE)

![til](./gifs/jupyter.gif)

## Installation

Make sure you have up-to-date stable versions of [IPython](https://ipython.org/) and [Jupyter](https://jupyter.org/).

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

### Ipython-datalinter magic
Clone the repository:
```
git clone https://github.com/zgornel/Ipython-datalinter
```

To run the demo, enter the `Ipython-datalinter` folder, run
```
jupyter notebook
```
and open `ipython/demo.ipynb`.

> Note: In order for the `datalintermagics` magic to be available, the `datalintermagics` folder/module needs to be available to the jupyter notebook. The easy way to do this is to have the `datalintermagics` directory and notebooks in the same location.
