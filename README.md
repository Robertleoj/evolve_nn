# Python/C++ project template

This is a template for a python/C++ monorepo.

## Repository structure

This is a mixed C++ and Python repository.
The main use of the C++ code is through [`pybind11`](https://pybind11.readthedocs.io/en/stable/) bindings, similar to the [`mitsuba`](https://github.com/mitsuba-renderer/mitsuba3) project.
That is, while you can write everything in C++, in general we use Python as a configuration and glue layer.

Until we find the need for something more complicated (like `colcon`, `bazel`) we will stick to the basics.

We use the following structure:

```
├── [build] # This contains the C++ build, not checked in to the repository.
├── data # Small data files.
├── external # Third party C++ libraries, managed using git submodules
├── include  # C++ headers
├── notebooks # Jupyter notebooks
├── scripts # Python scripts (e.g. a Recorder)
├── project # The package (including C++ sources)
├── tests # pytest tests
└── tools # Tools for linting and building
```

## Prerequisites

This project assumes the following system packages are installed:

- `sudo apt-get install ninja-build`

## First time use

Fork the repository to your GitHub account.

If you have no SSH key, create one following https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

If you already have an SSH key ensure it is registered to you GitHub account. You can check this by:

```shell-session
$ ssh -T git@github.com
Hi [GH-HANDLE]! You've successfully authenticated, but GitHub does not provide shell access.
```

If this does not succeed, please follow https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account.

Then, clone the fork:
```shell-session
git clone git@github.com:[GH_USER]/template.git
cd template
```

Add upstream for future merging with the main fork:
```shell-session
git remote add upstream git@github.com:example/example.git
```

We use Poetry to manage Python dependencies.

If you have not installed Poetry:

```shell-session
$ curl -sSL https://install.python-poetry.org | python3 -
```

You should also check that you're using python 3.10.6:

```shell-session
$ python --version
Python 3.10.6
```

If not, use `pyenv` to install Python 3.10.6 and set that as the default.

```shell-session
$ pyenv install 3.10.6
$ poetry env use 3.10.6
```

Initialize Poetry and enter the Poetry shell (virtual environment).

```shell-session
$ poetry shell
$ poetry install
```

We will always assume commands are executed from within the Poetry shell.

Set up pre-commit hooks (formatting, linting, ...)

```shell-session
pre-commit install --hook-type pre-commit --hook-type pre-push
```

Configure the Git submodules needed for building the C++ backend.

```shell-session
$ git submodule update --init --recursive
```

### Building the C++ backend

You need to do this for the initial setup and every time there are changes to the backend.

The C++ layer is called "foundation", and the code resides under `src/project/foundation`.

To build, you can run:

```shell-session
$ ./tools/build.py build # See also clean_build and clean options.
```

This builds the backend code and adds a symlink from the bindings module in the build tree to the Python source tree[^1].

[^1]: This is essentially what a tool like `colcon` does with more magic behind the scenes.

This makes the bindings show up transparently for Python allowing you to write:

```python
from project.foundation import ...
```

To confirm that the build and setup was successful, you can run the tests:

```shell-session
$ py.test
```

## Formatting and Linting

This is automatically done by `pre-commit`, but that only operates on committed files.

You can run everything manually as:

```shell-session
$ ruff check . # Linting + isort
$ # ruff check --fix . # Optionally automatically fix issues
$ python -m black . # Formatting.
$ ./tools/run_clang_format.py # C++ formatting.
```

## Testing

```shell-session
$ py.test
```

We also write all tests for C++ code in Python (exactly like in the `mitsuba` project).

Note: until we set up a CI system (and ideally always) make sure that all tests pass before merging any code to the main repository.
`pre-commit` is configured to run all Python tests before allowing push to automate this.


## Jupyter Notebooks

We use [Jupytext](https://github.com/mwouts/jupytext) [percent format](https://github.com/mwouts/jupytext/blob/main/docs/formats.md#The-percent-format) to store Jupyter notebooks in our repository.
This format strips all binary data (images, ...), but will transparently open in Jupyterlab.
This format makes it easy to test and modify notebooks since they'll behave like normal Python code.


## Data

Small data files can live in the `data` folder.

We will not store large binary data in this repository (if that is needed, we'll set up Git LFS).
We assume that large data files live under `/data/...` and we'll manually manage that storage (e.g. COCO, MPII datasets, pretrained neural networks).

## Secrets
Environment variables are used for secrets such as API keys.
These are stored in a `.env` file in the root of the repository.
This file is not checked in to the repository.
Instead, a `.env.example` file is checked in with dummy values.
To use the secrets, you need to obtain the real values and add them to your local `.env` file.

To load the environment variables, we use the `python-dotenv` package.
To use the environment variables, you need to import the package and call `load_dotenv()` in your code.

## User guide

# Developer guide

### Architecture

### Contributing
To contribute we'll make Pull Requests from our forks to the main repository.

Create a new branch:
```shell-session
git branch -b new_feature
```

Make modifications...

Commit changes, e.g.:
```shell-session
git add module_A.py
git commit -m "Adding module A for doing xyz"
git add utils.py
git commit -m "Adding a utility for module A"
```

Merge with upstream master:
```shell-session
git fetch upstream
git rebase upstream/main
```

Fix files with conflicts and add+commit changes

Push the changes to your fork:
```shell-session
git push origin new_feature
```

On GitHub.com navigate to your branch and make a pull request, tag reviewers

See also https://www.atlassian.com/git/tutorials/git-forks-and-upstreams

### Profiling

We support the [`Tracy`](https://github.com/wolfpld/tracy) profiler.
To enable the profiler set the `TRACY_ENABLE` CMake flag in the top level `CMakeLists.txt` to `ON`.
Please refer to the `Tracy` documentation for details on how to build and use the profiler.


### Known Issues
