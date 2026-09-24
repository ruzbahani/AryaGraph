# Installation

AryaGraph installs from its source repository with pip. The core library needs only Python and numpy; optional extras add converters for other libraries, a browser-free PNG and PDF exporter, and the tools for running the test suite.

## Requirements

- **Python 3.10 or newer.** The project's test workflow runs on Python 3.10, 3.11, 3.12 and 3.13, on Linux, Windows and macOS.
- **numpy 1.23 or newer**, the only runtime dependency. pip installs it for you if it is missing.
- **git**, to clone the repository (or download the source as a ZIP file from GitHub).

What each task needs:

| Task | Needs |
|---|---|
| Build graphs, run algorithms, compute layouts, run simulations | Core install (numpy) |
| Save SVG and interactive HTML, display figures in Jupyter | Core install |
| Save PNG or PDF | cairosvg, **or** an installed Chrome, Edge, Chromium or Brave browser |
| Convert to and from networkx, pandas and SciPy; `to_pandas()` methods | The `interop` extra |
| Run the test suite | The `test` extra |

## Create a virtual environment

A virtual environment keeps AryaGraph and its dependencies separate from other projects. It is optional but recommended:

::::{tab-set}

:::{tab-item} macOS and Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

:::

:::{tab-item} Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

:::

::::

While the environment is active, `python` and `pip` refer to the environment's own copies. Run `deactivate` to leave it.

## Install from source

Clone the repository and install it with pip:

```bash
git clone https://github.com/ruzbahani/AryaGraph.git
cd AryaGraph
pip install .
```

pip reads `pyproject.toml`, builds the package, installs numpy if needed, and adds the `aryagraph` [command-line tool](../user-guide/cli.md) to your environment. The bundled datasets, such as the University of Calgary campus network, are installed with the package, so they work offline.

```{tip}
If more than one Python is installed, `python -m pip install .` makes sure pip installs into the interpreter that `python` starts.
```

Without git, open the repository on GitHub, choose **Code › Download ZIP**, unpack the archive, and run `pip install .` inside the unpacked folder.

## Optional extras

Extras install optional dependencies alongside AryaGraph. Run these commands inside the cloned `AryaGraph` folder; the quotes keep shells such as zsh from interpreting the brackets.

| Extra | Installs | Enables |
|---|---|---|
| `interop` | networkx ≥ 3.0, pandas ≥ 1.5, SciPy ≥ 1.9 | Converters such as {py:func}`~aryagraph.io.interop.to_networkx`, and `to_pandas()` on results |
| `png` | cairosvg ≥ 2.5 | PNG and PDF export without a browser |
| `test` | pytest ≥ 7, networkx ≥ 3.4, pandas ≥ 1.5, SciPy ≥ 1.9 | Running the test suite |

```bash
pip install ".[interop]"
pip install ".[png]"
pip install ".[interop,png]"
```

The optional packages are imported only when a feature needs them. Calling such a feature without its package raises {py:class}`~aryagraph.core.exceptions.DependencyError`, which names the missing package and the extra that provides it.

### Development install

To work on AryaGraph itself, install it in editable mode with the test tools, then run the tests from the repository root:

```bash
pip install -e ".[test]"
python -m pytest
```

In editable mode, changes to the files under `src/aryagraph` take effect without reinstalling. The test suite compares AryaGraph's results with networkx wherever networkx has an equivalent function, which is why the `test` extra includes it. See [Contributing](../project/contributing.md) for the rest of the workflow.

## Verify the installation

Print the installed version:

```bash
python -c "import aryagraph as ag; print(ag.__version__)"
aryagraph --version
```

```text
0.1.0
AryaGraph 0.1.0
```

Then run a short smoke test. It loads a bundled dataset, draws it, and writes an SVG and an interactive HTML file, none of which needs anything beyond the core install:

```python
import aryagraph as ag

campus = ag.gen.ucalgary_campus()
fig = ag.draw(campus, node_color="kind")
fig.save("check.svg")
fig.save("check.html")
print(campus)
print(fig)
```

```text
<Graph 'University of Calgary main campus': 56 nodes, 83 edges>
<Figure 861×747: 56 nodes, 83 edges, stress layout, theme 'light'>
```

Open `check.html` in a browser to see the interactive view. If both lines print as shown, the installation works.

## PNG and PDF export

AryaGraph writes SVG and HTML itself. For PNG and PDF it converts the SVG it has drawn, using the first converter it finds:

1. **cairosvg**, if it can be imported (install it with the `png` extra);
2. otherwise a **Chromium-family browser** in headless mode: Chrome, Edge, Chromium or Brave.

The browser is looked up in this order: the path in the `ARYAGRAPH_BROWSER` environment variable; the commands `chrome`, `google-chrome`, `google-chrome-stable`, `chromium`, `chromium-browser`, `msedge`, `microsoft-edge` and `brave-browser` on your `PATH`; then the standard install locations of Chrome and Edge on Windows and of Chrome, Edge and Chromium on macOS. To see which browser AryaGraph will use:

```python
from aryagraph.render.export import find_browser

print(find_browser())
```

This prints the browser's path, or `None` when no browser was found. If your browser is installed somewhere else, point AryaGraph to it:

::::{tab-set}

:::{tab-item} macOS and Linux

```bash
export ARYAGRAPH_BROWSER="/path/to/chrome"
```

:::

:::{tab-item} Windows (PowerShell)

```powershell
$env:ARYAGRAPH_BROWSER = "C:\path\to\chrome.exe"
```

:::

::::

By default a PNG is rendered at twice the figure's pixel size; pass `scale` to change it, as in `fig.save("figure.png", scale=3)`. A PDF keeps the drawing as vectors at its natural size.

```{note}
cairosvg relies on the Cairo graphics library, which pip does not install. On Linux and macOS, install Cairo with your package manager (for example apt on Debian and Ubuntu, or Homebrew on macOS). On Windows, where Cairo is usually absent, the simplest route is to skip the `png` extra and let AryaGraph use Chrome or Edge.
```

On a server without a desktop, install cairosvg together with Cairo, or Chromium, before exporting PNG or PDF.

## Use AryaGraph in Jupyter

Install Jupyter into the same environment as AryaGraph, then start it from there:

```bash
pip install jupyterlab
jupyter lab
```

Figures, charts and reports display themselves. When a {py:class}`~aryagraph.render.figure.Figure` is the last expression of a cell, the notebook shows the interactive view inline; charts from `ag.charts` and simulation `plot()` calls, and the dashboard of an {py:func}`~aryagraph.analysis.report.analyze` report, display the same way. The inline view is the self-contained HTML page inside a frame, so it needs no notebook extension and no internet connection.

```pycon
>>> import aryagraph as ag
>>> campus = ag.gen.ucalgary_campus()
>>> ag.draw(campus, node_color="kind")   # displays inline as the last expression
```

`fig.show()` does the same from anywhere in a cell. Outside a notebook, `fig.show()` writes the interactive page to a temporary file and opens it in your default web browser, which is handy in scripts and in the Python shell.

If your notebook server runs from a different environment than AryaGraph, register the environment as a kernel and select it in the notebook:

```bash
pip install ipykernel
python -m ipykernel install --user --name aryagraph
```

## Update or uninstall

To update, pull the latest source and reinstall:

```bash
cd AryaGraph
git pull
pip install .
```

An editable install picks up the new source after `git pull` without reinstalling. To remove AryaGraph:

```bash
pip uninstall aryagraph
```

## Troubleshooting

`ModuleNotFoundError: No module named 'aryagraph'`
: AryaGraph was installed into a different Python than the one you are running. Check which interpreter runs with `python -c "import sys; print(sys.executable)"`, activate the right environment, and install with `python -m pip install .`. In Jupyter, make sure the notebook uses the kernel of that environment.

`ERROR: Package 'aryagraph' requires a different Python`
: Your Python is older than 3.10. Install Python 3.10 or newer and create the virtual environment with it.

`zsh: no matches found: .[png]`
: The shell interpreted the brackets. Quote the argument: `pip install ".[png]"`.

`DependencyError: ... requires the optional package 'cairosvg' (pip install cairosvg)`
: No PNG/PDF converter was found. Install Chrome, Edge, Chromium or Brave, or set `ARYAGRAPH_BROWSER` to a browser that is installed in an unusual place. Alternatively install cairosvg with `pip install cairosvg` (or `pip install ".[png]"` from the cloned folder); cairosvg also needs the Cairo library of your system, and without it AryaGraph uses the browser instead. For a missing networkx, pandas or SciPy, install that package, or run `pip install ".[interop]"` from the cloned folder.

`RuntimeError: the headless browser did not finish within 120 s`
: The headless browser did not produce the file in time. Try another browser through `ARYAGRAPH_BROWSER`, or install cairosvg with Cairo for browser-free export.

A figure shows as an empty frame in a notebook
: Some notebook front ends restrict HTML output from notebooks they do not trust. Mark the notebook as trusted, or save the figure with `fig.save("figure.html")` and open the file in a browser.

If a problem is not listed here, please [open an issue on GitHub](https://github.com/ruzbahani/AryaGraph/issues) with the full error message, your operating system, and the output of `python --version` and `python -c "import aryagraph, numpy; print(aryagraph.__version__, numpy.__version__)"`.
