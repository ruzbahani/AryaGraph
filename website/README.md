# AryaGraph website

Source of the AryaGraph documentation site, published at **https://ruzbahani.com/aryagraph/**.

The site is **static**: plain HTML, CSS, JavaScript and images. The search box and the interactive figures run in the
visitor's browser, so any ordinary web hosting (FTP or cPanel, Apache or Nginx) can serve it. Python is only needed
on the computer that builds the site, never on the server.

## Contents

| Folder | What it holds |
|---|---|
| `source/` | The pages (MyST Markdown), `conf.py`, styles and images |
| `source/reference/` | API reference pages, generated from the library's docstrings |
| `scripts/assets/` | One script per section that draws that section's figures with AryaGraph |
| `scripts/` | Build, check and packaging tools |
| `_build/html/` | The built site (created by the build) |
| `dist/` | The upload-ready folder and zip (created by the packaging step) |

## Build

From the repository root:

```bash
pip install -e .                          # AryaGraph itself
pip install -r website/requirements.txt   # Sphinx, the PyData theme, MyST, sphinx-design, sphinx-copybutton
cd website
python scripts/build_site.py              # API pages + figures + strict HTML build
python scripts/package_site.py            # dist/aryagraph/ and dist/aryagraph-website.zip
```

`build_site.py --no-assets` skips regenerating the figures when only text changed.

## Publish on ordinary hosting

1. Build and package as above.
2. Upload the contents of `dist/aryagraph/` to `public_html/aryagraph/` with FTP, **or** upload
   `dist/aryagraph-website.zip` through the cPanel File Manager and extract it inside `public_html/aryagraph/`.
3. Open https://ruzbahani.com/aryagraph/.

All links inside the site are relative, so it works from `/aryagraph/` or from any other folder. The included
`.htaccess` sets `index.html` as the folder index and sends unknown URLs back to the home page on Apache hosts
(adjust the path in it if the site lives somewhere other than `/aryagraph/`).

## Writing and checking pages

- Pages are MyST Markdown. Every ```` ```python ```` block is executed by the checker, so examples stay correct.
- `python scripts/check_docs.py` runs every page's code and checks the writing rules (no em dash character; absolute
  wording such as "always" or "never" is flagged for review).
- Figures are never drawn by hand: each section's script in `scripts/assets/` produces them with AryaGraph, and
  `python scripts/build_assets.py <section>` regenerates one section.
