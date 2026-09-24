# Languages

Node labels, edge labels, titles, legends and tooltips can be written in any script, including Persian and other right-to-left scripts (Arabic, Hebrew) and Chinese, Japanese and Korean (CJK). This page explains how AryaGraph measures such text, how it handles direction, and what to keep in mind for fonts and export.

## A Persian-labeled DAG

Persian node names work like any other string. This course-prerequisite DAG uses Persian names for the nodes and for the attribute that groups them:

```python
import aryagraph as ag

courses = ag.DAG([
    ("ریاضی ۱", "ریاضی ۲"),
    ("ریاضی ۲", "آمار و احتمال"),
    ("مبانی برنامه‌نویسی", "ساختمان داده"),
    ("ساختمان داده", "طراحی الگوریتم"),
    ("ریاضی گسسته", "طراحی الگوریتم"),
    ("آمار و احتمال", "یادگیری ماشین"),
    ("طراحی الگوریتم", "یادگیری ماشین"),
    ("ساختمان داده", "پایگاه داده"),
])
for course in courses:
    is_math = "ریاضی" in course or "آمار" in course
    courses.nodes[course]["گروه"] = "ریاضی" if is_math else "کامپیوتر"

fig = ag.draw(
    courses,
    node_color=ag.by("گروه", counts=False),
    layout_options={"orientation": "RL"},
    title="پیش‌نیازهای درسی",
    subtitle="Course prerequisites with Persian labels, flowing right to left",
)
fig.save("courses_fa.svg")
print(courses.topological_order()[0], "|", courses.sinks())
# ریاضی ۱ | ['یادگیری ماشین', 'پایگاه داده']
```

```{figure} ../_static/generated/guide_core/languages_persian.png
:alt: Layered DAG of nine Persian course names in rounded boxes, flowing from right to left, with math courses in blue and computer courses in orange.
:width: 90%

A Persian-labeled DAG laid out with `orientation="RL"`, so prerequisites sit on the right and the flow reads right to left, the reading direction of the labels.
```

Three details in this example are worth noting:

- **Box sizes follow the text.** Each box is as wide as its estimated label, so "مبانی برنامه‌نویسی" gets a wider box than "ریاضی ۱".
- **The layout direction is your choice.** The direction of the text and the direction of the layout are independent. `orientation="RL"` (in `layout_options`) mirrors the layered layout so it reads right to left; the default `"TB"` works equally well for Persian labels.
- **Legend counts are switched off.** `counts=False` drops the per-category counts. In this version, a count drawn after a right-to-left category name can overlap the name, so leave the counts out when category values are in a right-to-left script.

## Many scripts in one drawing

Scripts can be mixed freely, in one drawing and within one label:

```python
hello = {
    "English": "Hello", "فارسی": "سلام", "العربية": "مرحبا", "עברית": "שלום",
    "中文": "你好", "日本語": "こんにちは", "한국어": "안녕하세요",
    "Ελληνικά": "Γειά σου", "हिन्दी": "नमस्ते",
}
greetings = ag.Graph([("🌐", language) for language in hello])
fig = ag.draw(
    greetings,
    layout="radial",
    scale=0.6,   # radial layouts are sized in pixels; 0.6 draws the ring at 60% of its radius
    labels=lambda n: n if n == "🌐" else f"{n}: {hello[n]}",
    title="Hello in nine languages",
)
fig.save("hello.svg")
print(sum(ag.style.is_rtl(f"{k}: {v}") for k, v in hello.items()))   # 3
```

```{figure} ../_static/generated/guide_core/languages_hello.png
:alt: Radial star of nine nodes around a globe emoji, labeled with the word hello in English, Persian, Arabic, Hebrew, Chinese, Japanese, Korean, Greek and Hindi.
:width: 100%

Nine languages in eight scripts. Labels to the right of their nodes start next to the node whichever direction they read in.
```

## How text is measured

Layout decisions happen before any text is drawn: box sizes, label collision tests, legend widths and the size of the canvas all need text widths in advance. AryaGraph estimates them from per-character advance widths, without a font engine, so computing a drawing needs no font files and gives the same geometry on every machine.

| Characters | Estimated width |
|---|---|
| ASCII | Helvetica advance widths (for example 0.556 em for "a", 0.278 em for a space) |
| accented Latin letters | the width of the base letter |
| Arabic and Persian letters | 0.52 em each; joined forms average narrower than Latin capitals |
| Arabic diacritics (harakat) and other combining marks | zero |
| zero-width non-joiner and joiner, direction marks | zero |
| Hebrew characters | 0.56 em |
| CJK ideographs, kana, Hangul, full-width forms | 1 em |
| emoji | 1.2 em |
| anything else | 0.6 em |

The total is multiplied by 1.04, since common interface fonts run slightly wider than Helvetica, and by 1.03 or 1.065 for medium and bold weights. The estimates are meant to err on the wide side, so a label fits inside the box sized for it. {py:func}`ag.style.text_width() <aryagraph.style.text.text_width>` exposes the estimate in pixels:

```python
for text in ("Hello world", "سلام دنیا", "שלום עולם", "你好世界", "안녕하세요"):
    print(f"{ag.style.text_width(text, 12):5.1f}  {text}")
#  61.7  Hello world
#  55.4  سلام دنیا
#  59.4  שלום עולם
#  49.9  你好世界
#  62.4  안녕하세요
```

Labels inside boxes wrap when they are wider than `label_max_width` (160 px by default). Breaks go at spaces first, then after `_`, `-`, `/` or `.`, and at lowercase-to-uppercase boundaries; text without spaces, such as a Chinese phrase, is broken between characters when it does not fit. The zero-width non-joiner in Persian words such as "برنامه‌نویسی" is kept, so the word is not split there:

```python
lines = ag.style.wrap("مبانی برنامه‌نویسی پیشرفته برای دانشجویان مهندسی", 120, 11.5)
print(len(lines), "‌" in lines[0])   # 3 True
```

## Direction

{py:func}`ag.style.is_rtl() <aryagraph.style.text.is_rtl>` reports whether a string contains characters from a right-to-left script (the Hebrew and Arabic Unicode blocks and their presentation forms). A label that contains any such character is written with `direction="rtl"` in the SVG, and its anchor is mirrored so the label keeps the position the placer computed: a label to the right of its node still starts next to the node.

```python
print(ag.style.is_rtl("ریاضی ۱"), ag.style.is_rtl("CS101"), ag.style.is_rtl("CS101 مبانی"))
# True False True
```

Within a label, the order of mixed runs (Persian words with Latin letters or digits) and the joining of Arabic-script letters are handled by the browser or SVG renderer that displays the file, following the Unicode bidirectional algorithm. Persian digits (۰ to ۹) and Latin digits both work.

Titles, subtitles, captions and legend entries are placed from the left edge of the figure in every script; the renderer orders the text inside each of them.

## Ids and labels

Nodes are matched by their ids, so any string can be an id. When your data already has short ASCII ids, keep them as ids and show the translated names as labels; searches in the [interactive view](interactive.md) match both the label and the id:

```python
catalog = ag.DAG([("MATH101", "MATH102"), ("MATH102", "STAT220")])
persian_names = {"MATH101": "ریاضی ۱", "MATH102": "ریاضی ۲", "STAT220": "آمار و احتمال"}
fig = ag.draw(catalog, labels=persian_names)
print([m.text for m in fig.scene.nodes])
# ['ریاضی ۱', 'ریاضی ۲', 'آمار و احتمال']
```

Setting a `label` attribute on each node has the same effect without passing `labels=`.

## Fonts and export

The default font list names the common system interface fonts first, then "Noto Sans" and "Vazirmatn", then the generic `sans-serif`. A browser uses the first font in the list that is installed and falls back per character, so on most systems Persian, Arabic, Hebrew and CJK labels render without extra setup. To choose a font, derive a theme:

```python
persian_theme = ag.get_theme("light").with_(name="fa", font='Vazirmatn, "Noto Sans Arabic", Tahoma, sans-serif')
fig = ag.draw(courses, theme=persian_theme, title="پیش‌نیازهای درسی")
fig.save("courses_vazirmatn.svg")
```

The font has to be available where the file is displayed or rendered:

- **SVG and HTML** are displayed with the viewer's fonts. For the web, load the font on your page (for example with a `@font-face` rule) or rely on system fonts.
- **PNG and PDF** made with the browser backend use the fonts installed on the machine that exports them, with the same shaping as the interactive view; the figures on this page were made that way. See [Exporting](exporting.md#png-and-pdf-backends).

Width estimates do not depend on the font you choose. A font much wider than the interface fonts the estimates assume can make labels run closer to box edges; raising `label_max_width` or choosing a slightly narrower font keeps them clear.

## Next steps

- [Drawing](drawing.md#label-placement): how labels are placed and when they are hidden.
- [Styling](styling.md): themes and fonts.
- [Interactive HTML](interactive.md): search and tooltips for multilingual graphs.
