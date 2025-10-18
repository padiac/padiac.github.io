import re
from pathlib import Path

from latex2mathml.converter import convert as latex_to_mathml
from markdown import markdown

ROOT = Path(__file__).resolve().parent
NOTES_DIR = ROOT / 'notes'
OUTPUT_DIR = ROOT / 'public-notes'

BLOCK_PATTERN = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
INLINE_PATTERN = re.compile(r'(?<!\$)\$(.+?)\$(?!\$)', re.DOTALL)


def to_mathml(latex: str, inline: bool) -> str:
    mathml = latex_to_mathml(latex)
    wrapper = 'span' if inline else 'div'
    cls = 'math-inline' if inline else 'math-block'
    return f'<{wrapper} class="{cls}">{mathml}</{wrapper}>'


def render_markdown(source: str) -> str:
    placeholders = {}
    counter = {'block': 0, 'inline': 0}

    def replace_block(match):
        counter['block'] += 1
        key = f'@@MATH_BLOCK_{counter["block"]}@@'
        placeholders[key] = to_mathml(match.group(1).strip(), inline=False)
        return key

    def replace_inline(match):
        counter['inline'] += 1
        key = f'@@MATH_INLINE_{counter["inline"]}@@'
        placeholders[key] = to_mathml(match.group(1).strip(), inline=True)
        return key

    templated = BLOCK_PATTERN.sub(replace_block, source)
    templated = INLINE_PATTERN.sub(replace_inline, templated)

    html = markdown(templated, extensions=['extra', 'tables', 'sane_lists'])

    for key, value in placeholders.items():
        if key.startswith('@@MATH_BLOCK_'):
            html = html.replace(f'<p>{key}</p>', value)
        html = html.replace(key, value)

    return html


def main():
    if not NOTES_DIR.exists():
        raise SystemExit('notes directory not found')
    OUTPUT_DIR.mkdir(exist_ok=True)

    for path in sorted(NOTES_DIR.glob('*.md')):
        slug = path.stem
        html = render_markdown(path.read_text(encoding='utf-8'))
        target = OUTPUT_DIR / f'{slug}.html'
        target.write_text(html, encoding='utf-8')
        print(f'Rendered {target.relative_to(ROOT)}')

if __name__ == '__main__':
    main()
