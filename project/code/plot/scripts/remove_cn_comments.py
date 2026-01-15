import os
import re
import sys
from typing import List, Tuple, Optional


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


# Map file extensions to comment syntax
COMMENT_MAP = {
    # Hash comment languages
    'py': {'line': ['#'], 'block': []},
    'rb': {'line': ['#'], 'block': []},
    'sh': {'line': ['#'], 'block': []},
    'bash': {'line': ['#'], 'block': []},
    'zsh': {'line': ['#'], 'block': []},
    'yaml': {'line': ['#'], 'block': []},
    'yml': {'line': ['#'], 'block': []},
    'toml': {'line': ['#'], 'block': []},
    'ps1': {'line': ['#'], 'block': []},
    'psm1': {'line': ['#'], 'block': []},

    # C/JS-style
    'js': {'line': ['//'], 'block': [('/*', '*/')]},
    'jsx': {'line': ['//'], 'block': [('/*', '*/')]},
    'ts': {'line': ['//'], 'block': [('/*', '*/')]},
    'tsx': {'line': ['//'], 'block': [('/*', '*/')]},
    'java': {'line': ['//'], 'block': [('/*', '*/')]},
    'c': {'line': ['//'], 'block': [('/*', '*/')]},
    'cc': {'line': ['//'], 'block': [('/*', '*/')]},
    'cpp': {'line': ['//'], 'block': [('/*', '*/')]},
    'cxx': {'line': ['//'], 'block': [('/*', '*/')]},
    'h': {'line': ['//'], 'block': [('/*', '*/')]},
    'hh': {'line': ['//'], 'block': [('/*', '*/')]},
    'hpp': {'line': ['//'], 'block': [('/*', '*/')]},
    'go': {'line': ['//'], 'block': [('/*', '*/')]},
    'cs': {'line': ['//'], 'block': [('/*', '*/')]},
    'swift': {'line': ['//'], 'block': [('/*', '*/')]},
    'kt': {'line': ['//'], 'block': [('/*', '*/')]},
    'kts': {'line': ['//'], 'block': [('/*', '*/')]},
    'scala': {'line': ['//'], 'block': [('/*', '*/')]},
    'rs': {'line': ['//'], 'block': [('/*', '*/')]},
    'php': {'line': ['//', '#'], 'block': [('/*', '*/')]},

    # Styles
    'css': {'line': [], 'block': [('/*', '*/')]},
    'scss': {'line': ['//'], 'block': [('/*', '*/')]},
    'less': {'line': ['//'], 'block': [('/*', '*/')]},

    # Markup
    'html': {'line': [], 'block': [('<!--', '-->')]},
    'htm': {'line': [], 'block': [('<!--', '-->')]},
    'xml': {'line': [], 'block': [('<!--', '-->')]},
    'svg': {'line': [], 'block': [('<!--', '-->')]},
    'vue': {'line': [], 'block': [('<!--', '-->')]},
    'svelte': {'line': [], 'block': [('<!--', '-->')]},

    # INI-like
    'ini': {'line': [';', '#'], 'block': []},
    'cfg': {'line': [';', '#'], 'block': []},
    'properties': {'line': ['#', '!'], 'block': []},

    # Batch (only treat full-line comments)
    'bat': {'line': ['::', 'REM'], 'block': []},
    'cmd': {'line': ['::', 'REM'], 'block': []},
}


TEXT_EXTS = set(COMMENT_MAP.keys())


def looks_binary(data: bytes) -> bool:
    if b"\0" in data:
        return True
    # Heuristic: high ratio of non-text bytes
    text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)))
    nontext = data.translate(None, text_chars)
    return float(len(nontext)) / max(1, len(data)) > 0.30


def read_text_safe(path: str) -> Optional[str]:
    try:
        with open(path, 'rb') as f:
            data = f.read()
        if looks_binary(data):
            return None
        return data.decode('utf-8', errors='ignore')
    except Exception:
        return None


def contains_cjk(s: str) -> bool:
    return bool(CJK_RE.search(s))


def strip_line_comment_if_cn(line: str, extsyntax: dict, ext: str) -> Tuple[str, int]:
    original = line
    s = line.rstrip('\n\r')
    removed = 0

    # Batch special handling: only full-line comments
    if ext in ('bat', 'cmd'):
        st = s.lstrip()
        if st.startswith('::'):
            if contains_cjk(st[2:]):
                return ('', 1)
        if st.upper().startswith('REM'):
            rest = st[3:]
            if contains_cjk(rest):
                return ('', 1)
        return (original, 0)

    # General line comment handling
    line_markers: List[str] = extsyntax.get('line', [])

    # Simple quote-aware scan to ignore markers inside quotes
    def find_marker_outside_quotes(text: str, marker: str) -> int:
        i = 0
        in_single = False
        in_double = False
        in_backtick = False
        while i <= len(text) - len(marker):
            ch = text[i]
            if ch == '\\':
                i += 2
                continue
            if not in_double and not in_backtick and ch == '\'' and (i == 0 or text[i-1] != 'r'):
                in_single = not in_single
                i += 1
                continue
            if not in_single and not in_backtick and ch == '"':
                in_double = not in_double
                i += 1
                continue
            if not in_single and not in_double and ch == '`':
                in_backtick = not in_backtick
                i += 1
                continue
            if not in_single and not in_double and not in_backtick:
                if text.startswith(marker, i):
                    # avoid http:// like sequences for //
                    if marker == '//' and i >= 1 and text[i-1] == ':':
                        i += 2
                        continue
                    return i
            i += 1
        return -1

    best_pos = None
    best_marker = None
    for m in line_markers:
        pos = find_marker_outside_quotes(s, m)
        if pos != -1 and (best_pos is None or pos < best_pos):
            best_pos = pos
            best_marker = m

    if best_pos is not None:
        comment_text = s[best_pos + len(best_marker):]
        if contains_cjk(comment_text):
            prefix = s[:best_pos].rstrip()
            if prefix == '':
                return ('', 1)
            else:
                return (prefix + ('\n' if original.endswith('\n') else ''), 1)

    return (original, removed)


def remove_block_comments_with_cn(text: str, blocks: List[Tuple[str, str]]) -> Tuple[str, int]:
    removed = 0
    out = text
    for start, end in blocks:
        # Build non-greedy pattern
        pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)

        def repl(match):
            nonlocal removed
            inner = match.group(0)[len(start):-len(end)] if len(end) else match.group(0)[len(start):]
            if contains_cjk(inner):
                removed += 1
                return ''
            return match.group(0)

        out = pattern.sub(repl, out)
    return out, removed


def process_file(path: str, dry_run: bool = True) -> Tuple[int, int]:
    ext = os.path.splitext(path)[1].lstrip('.').lower()
    syntax = COMMENT_MAP.get(ext)
    if not syntax:
        return (0, 0)

    content = read_text_safe(path)
    if content is None:
        return (0, 0)

    total_removed = 0

    # First, remove block comments containing CJK
    new_content, removed_blocks = remove_block_comments_with_cn(content, syntax.get('block', []))
    total_removed += removed_blocks

    # Then process line comments
    new_lines = []
    line_removed_count = 0
    for line in new_content.splitlines(keepends=True):
        new_line, removed = strip_line_comment_if_cn(line, syntax, ext)
        line_removed_count += removed
        # Preserve original newline if we dropped line
        if new_line == '' and line.endswith('\n'):
            # drop entire line (comment-only line)
            continue
        new_lines.append(new_line)

    total_removed += line_removed_count
    final_text = ''.join(new_lines)

    if not dry_run and final_text != content:
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write(final_text)

    return (1 if final_text != content else 0, total_removed)


def iter_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        # skip common vendor/build dirs
        skip_dirs = {'.git', '.hg', '.svn', 'node_modules', 'dist', 'build', '__pycache__', '.venv', 'venv', '.idea', '.vscode'}
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for filename in filenames:
            ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
            if ext in TEXT_EXTS:
                yield os.path.join(dirpath, filename)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Remove Chinese comments from code files.')
    parser.add_argument('--apply', action='store_true', help='Apply changes (otherwise dry-run).')
    parser.add_argument('--root', default='.', help='Root directory to scan.')
    args = parser.parse_args()

    changed_files = 0
    total_removed = 0
    touched = 0
    report = []

    for path in iter_files(args.root):
        touched += 1
        changed, removed = process_file(path, dry_run=not args.apply)
        total_removed += removed
        if changed:
            changed_files += 1
            report.append((path, removed))

    mode = 'APPLY' if args.apply else 'DRY-RUN'
    print(f'[{mode}] scanned_files={touched} changed_files={changed_files} removed_items={total_removed}')
    if report:
        for p, r in sorted(report, key=lambda x: (-x[1], x[0]))[:200]:
            print(f'  {p} -> removed {r} comment segment(s)')
        if len(report) > 200:
            print(f'  ... and {len(report)-200} more files')


if __name__ == '__main__':
    main()

