"""Compact table syntax without selecting, correcting, or inventing source cells.

Original text, source hashes and geometry remain in the citation catalog. This
projection is only an input representation; images remain authoritative when an
extractor transcription disagrees with the page.
"""
import json
import re
from html.parser import HTMLParser


class _Cells(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []; self.row = None; self.cell = None; self.stack = []

    def handle_starttag(self, tag, attrs):
        if tag not in ('table', 'thead', 'tbody', 'tfoot', 'tr', 'td', 'th'):
            raise ValueError('Unsupported table markup')
        if tag in ('td', 'th'):
            if self.row is None or self.cell is not None: raise ValueError('Bad cell topology')
            spans = {}
            for key, value in attrs:
                if key not in ('rowspan', 'colspan') or key in spans:
                    raise ValueError('Non-span cell attributes')
                if value is None or not value.isdecimal() or not 1 <= int(value) <= 100:
                    raise ValueError('Invalid span')
                spans[key] = int(value)
            self.cell = {'text': '', **spans}
        elif attrs:
            raise ValueError('Unsupported table attributes')
        if tag == 'tr':
            if self.row is not None: raise ValueError('Nested row')
            self.row = []
        if tag == 'table' and self.stack: raise ValueError('Nested table')
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack.pop() != tag: raise ValueError('Unbalanced table')
        if tag in ('td', 'th'):
            if self.cell is None or self.row is None: raise ValueError('Bad cell close')
            self.row.append(self.cell['text'] if set(self.cell) == {'text'} else self.cell)
            self.cell = None
        if tag == 'tr':
            if self.row is None: raise ValueError('Bad row close')
            self.rows.append(self.row); self.row = None

    def handle_data(self, data):
        if self.cell is not None: self.cell['text'] += data
        elif data.strip(): raise ValueError('Unassigned table text')

    def handle_comment(self, data): raise ValueError('Commented table')


def project_source_text(text):
    """Return compact syntax and projection count; unsupported markup stays exact."""
    count = 0
    def replace(match):
        nonlocal count
        raw = match.group(0)
        try:
            parser = _Cells(); parser.feed(raw); parser.close()
            if parser.stack or not parser.rows: return raw
            compact = 'TABLE_CELL_ROWS ' + json.dumps(parser.rows, ensure_ascii=False, separators=(',', ':'))
            if len(compact) >= len(raw): return raw
        except (ValueError, AssertionError): return raw
        count += 1
        return compact
    projected = re.sub(r'<table(?:\s[^>]*)?>.*?</table\s*>', replace, text, flags=re.I | re.S)
    return projected, count
