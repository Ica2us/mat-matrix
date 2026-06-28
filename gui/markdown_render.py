"""
简易 Markdown → HTML 渲染器。

支持子集:
  # heading, ## heading, ### heading
  **bold**, *italic*
  `inline code`
  - unordered list, 1. ordered list
  [link](url), ![image](url)
  空行分隔段落
"""
from __future__ import annotations

import re
from html import escape as _escape
from typing import Match


def _repl_bold(m: Match[str]) -> str:
    return f"<strong>{m.group(1)}</strong>"

def _repl_italic(m: Match[str]) -> str:
    return f"<em>{m.group(1)}</em>"

def _repl_code(m: Match[str]) -> str:
    return f"<code>{m.group(1)}</code>"

def _repl_link(m: Match[str]) -> str:
    return f'<a href="{_escape(m.group(2), quote=True)}">{_escape(m.group(1))}</a>'

def _repl_image(m: Match[str]) -> str:
    return f'<img src="{_escape(m.group(2), quote=True)}" alt="{_escape(m.group(1))}" style="max-width:100%">'


def md_to_html(text: str) -> str:
    """将 Markdown 文本转换为 HTML 片段。"""
    # 先 HTML 转义（避免注入），再还原 markdown 语法标记
    text = _escape(text)

    # 行级规则（从高精度到低精度）
    text = re.sub(r"!\[([^\]]+)\]\(([^)]+)\)", _repl_image, text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _repl_link, text)

    # 行内样式
    text = re.sub(r"\*\*(.+?)\*\*", _repl_bold, text)
    text = re.sub(r"\*(.+?)\*", _repl_italic, text)
    text = re.sub(r"`(.+?)`", _repl_code, text)

    # 标题
    text = re.sub(r"^### (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.+)$", r"<h2>\1</h2>", text, flags=re.MULTILINE)
    text = re.sub(r"^# (.+)$", r"<h1>\1</h1>", text, flags=re.MULTILINE)

    # 列表
    text = re.sub(r"^- (.+)$", r"<li>\1</li>", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\. (.+)$", r"<li>\1</li>", text, flags=re.MULTILINE)

    # 段落（双换行 → </p><p>）
    paragraphs = text.split("\n\n")
    parts = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        # 标题和列表项不包裹 <p>
        if p.startswith(("<h", "<li")):
            parts.append(p)
        else:
            # 内联换行变 <br>
            p = p.replace("\n", "<br>")
            parts.append(f"<p>{p}</p>")

    html_body = "\n".join(parts)
    return f"<div style='font-size:13px; line-height:1.6'>{html_body}</div>"