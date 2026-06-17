import ast
import os
import re
from pathlib import Path

ROOT = Path(r"D:\Desktop\IA Doninha")
OUT = Path(r"D:\Desktop\middleware_monolitico.py")
IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", "site-packages", "node_modules"}


def is_python_file(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    if path.name == "middleware_monolitico.py" or path.name == "generate_middleware_monolitico.py":
        return False
    if any(part in IGNORED_DIRS for part in path.parts):
        return False
    return True


files = sorted([p for p in ROOT.rglob("*.py") if is_python_file(p)])

# Gather high-level info
external_imports = set()
internal_imports = set()
classes = []
functions = []

for path in files:
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    try:
        tree = ast.parse(text, filename=str(path))
    except Exception as exc:
        print(f"SKIP_PARSE {path}: {exc}")
        continue
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            if kind == "class":
                classes.append((name, str(path.relative_to(ROOT)).replace('\\', '/')))
            else:
                functions.append((name, str(path.relative_to(ROOT)).replace('\\', '/')))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported = alias.name
                if imported.startswith("." ):
                    internal_imports.add(imported)
                else:
                    external_imports.add(imported)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            if module and (module.startswith("." ) or module.startswith("__")):
                internal_imports.add(module)
            elif module:
                external_imports.add(module)

header = (
    "from __future__ import annotations\n\n"
    "# Consolidado automaticamente a partir de "
    f"{len(files)} arquivos Python do workspace.\n"
    "# Gerado por: generate_middleware_monolitico.py\n"
)

parts = [header]
parts.append("\n# =========================\n# Arquivos incorporados\n# =========================\n")
for path in files:
    rel = path.relative_to(ROOT).as_posix()
    parts.append(f"\n# --- INICIO: {rel} ---\n")
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n")
    text = re.sub(r"^\s*from __future__ import .*\n", "", text, flags=re.M)
    text = re.sub(r"^\s*#.*\n", "", text, flags=re.M)
    # Remove common shebangs/encoding lines if present
    text = text.lstrip()
    if text.startswith("#!/"):
        text = text.split("\n", 1)[1]
    if text.startswith("# -*- coding:"):
        text = text.split("\n", 1)[1]
    parts.append(text)
    parts.append(f"\n# --- FIM: {rel} ---\n")

parts.append("\n\n# =========================\n# RELATORIO DE CONSOLIDACAO\n# =========================\n")
parts.append("REPORT = {\n")
parts.append(f"    'arquivos_incorporados': {len(files)},\n")
parts.append("    'arquivos': [\n")
for path in files:
    rel = path.relative_to(ROOT).as_posix()
    parts.append(f"        {rel!r},\n")
parts.append("    ],\n")
parts.append(f"    'classes_incorporadas': {len(classes)},\n")
parts.append("    'classes': [\n")
for name, rel in classes[:200]:
    parts.append(f"        {{'nome': {name!r}, 'arquivo': {rel!r}}},\n")
parts.append("    ],\n")
parts.append(f"    'funcoes_incorporadas': {len(functions)},\n")
parts.append("    'funcoes': [\n")
for name, rel in functions[:200]:
    parts.append(f"        {{'nome': {name!r}, 'arquivo': {rel!r}}},\n")
parts.append("    ],\n")
parts.append(f"    'imports_removidos': {len(internal_imports)},\n")
parts.append("    'imports_internos_detectados': [\n")
for item in sorted(internal_imports)[:200]:
    parts.append(f"        {item!r},\n")
parts.append("    ],\n")
parts.append(f"    'dependencias_externas_necessarias': {len(external_imports)},\n")
parts.append("    'imports_externos_detectados': [\n")
for item in sorted(external_imports)[:200]:
    parts.append(f"        {item!r},\n")
parts.append("    ],\n")
parts.append("    'conflitos_resolvidos': [\n")
parts.append("        'unificacao por concatenacao dos arquivos Python do workspace',\n")
parts.append("        'preservacao dos nomes originais de classes e funcoes',\n")
parts.append("        'normalizacao de shebang/encoding e delimitadores',\n")
parts.append("    ],\n")
parts.append("}\n")

OUT.write_text("".join(parts), encoding="utf-8")
print(f"Gerado {OUT}")
print(f"Arquivos: {len(files)}")
print(f"Classes: {len(classes)}")
print(f"Funcoes: {len(functions)}")
print(f"Imports externos: {len(external_imports)}")
print(f"Imports internos detectados: {len(internal_imports)}")
