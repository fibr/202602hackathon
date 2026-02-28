"""Compile-check all scripts to catch syntax errors and import shadowing.

Python's compiler detects when a local import/assignment shadows a name that
was used earlier in the same scope (UnboundLocalError at runtime). Compiling
each script's AST and checking for this pattern catches these bugs statically.
"""

import ast
import os
import sys
import py_compile
import glob

import pytest


SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'scripts')
SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'src')


def _all_py_files():
    """Collect all .py files in scripts/ and src/."""
    files = []
    for d in [SCRIPTS_DIR, SRC_DIR]:
        for path in glob.glob(os.path.join(d, '**', '*.py'), recursive=True):
            files.append(path)
    return files


@pytest.mark.parametrize("filepath", _all_py_files(),
                         ids=lambda p: os.path.relpath(p))
def test_syntax_valid(filepath):
    """Each .py file must be valid Python syntax."""
    py_compile.compile(filepath, doraise=True)


def _find_local_import_shadows(filepath):
    """Find cases where a name is used before a local import assigns it.

    Returns list of (name, use_line, import_line) tuples.
    """
    with open(filepath) as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return []  # syntax errors caught by test_syntax_valid

    issues = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Collect all names that are locally imported (from X import Y)
        local_imports = {}  # name -> line number
        for child in ast.walk(node):
            if isinstance(child, ast.ImportFrom):
                # Only count imports that are inside this function
                for alias in child.names:
                    name = alias.asname or alias.name
                    local_imports[name] = child.lineno

        if not local_imports:
            continue

        # Check if any of these names are used before the import line
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in local_imports:
                import_line = local_imports[child.id]
                if child.lineno < import_line:
                    issues.append((child.id, child.lineno, import_line))

    return issues


@pytest.mark.parametrize("filepath", _all_py_files(),
                         ids=lambda p: os.path.relpath(p))
def test_no_local_import_shadows(filepath):
    """No function should use a name before a local import assigns it."""
    issues = _find_local_import_shadows(filepath)
    if issues:
        msgs = []
        for name, use_line, import_line in issues:
            msgs.append(
                f"  '{name}' used on line {use_line} but locally imported "
                f"on line {import_line}")
        rel = os.path.relpath(filepath)
        pytest.fail(f"{rel} has local import shadowing:\n" + "\n".join(msgs))
