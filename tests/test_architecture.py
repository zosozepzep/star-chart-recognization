"""Transitive source import guard for the inference and star confirmation paths."""
import ast
from pathlib import Path


def test_inference_has_no_transitive_truth_dependency():
    root = Path(__file__).resolve().parents[1]
    graph = {}
    for path in (root / "src").rglob("*.py"):
        module = ".".join(path.relative_to(root).with_suffix("").parts).removesuffix(".__init__")
        imports = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imports.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parent = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
                    base = ".".join(parent[:len(parent) - node.level + 1] + ([base] if base else []))
                imports.add(base)
                imports.update(base + "." + a.name for a in node.names)
        graph[module] = imports
    protected = {"src.pipeline", "src.analysis.star_catalog", "src.validate.repeatability"}
    protected.update(m for m in graph if m.startswith(("src.detect", "src.register", "src.target")))
    for entry in protected:
        seen, pending = set(), [(entry, [entry])]
        while pending:
            module, chain = pending.pop()
            assert module != "src.validate.truth", " -> ".join(chain)
            if module not in seen:
                seen.add(module)
                pending.extend((dep, chain + [dep]) for dep in graph.get(module, ()) if dep in graph)
