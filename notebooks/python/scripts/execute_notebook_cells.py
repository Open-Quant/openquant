from __future__ import annotations

import argparse
import contextlib
import io
import json
import traceback
from pathlib import Path


def execute_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    g: dict[str, object] = {"__name__": "__main__"}
    exec_count = 1

    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        code = "".join(cell.get("source", []))
        stdout = io.StringIO()
        outputs = []
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(code, f"{path.name}:cell-{idx}", "exec"), g, g)
        except Exception:
            tb = traceback.format_exc()
            outputs.append({
                "output_type": "error",
                "ename": "ExecutionError",
                "evalue": f"cell {idx}",
                "traceback": tb.splitlines(),
            })
            cell["execution_count"] = exec_count
            cell["outputs"] = outputs
            path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
            raise RuntimeError(f"Notebook execution failed at cell {idx}\n{tb}")

        text = stdout.getvalue()
        if text:
            outputs.append({"output_type": "stream", "name": "stdout", "text": text})
        cell["execution_count"] = exec_count
        cell["outputs"] = outputs
        exec_count += 1

    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute code cells in a .ipynb via plain Python exec")
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    execute_notebook(args.notebook)
    print(f"executed: {args.notebook}")


if __name__ == "__main__":
    main()
