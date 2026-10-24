# ADDED FILE: every line in this file is new (it is not part of the diffusers codebase).
"""Implementation evidence for the report: original vs modified training script.

  python make_diff.py --out_dir figures

Compares original/train_instruct_pix2pix.py (diffusers v0.41.0, unchanged) with train_instruct_pix2pix.py and
writes changed_lines.md (each change as original lines next to modified lines, with line numbers) and
implementation_diff.html (full side-by-side view). It also fails if any changed or added line is missing
its "# MODIFIED" / "# ADDED" marker.
"""
import argparse
import difflib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="figures")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    orig = open(os.path.join(HERE, "original", "train_instruct_pix2pix.py"), encoding="utf8").read().splitlines()
    new = open(os.path.join(HERE, "train_instruct_pix2pix.py"), encoding="utf8").read().splitlines()
    ops = [op for op in difflib.SequenceMatcher(None, orig, new, autojunk=False).get_opcodes() if op[0] != "equal"]

    unmarked, md = [], ["# Changes to train_instruct_pix2pix.py (diffusers v0.41.0 example)\n"]
    n_added = n_modified = 0
    for k, (tag, i1, i2, j1, j2) in enumerate(ops, start=1):
        for j in range(j1, j2):
            if "# ADDED" in new[j]:
                n_added += 1
            elif "# MODIFIED" in new[j]:
                n_modified += 1
            else:
                unmarked.append(f"line {j + 1}: {new[j]}")
        o_range = f"lines {i1 + 1}-{i2}" if i2 > i1 else f"(insert after line {i1})"
        m_range = f"lines {j1 + 1}-{j2}" if j2 > j1 else "(removed)"
        md.append(f"## Change {k} ({tag})\n\n**Original**, {o_range}:\n\n```python\n"
                  + ("\n".join(f"{i + 1:4d}  {orig[i]}" for i in range(i1, i2)) or "    (nothing)")
                  + f"\n```\n\n**Modified**, {m_range}:\n\n```python\n"
                  + "\n".join(f"{j + 1:4d}  {new[j]}" for j in range(j1, j2)) + "\n```\n")
    md.insert(1, f"{len(ops)} change blocks; {n_modified} lines marked `# MODIFIED`, {n_added} lines marked `# ADDED`.\n")
    with open(os.path.join(args.out_dir, "changed_lines.md"), "w", encoding="utf8") as f:
        f.write("\n".join(md))
    html = difflib.HtmlDiff(wrapcolumn=95).make_file(orig, new, "original (diffusers v0.41.0)", "modified", context=True, numlines=2)
    with open(os.path.join(args.out_dir, "implementation_diff.html"), "w", encoding="utf8") as f:
        f.write(html)
    print(f"{len(ops)} change blocks, {n_modified} MODIFIED, {n_added} ADDED lines")
    if unmarked:
        print("Unmarked changed lines:\n  " + "\n  ".join(unmarked))
        sys.exit(1)


if __name__ == "__main__":
    main()
