#!/usr/bin/env python3
"""Learning helper: compare baseline vs paged-kv, and inspect compile-stage IR.

Two main modes (as suggested for studying this repo):

1. ``diff``   – side-by-side source diff of the four meaningful blocks:
                attention / export-spec / pipeline / runtime.
2. ``stages`` – export a model, then ``show()`` (or dump) Relax IR after
                export → LegalizeOps → FuseOps+FuseTIR → DLight.

Examples:
    python3 learn_compare.py diff
    python3 learn_compare.py diff --block attention
    python3 learn_compare.py stages --model baseline          # 默认：对比表 + 阶段差分（不刷全文）
    python3 learn_compare.py stages --model baseline --dump-ir # 额外导出完整 Relax IR
    python3 learn_compare.py stages --model paged --func prefill
    python3 learn_compare.py stages --model baseline --show-tir matmul
    python3 learn_compare.py stages --model both --out /tmp/ir_dump
    python3 learn_compare.py all --out /tmp/learn
"""

from __future__ import annotations

import argparse
import ast
import difflib
import os
import sys
import textwrap
from pathlib import Path

# CUDA 13.2 + default NVRTC may miss headers; match the tutorial scripts.
os.environ.setdefault("TVM_CUDA_COMPILE_MODE", "nvcc")

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "simple_llm_decoder.py"
PAGED = ROOT / "paged_kv_cache" / "decoder_paged_kv.py"

BLOCKS = ("attention", "export", "pipeline", "runtime")
STAGES = ("export", "legalize", "fuse", "dlight")


# ---------------------------------------------------------------------------
# Source extraction (AST line ranges)
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _lines(src: str, start: int, end: int) -> str:
    """1-based inclusive line slice."""
    rows = src.splitlines(keepends=True)
    return "".join(rows[start - 1 : end])


def _find_class(tree: ast.AST, name: str) -> ast.ClassDef:
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise LookupError(f"class {name!r} not found")


def _find_func(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise LookupError(f"function {name!r} not found")


def _find_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise LookupError(f"method {cls.name}.{name!r} not found")


def _find_call_in_func(func: ast.AST, qual_suffix: str) -> ast.Call:
    """Find a Call whose callee text endswith qual_suffix (e.g. 'export_tvm')."""
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        try:
            callee = ast.unparse(node.func)
        except Exception:
            continue
        if callee.endswith(qual_suffix):
            return node
    raise LookupError(f"call ending with {qual_suffix!r} not found")


def _assign_or_expr_span(func: ast.FunctionDef, call: ast.Call) -> tuple[int, int]:
    """Prefer the enclosing Assign / Expr statement span for a Call."""
    for node in func.body:
        for child in ast.walk(node):
            if child is call or (isinstance(child, ast.Call) and child.lineno == call.lineno):
                return node.lineno, node.end_lineno or node.lineno
        # match by containing the call's lineno
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr))
            and node.lineno <= call.lineno
            and (node.end_lineno or node.lineno) >= call.lineno
        ):
            return node.lineno, node.end_lineno or node.lineno
    return call.lineno, call.end_lineno or call.lineno


def extract_baseline_blocks(src: str) -> dict[str, str]:
    tree = ast.parse(src)
    attn = _find_class(tree, "CausalSelfAttention")
    main = _find_func(tree, "main")

    export_call = _find_call_in_func(main, "export_tvm")
    export_lo, export_hi = _assign_or_expr_span(main, export_call)

    # Pipeline: Sequential(...) applied in main
    seq_call = _find_call_in_func(main, "Sequential")
    # Prefer the outer `mod = Sequential(...)(mod)` assign if present.
    pipe_lo, pipe_hi = _assign_or_expr_span(main, seq_call)
    # If Sequential is nested inside another Call (Sequential(...)(mod)), widen.
    for node in main.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            text = ast.unparse(node)
        except Exception:
            continue
        if "Sequential" in text and "get_pipeline" in text:
            pipe_lo, pipe_hi = node.lineno, node.end_lineno or node.lineno
            break

    # Runtime: from tvm.compile to end of main
    compile_call = _find_call_in_func(main, "compile")
    rt_lo = compile_call.lineno
    # walk up to enclosing assign
    for node in main.body:
        if node.lineno <= compile_call.lineno <= (node.end_lineno or node.lineno):
            rt_lo = node.lineno
            break
    rt_hi = main.end_lineno or rt_lo

    return {
        "attention": _lines(src, attn.lineno, attn.end_lineno or attn.lineno),
        "export": _lines(src, export_lo, export_hi),
        "pipeline": _lines(src, pipe_lo, pipe_hi),
        "runtime": _lines(src, rt_lo, rt_hi),
    }


def extract_paged_blocks(src: str) -> dict[str, str]:
    tree = ast.parse(src)
    attn = _find_class(tree, "CausalSelfAttention")
    decoder = _find_class(tree, "SimpleDecoder")
    spec = _find_method(decoder, "get_default_spec")
    # Pipeline factory registered as _pipeline
    pipe = _find_func(tree, "_pipeline")
    main = _find_func(tree, "main")

    # Runtime focus: KV helpers + prefill/decode loop (after create_tir_paged_kv_cache)
    rt_lo = main.lineno
    for node in main.body:
        try:
            text = ast.unparse(node)
        except Exception:
            continue
        if "kv_state_add_sequence" in text or "create_tir_paged_kv_cache" in text:
            rt_lo = node.lineno
            break
    # If we found create_cache, start from add_sequence block when possible
    for node in main.body:
        try:
            text = ast.unparse(node)
        except Exception:
            continue
        if "kv_state_add_sequence" in text:
            rt_lo = node.lineno
            break
    rt_hi = main.end_lineno or rt_lo

    return {
        "attention": _lines(src, attn.lineno, attn.end_lineno or attn.lineno),
        "export": _lines(src, spec.lineno, spec.end_lineno or spec.lineno),
        "pipeline": _lines(src, pipe.lineno, pipe.end_lineno or pipe.lineno),
        "runtime": _lines(src, rt_lo, rt_hi),
    }


def cmd_diff(args: argparse.Namespace) -> int:
    base_src = _read(BASELINE)
    paged_src = _read(PAGED)
    base_blocks = extract_baseline_blocks(base_src)
    paged_blocks = extract_paged_blocks(paged_src)

    selected = BLOCKS if args.block == "all" else (args.block,)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for name in selected:
        left = base_blocks[name].splitlines(keepends=True)
        right = paged_blocks[name].splitlines(keepends=True)
        diff = list(
            difflib.unified_diff(
                left,
                right,
                fromfile=f"baseline/{name}",
                tofile=f"paged_kv/{name}",
            )
        )
        header = f"\n{'=' * 72}\n# BLOCK: {name}\n{'=' * 72}\n"
        if args.side_by_side:
            body = _side_by_side(base_blocks[name], paged_blocks[name], name)
        elif diff:
            body = "".join(diff)
            if not body.endswith("\n"):
                body += "\n"
        else:
            body = "(no textual diff)\n"
        text = header + body

        print(text)
        if out_dir:
            (out_dir / f"diff_{name}.patch").write_text(text, encoding="utf-8")
            (out_dir / f"baseline_{name}.py").write_text(base_blocks[name], encoding="utf-8")
            (out_dir / f"paged_{name}.py").write_text(paged_blocks[name], encoding="utf-8")

    if out_dir:
        print(f"\n[wrote block dumps + patches under {out_dir}]")
    return 0


def _side_by_side(left: str, right: str, name: str, width: int = 68) -> str:
    lrows = left.splitlines()
    rrows = right.splitlines()
    n = max(len(lrows), len(rrows))
    out = [f"{'baseline/' + name:<{width}} | {'paged_kv/' + name}"]
    out.append("-" * width + "-+-" + "-" * width)
    for i in range(n):
        a = lrows[i] if i < len(lrows) else ""
        b = rrows[i] if i < len(rrows) else ""
        out.append(f"{a[:width]:<{width}} | {b[:width]}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# IR stage inspection
# ---------------------------------------------------------------------------


def _ensure_sys_path() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _is_scheduled(fn) -> bool:
    if not fn.attrs:
        return False
    val = fn.attrs.get("tirx.is_scheduled", False)
    return bool(val)


def _clone_mod(mod):
    """Shallow-clone IRModule so later in-place passes do not mutate snapshots."""
    import tvm

    cloned = tvm.IRModule({gv: fn for gv, fn in mod.functions_items()})
    if mod.attrs:
        cloned = cloned.with_attrs(mod.attrs)
    return cloned


def _resolve_func_name(mod, func_name: str | None) -> str | None:
    names = [gv.name_hint for gv, fn in mod.functions_items() if type(fn).__name__ == "Function"]
    if func_name is None:
        for cand in ("forward", "prefill", "decode", "embed"):
            if cand in names:
                return cand
        return sorted(names)[0] if names else None
    return func_name if func_name in names else None


def _count_substr(text: str, needle: str) -> int:
    return text.count(needle)


def _analyze_stage(mod, func_name: str | None) -> dict:
    """Collect compact metrics used for stage comparison tables."""
    relax_fns, tir_sched, tir_raw = [], [], []
    for gv, fn in mod.functions_items():
        name = gv.name_hint
        if type(fn).__name__ == "Function":
            relax_fns.append(name)
        elif _is_scheduled(fn):
            tir_sched.append(name)
        else:
            tir_raw.append(name)

    func = _resolve_func_name(mod, func_name)
    script = mod[func].script(show_meta=False) if func else ""

    # High-level vs lowered op fingerprints in the chosen Relax function.
    fingerprints = {
        "R.matmul": _count_substr(script, "R.matmul"),
        "R.nn.softmax": _count_substr(script, "R.nn.softmax") + _count_substr(script, "R.softmax"),
        "R.nn.rms_norm": _count_substr(script, "R.nn.rms_norm"),
        "R.permute_dims": _count_substr(script, "R.permute_dims"),
        "R.call_tir": _count_substr(script, "R.call_tir"),
        "R.call_dps_packed": _count_substr(script, "R.call_dps_packed"),
        "R.call_pure_packed": _count_substr(script, "R.call_pure_packed"),
        "attention_with_fused_qkv": _count_substr(script, "attention_with_fused_qkv"),
        "fused_* call sites": len(
            [ln for ln in script.splitlines() if "call_tir(fused_" in ln or "call_tir(fused" in ln]
        ),
    }

    tir_scripts = {
        gv.name_hint: fn.script(show_meta=False)
        for gv, fn in mod.functions_items()
        if type(fn).__name__ != "Function"
    }
    n_blockidx = sum(1 for s in tir_scripts.values() if "blockIdx" in s or "threadIdx" in s)
    fused_tir = sorted(n for n in (tir_sched + tir_raw) if n.startswith("fused_"))
    kv_tir = sorted(
        n
        for n in (tir_sched + tir_raw)
        if any(k in n for k in ("paged_kv", "kv_cache", "fused_rope", "attention"))
    )

    # A few distinctive Relax lines (skip weight unpacking noise).
    highlight_lines = []
    for ln in script.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("R.func_attr"):
            continue
        if "packed_params[" in s or s.endswith(": R.Tensor"):
            continue
        interesting = any(
            k in s
            for k in (
                "R.matmul",
                "R.nn.",
                "R.call_tir",
                "R.call_dps",
                "R.call_pure_packed",
                "attention",
                "softmax",
                "rms_norm",
                "permute_dims",
                "triu",
                "silu",
            )
        )
        if interesting:
            highlight_lines.append(s)
        if len(highlight_lines) >= 12:
            break

    return {
        "func": func,
        "relax_fns": sorted(relax_fns),
        "n_relax": len(relax_fns),
        "tir_sched": sorted(tir_sched),
        "tir_raw": sorted(tir_raw),
        "n_tir": len(tir_sched) + len(tir_raw),
        "n_tir_sched": len(tir_sched),
        "n_tir_raw": len(tir_raw),
        "n_fused_tir": len(fused_tir),
        "fused_tir": fused_tir,
        "kv_tir": kv_tir,
        "n_gpu_bound_tir": n_blockidx,
        "fingerprints": fingerprints,
        "highlights": highlight_lines,
        "script": script,
        "tir_scripts": tir_scripts,
    }


STAGE_WHAT = {
    "export": "前端导出的高级 Relax 图（还没有 TIR kernel）",
    "legalize": "高级算子 → call_tir + 生成 PrimFunc（能算，但未融合/未调度）",
    "fuse": "小 kernel 合并成 fused_*（减少 launch；仍可能未 GPU 调度）",
    "dlight": "给 PrimFunc 绑 GPU thread（Relax 图常不变，看 TIR body）",
}


def _fmt_table(rows: list[dict]) -> str:
    """ASCII comparison table across stages."""
    headers = [
        ("stage", 10),
        ("relax_fn", 8),
        ("tir", 5),
        ("sched", 5),
        ("raw", 5),
        ("fused", 5),
        ("gpu", 5),
        ("R.matmul", 8),
        ("call_tir", 8),
        ("softmax", 7),
    ]
    line = " | ".join(h.center(w) for h, w in headers)
    sep = "-+-".join("-" * w for _, w in headers)
    out = [line, sep]
    for m in rows:
        fp = m["fingerprints"]
        vals = [
            m["stage"].ljust(10),
            str(m["n_relax"]).rjust(8),
            str(m["n_tir"]).rjust(5),
            str(m["n_tir_sched"]).rjust(5),
            str(m["n_tir_raw"]).rjust(5),
            str(m["n_fused_tir"]).rjust(5),
            str(m["n_gpu_bound_tir"]).rjust(5),
            str(fp["R.matmul"]).rjust(8),
            str(fp["R.call_tir"]).rjust(8),
            str(fp["R.nn.softmax"]).rjust(7),
        ]
        out.append(" | ".join(vals))
    out.append("")
    out.append("列含义: tir=PrimFunc总数  sched=已DLight调度  raw=未调度")
    out.append("       fused=fused_* 数量  gpu=含 blockIdx/threadIdx 的 PrimFunc 数")
    out.append("       R.matmul/call_tir/softmax = 所选 Relax 函数里的文本计数")
    return "\n".join(out) + "\n"


def _set_diff(a: list[str], b: list[str]) -> tuple[list[str], list[str]]:
    sa, sb = set(a), set(b)
    return sorted(sb - sa), sorted(sa - sb)


def _fmt_delta(prev: dict, curr: dict) -> str:
    lines = [
        f"Δ {prev['stage']} → {curr['stage']}",
        f"  本阶段作用: {STAGE_WHAT.get(curr['stage'], '')}",
    ]
    # Numeric deltas
    keys = [
        ("n_tir", "PrimFunc 总数"),
        ("n_tir_sched", "已调度 PrimFunc"),
        ("n_tir_raw", "未调度 PrimFunc"),
        ("n_fused_tir", "fused_* PrimFunc"),
        ("n_gpu_bound_tir", "含 GPU thread 绑定"),
    ]
    for k, label in keys:
        d = curr[k] - prev[k]
        if d:
            lines.append(f"  {label}: {prev[k]} → {curr[k]}  ({d:+d})")

    for name, pv, cv in [
        (k, prev["fingerprints"][k], curr["fingerprints"][k])
        for k in prev["fingerprints"]
    ]:
        if pv != cv:
            lines.append(f"  Relax「{name}」: {pv} → {cv}  ({cv - pv:+d})")

    added, removed = _set_diff(prev["tir_sched"] + prev["tir_raw"], curr["tir_sched"] + curr["tir_raw"])
    if added:
        show = added[:8]
        more = f" ...(+{len(added) - 8})" if len(added) > 8 else ""
        lines.append(f"  +PrimFunc: {show}{more}")
    if removed:
        show = removed[:8]
        more = f" ...(+{len(removed) - 8})" if len(removed) > 8 else ""
        lines.append(f"  -PrimFunc: {show}{more}")

    # What to look at
    if prev["stage"] == "export" and curr["stage"] == "legalize":
        lines.append("  看点: R.matmul/R.nn.* 消失，出现大量 R.call_tir 与同名 PrimFunc")
    elif curr["stage"] == "fuse":
        lines.append("  看点: 出现 fused_* PrimFunc；call_tir 次数常下降")
    elif curr["stage"] == "dlight":
        lines.append("  看点: Relax 图几乎不变；sched/gpu 上升；TIR 里出现 blockIdx/threadIdx")

    return "\n".join(lines) + "\n"


def _pick_demo_tir(metrics_by_stage: list[dict], explicit: str | None) -> str | None:
    if explicit and explicit != "auto":
        return explicit
    if explicit == "none":
        return None
    # Prefer a name present in the last stage: matmul* then fused_matmul* then fused_rope
    last = metrics_by_stage[-1]
    names = last["tir_sched"] + last["tir_raw"]
    for cand in ("matmul", "matmul1", "fused_rope"):
        if cand in names:
            return cand
    for n in names:
        if n.startswith("fused_matmul"):
            return n
    for n in names:
        if "matmul" in n:
            return n
    return names[0] if names else None


def _tir_snippet(script: str, *, max_lines: int = 18) -> str:
    """Keep a short, readable slice of a PrimFunc (signature + first loops)."""
    rows = script.splitlines()
    # drop leading imports/comments noise but keep def and body start
    start = 0
    for i, ln in enumerate(rows):
        if ln.startswith("def ") or ln.startswith("@T.prim_func") or "prim_func" in ln:
            start = i
            break
    chunk = rows[start : start + max_lines]
    return "\n".join(chunk) + ("\n" if chunk else "")


def _fmt_tir_compare(prev: dict, curr: dict, tir_name: str) -> str:
    ps = prev["tir_scripts"].get(tir_name)
    cs = curr["tir_scripts"].get(tir_name)
    if ps is None and cs is None:
        return f"(demo TIR {tir_name!r} not in either stage)\n"
    lines = [f"TIR 对照样例: {tir_name}"]
    if ps is None:
        lines.append(f"  {prev['stage']}: (不存在)")
    else:
        lines.append(
            f"  {prev['stage']}: scheduled={tir_name in prev['tir_sched']}  "
            f"gpu_bound={'blockIdx' in ps or 'threadIdx' in ps}"
        )
        lines.append(_indent(_tir_snippet(ps), 4))
    if cs is None:
        lines.append(f"  {curr['stage']}: (不存在)")
    else:
        lines.append(
            f"  {curr['stage']}: scheduled={tir_name in curr['tir_sched']}  "
            f"gpu_bound={'blockIdx' in cs or 'threadIdx' in cs}"
        )
        lines.append(_indent(_tir_snippet(cs), 4))
    # tiny unified diff of snippets
    if ps and cs:
        diff = list(
            difflib.unified_diff(
                _tir_snippet(ps, max_lines=30).splitlines(),
                _tir_snippet(cs, max_lines=30).splitlines(),
                fromfile=f"{prev['stage']}/{tir_name}",
                tofile=f"{curr['stage']}/{tir_name}",
                lineterm="",
                n=2,
            )
        )
        if diff:
            lines.append("  --- snippet diff ---")
            lines.extend("  " + x for x in diff[:40])
            if len(diff) > 40:
                lines.append(f"  ... ({len(diff) - 40} more diff lines)")
    return "\n".join(lines) + "\n"


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + ln if ln else ln for ln in text.splitlines())


def _fmt_highlights(m: dict) -> str:
    if not m["highlights"]:
        return "  (no distinctive Relax ops sampled)\n"
    lines = [f"  Relax「{m['func']}」关键行（采样，非全文）:"]
    for ln in m["highlights"]:
        lines.append(f"    {ln}")
    return "\n".join(lines) + "\n"


def _export_baseline():
    import tvm
    from tvm.relax.frontend import nn

    from common_decoder import ModelConfig
    from simple_llm_decoder import SimpleDecoder

    config = ModelConfig()
    model = SimpleDecoder(config)
    mod, _ = model.export_tvm(
        spec={
            "forward": {
                "input_ids": nn.spec.Tensor([1, config.seq_len], "int32"),
            }
        }
    )
    return mod, "forward", tvm


def _export_paged():
    import tvm

    from common_decoder import ModelConfig

    pk = str(ROOT / "paged_kv_cache")
    if pk not in sys.path:
        sys.path.insert(0, pk)
    import decoder_paged_kv as paged_mod  # registers opt_llm, sets target

    config = ModelConfig()
    model = paged_mod.SimpleDecoder(config)
    mod, _ = model.export_tvm(spec=model.get_default_spec())
    return mod, "prefill", tvm


def _get_target(tvm):
    try:
        dev = tvm.device("cuda", 0)
        if dev.exist:
            return tvm.target.Target.from_device(dev), True
    except Exception:
        pass
    return tvm.target.Target("llvm"), False


def _run_stages(model_key: str, args: argparse.Namespace) -> int:
    _ensure_sys_path()
    from tvm import relax
    from tvm.s_tir import dlight as dl

    if model_key == "baseline":
        mod, default_func, tvm = _export_baseline()
    else:
        pk = str(ROOT / "paged_kv_cache")
        if pk not in sys.path:
            sys.path.insert(0, pk)
        mod, default_func, tvm = _export_paged()

    func = args.func or default_func
    want = set(STAGES if args.stage == "all" else [s.strip() for s in args.stage.split(",")])
    unknown = want - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stage(s): {sorted(unknown)}; choose from {STAGES}")

    target, has_cuda = _get_target(tvm)
    if "dlight" in want and not has_cuda:
        print("[warn] CUDA not available; DLight GPU schedules may be weak on llvm target.")

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    snapshots: list[tuple[str, object]] = []

    if "export" in want:
        snapshots.append(("export", _clone_mod(mod)))

    with target:
        if "legalize" in want or "fuse" in want or "dlight" in want:
            mod_l = relax.transform.LegalizeOps()(mod)
            mod_l = relax.transform.AnnotateTIROpPattern()(mod_l)
            mod_l = relax.transform.FoldConstant()(mod_l)
            if "legalize" in want:
                snapshots.append(("legalize", _clone_mod(mod_l)))

            mod_f = mod_l
            if "fuse" in want or "dlight" in want:
                mod_f = relax.transform.FuseOps()(mod_f)
                mod_f = relax.transform.FuseTIR()(mod_f)
                mod_f = relax.transform.DeadCodeElimination()(mod_f)
                if "fuse" in want:
                    snapshots.append(("fuse", _clone_mod(mod_f)))

            if "dlight" in want:
                mod_d = dl.ApplyDefaultSchedule(
                    dl.gpu.Matmul(),
                    dl.gpu.GEMV(),
                    dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(),
                    dl.gpu.Fallback(),
                )(_clone_mod(mod_f))
                snapshots.append(("dlight", mod_d))

    # Analyze all stages first, then print a readable report.
    metrics: list[dict] = []
    for stage_name, stage_mod in snapshots:
        m = _analyze_stage(stage_mod, func)
        m["stage"] = stage_name
        m["mod"] = stage_mod
        metrics.append(m)

    report: list[str] = []
    report.append("=" * 72)
    report.append(f"STAGES REPORT  model={model_key}  func={func}  target={target}")
    report.append("=" * 72)
    report.append("")
    report.append("【1】各阶段一览（先看这张表）")
    report.append(_fmt_table(metrics))
    report.append("阶段含义:")
    for s, _mod in snapshots:
        report.append(f"  - {s}: {STAGE_WHAT[s]}")
    report.append("")

    report.append("【2】相邻阶段差分（数字/名单变化）")
    report.append("-" * 72)
    for i in range(1, len(metrics)):
        report.append(_fmt_delta(metrics[i - 1], metrics[i]))
    report.append("")

    report.append("【3】每阶段 Relax 关键算子采样")
    report.append("-" * 72)
    for i, m in enumerate(metrics):
        report.append(f"[{m['stage']}]")
        if i > 0 and m["highlights"] == metrics[i - 1]["highlights"]:
            report.append(
                f"  (= 与 [{metrics[i - 1]['stage']}] 的 Relax 采样相同 → 本阶段变化在 TIR，见【4】)\n"
            )
        else:
            report.append(_fmt_highlights(m))
        if m["fused_tir"] and (
            i == 0 or m["fused_tir"] != metrics[i - 1]["fused_tir"]
        ):
            show = m["fused_tir"][:10]
            more = f" ...(+{len(m['fused_tir']) - 10})" if len(m["fused_tir"]) > 10 else ""
            report.append(f"  fused_* : {show}{more}")
        if m["kv_tir"] and (i == 0 or m["kv_tir"] != metrics[i - 1]["kv_tir"]):
            show = m["kv_tir"][:10]
            more = f" ...(+{len(m['kv_tir']) - 10})" if len(m["kv_tir"]) > 10 else ""
            report.append(f"  kv/attn : {show}{more}")
        report.append("")

    # Auto TIR compare on the transition where it matters most.
    show_tir = getattr(args, "show_tir", "auto")
    tir_name = _pick_demo_tir(metrics, show_tir)
    if tir_name and len(metrics) >= 2:
        report.append("【4】TIR 样例对照（最能看出 DLight）")
        report.append("-" * 72)
        # Prefer fuse→dlight; else last two stages.
        pair = None
        stages = [m["stage"] for m in metrics]
        if "fuse" in stages and "dlight" in stages:
            pair = (stages.index("fuse"), stages.index("dlight"))
        elif "legalize" in stages and "fuse" in stages:
            pair = (stages.index("legalize"), stages.index("fuse"))
        else:
            pair = (len(metrics) - 2, len(metrics) - 1)
        report.append(_fmt_tir_compare(metrics[pair[0]], metrics[pair[1]], tir_name))
        report.append(f"(用 --show-tir NAME 换样例；--show-tir none 关闭；当前={tir_name})")
        report.append("")

    how_to_read = textwrap.dedent(
        f"""\
        【怎么读】
          export  → legalize : 高级 op 变 call_tir，PrimFunc 从 0 涨起来
          legalize → fuse    : 出现 fused_*，PrimFunc/call_tir 往往变少
          fuse → dlight      : Relax 行几乎一样；sched/gpu 上升；TIR 循环被拆到 thread
          完整 IR 默认不刷屏；加 --dump-ir 或 --out DIR 落盘再慢慢看
        """
    )
    report.append(how_to_read)

    text = "\n".join(report)
    print(text)

    if out_dir:
        summary_path = out_dir / f"{model_key}_{func}_SUMMARY.txt"
        summary_path.write_text(text, encoding="utf-8")
        print(f"[wrote {summary_path}]")

    # Optional full IR dumps (off by default to keep terminal readable).
    if getattr(args, "dump_ir", False) or out_dir:
        dump_root = out_dir if out_dir else Path.cwd() / "ir_dump"
        dump_root.mkdir(parents=True, exist_ok=True)
        for m in metrics:
            stage_mod = m["mod"]
            func_name = m["func"] or "unknown"
            body = m["script"]
            if args.max_chars > 0 and len(body) > args.max_chars:
                body = body[: args.max_chars] + f"\n\n... truncated ({len(m['script'])} chars total)\n"
            path = dump_root / f"{model_key}_{m['stage']}_{func_name}.relax.txt"
            header = (
                f"# {model_key} @ {m['stage']}  func={func_name}\n"
                f"# {STAGE_WHAT.get(m['stage'], '')}\n\n"
            )
            path.write_text(header + body, encoding="utf-8")
            if tir_name and tir_name in m["tir_scripts"]:
                tpath = dump_root / f"{model_key}_{m['stage']}_{tir_name}.tir.txt"
                tpath.write_text(m["tir_scripts"][tir_name], encoding="utf-8")
            if getattr(args, "dump_ir", False) and not out_dir:
                print(f"[wrote {path}]")
        if out_dir:
            print(f"[wrote per-stage IR under {dump_root}]")

    return 0


def cmd_stages(args: argparse.Namespace) -> int:
    models = ("baseline", "paged") if args.model == "both" else (args.model,)
    rc = 0
    for m in models:
        # isolate output subdir per model when dumping both
        if args.out and args.model == "both":
            sub = argparse.Namespace(**vars(args))
            sub.out = str(Path(args.out) / m)
            rc = _run_stages(m, sub) or rc
        else:
            rc = _run_stages(m, args) or rc
    return rc


def cmd_all(args: argparse.Namespace) -> int:
    print(textwrap.dedent("""\
        ############################################################
        # 1) Source diff (attention / export / pipeline / runtime)
        ############################################################
    """))
    dargs = argparse.Namespace(
        block=args.block,
        side_by_side=args.side_by_side,
        out=str(Path(args.out) / "diff") if args.out else None,
    )
    cmd_diff(dargs)

    print(textwrap.dedent("""\
        ############################################################
        # 2) Compile-stage IR (export / legalize / fuse / dlight)
        ############################################################
    """))
    sargs = argparse.Namespace(
        model=args.model,
        func=args.func,
        stage=args.stage,
        max_chars=args.max_chars,
        show_tir=args.show_tir,
        dump_ir=args.dump_ir,
        out=str(Path(args.out) / "stages") if args.out else None,
    )
    return cmd_stages(sargs)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare baseline vs paged-kv and inspect Relax IR by compile stage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_diff = sub.add_parser("diff", help="Unified diff of the four source blocks")
    p_diff.add_argument(
        "--block",
        choices=[*BLOCKS, "all"],
        default="all",
        help="Which block to diff (default: all)",
    )
    p_diff.add_argument(
        "--side-by-side",
        action="store_true",
        help="Print side-by-side instead of unified diff",
    )
    p_diff.add_argument("--out", type=str, default=None, help="Directory to write dumps/patches")
    p_diff.set_defaults(func_cmd=cmd_diff)

    p_st = sub.add_parser(
        "stages",
        help="Compare IR across export/legalize/fuse/dlight (table + deltas)",
    )
    p_st.add_argument(
        "--model",
        choices=("baseline", "paged", "both"),
        default="baseline",
        help="Which tutorial model to export (default: baseline)",
    )
    p_st.add_argument(
        "--func",
        default=None,
        help="Relax function to analyze (baseline: forward; paged: prefill/decode/embed)",
    )
    p_st.add_argument(
        "--stage",
        default="all",
        help=f"Comma-separated stages or 'all'. Choices: {', '.join(STAGES)}",
    )
    p_st.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="When dumping IR, truncate scripts to N chars (0 = full; default 0)",
    )
    p_st.add_argument(
        "--show-tir",
        default="auto",
        metavar="NAME",
        help="TIR demo name, 'auto' (default), or 'none'",
    )
    p_st.add_argument(
        "--dump-ir",
        action="store_true",
        help="Also write full Relax/TIR scripts to disk (and print paths)",
    )
    p_st.add_argument(
        "--out",
        type=str,
        default=None,
        help="Directory for SUMMARY + per-stage IR dumps",
    )
    p_st.set_defaults(func_cmd=cmd_stages)

    p_all = sub.add_parser("all", help="Run diff then stages")
    p_all.add_argument("--block", choices=[*BLOCKS, "all"], default="all")
    p_all.add_argument("--side-by-side", action="store_true")
    p_all.add_argument("--model", choices=("baseline", "paged", "both"), default="both")
    p_all.add_argument("--func", default=None)
    p_all.add_argument("--stage", default="all")
    p_all.add_argument("--max-chars", type=int, default=0)
    p_all.add_argument("--show-tir", default="auto", metavar="NAME")
    p_all.add_argument("--dump-ir", action="store_true")
    p_all.add_argument("--out", type=str, default=None, help="Root dir for all dumps")
    p_all.set_defaults(func_cmd=cmd_all)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func_cmd(args)


if __name__ == "__main__":
    raise SystemExit(main())
