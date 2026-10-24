"""Generate the roxygen2 help of the R distribution wrappers from the Python docstrings.

Single source: the docstrings of BayesForge's ``UnifiedDist`` methods (Python).
For every ``bf.dist.<name> = function(...)`` in ``R/*.R`` the script finds the
Python method the wrapper calls (``$dist$<method>(``), converts its docstring
with fixed rules -- no language model, so the output is reproducible -- and
replaces the roxygen block (the ``#'`` lines) just above the function:

* title, description, ``@param`` for every R argument, ``@return``,
  ``@examples`` (the Python example rewritten in R), ``@seealso`` (the NumPyro
  page wrapped), ``@references``, ``@export``;
* math: ``$$...$$`` -> ``\\deqn{}``, ``$...$`` -> ``\\eqn{}``;
* Python -> R: ``True``/``False``/``None`` -> ``TRUE``/``FALSE``/``py_none()``,
  ``[a, b]`` / ``(a, b)`` -> ``c(a, b)``, ``m.dist.f(...)`` -> ``bf.dist.f(...)``,
  ``jnp.f`` -> ``jnp$f``;
* arguments that every wrapper shares (shape, sample, obs, name, ...) get one
  standard R description (STANDARD below), the others their Python text.

Usage (from the BFR root, with the bayesforge Python package importable):
    python tools/build_r_docs.py            # rewrite R/*.R
    python tools/build_r_docs.py --check    # exit 1 if a file would change
Then run roxygen2::roxygenise() to rebuild man/.
"""
import argparse
import ast
import inspect
import os
import re
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STANDARD = {
    "shape": "Integer vector. With `sample = FALSE`, the batch shape of the "
             "distribution in the model; with `sample = TRUE`, the shape of the draw. "
             "Defaults to `c()` (no extra dimensions).",
    "event": "Integer. Number of batch dimensions reinterpreted as event dimensions "
             "(model building only). Defaults to `0`.",
    "mask": "Optional logical vector or array masking observations (`FALSE` = ignored). "
            "Defaults to `py_none()`.",
    "sample": "Logical. If `TRUE`, directly draw values from the distribution and return "
              "them; if `FALSE` (default), create a random variable inside a model.",
    "seed": "Integer seed for this draw when `sample = TRUE`. By default (`py_none()`) "
            "the draw comes from the session's random stream, set by "
            "`importBF(rand_seed = ...)`, so consecutive draws differ but the whole "
            "sequence is reproducible. No effect when `sample = FALSE`.",
    "obs": "Numeric vector or array of observed values. If given, the random variable "
           "is conditioned on them (likelihood); otherwise it is latent. "
           "Defaults to `py_none()`.",
    "name": "Character. Name of the random variable in the model. When not given, an "
            "observed site is named after the variable passed to `obs` "
            "(`obs = y` gives `\"y\"`); use [bf_model()] to name parameters after "
            "the variables they are assigned to.",
    "create_obj": "Logical. If `TRUE`, return the distribution object itself instead of "
                  "a random variable (e.g. to build mixtures). Defaults to `FALSE`.",
    "validate_args": "Logical. Whether to validate the parameter values. "
                     "Defaults to `py_none()` (the NumPyro default).",
    "to_jax": "Logical. Return a JAX array (`TRUE`, default) or convert to R.",
    "infer": "Optional named list passed to NumPyro's `infer` argument "
             "(e.g. `list(enumerate = \"parallel\")`).",
    "enum": "Logical. If `TRUE`, the discrete latent is summed out exactly (parallel "
            "enumeration) instead of sampled; same as "
            "`infer = list(enumerate = \"parallel\")`. Defaults to `FALSE`.",
}

HEADING = re.compile(r"^\s*(#{1,6})\s*(.*?)\s*$")
SECTION_KEYS = (("arg", "args"), ("param", "args"), ("return", "returns"),
                ("example", "example"), ("wrapper", "wrapper"), ("reference", "references"))


# ----------------------------------------------------------------------------- parsing
def parse_docstring(doc):
    """Split a BayesForge docstring into title / description / sections."""
    lines = textwrap.dedent(doc or "").strip("\n").splitlines()
    out = {"title": None, "description": [], "args": {}, "returns": [], "example": [],
           "wrapper": None, "references": []}
    section = "description"
    current_arg = None
    for line in lines:
        h = HEADING.match(line)
        bare = re.match(r"^\s*(Args|Returns|Parameters|Example Usage)\s*:\s*$", line)
        if h or bare:
            text = (h.group(2) if h else bare.group(1)).strip("# ").strip()
            low = text.lower()
            key = next((k for word, k in SECTION_KEYS if low.startswith(word)), None)
            if key:
                section, current_arg = key, None
                if key == "wrapper":
                    url = re.search(r"https?://[^\s)\]]+", text)
                    out["wrapper"] = url.group(0) if url else None
                continue
            if out["title"] is None and section == "description" and not out["description"]:
                out["title"] = text
                continue
        if section == "description":
            out["description"].append(line)
        elif section == "args":
            m = re.match(r"^\s*[-*]\s*\*{1,2}([A-Za-z_][\w]*)\*{1,2}\s*(?:\(([^)]*)\))?\s*:?\s*(.*)$", line)
            if m:
                current_arg = m.group(1)
                out["args"][current_arg] = [m.group(3).strip()]
            elif current_arg and line.strip():
                out["args"][current_arg].append(line.strip())
        elif section == "returns":
            if line.strip():
                out["returns"].append(line.strip())
        elif section == "example":
            out["example"].append(line)
        elif section == "wrapper":
            url = re.search(r"https?://[^\s)\]]+", line)
            if url and not out["wrapper"]:
                out["wrapper"] = url.group(0)
        elif section == "references":
            if line.strip():
                out["references"].append(line.strip())
    return out


# ------------------------------------------------------------------------- conversion
def math_to_rd(text):
    """Math to Rd macros and markdown made safe for roxygen.

    $$...$$ and \\[...\\] -> \\deqn{}; $...$ and \\(...\\) -> \\eqn{}. Outside math:
    square brackets that are not a markdown link [text](url) are escaped (else
    roxygen reads "[0, 1]" as a link), horizontal rules and block-quote markers
    are dropped (not supported by roxygen markdown), % is escaped for Rd.
    """
    keep = []

    def stash(macro):
        def f(m):
            keep.append(f"\\{macro}{{" + " ".join(m.group(1).split()).replace("%", "\\%") + "}")
            return f"\x00{len(keep) - 1}\x00"
        return f

    text = re.sub(r"\$\$(.+?)\$\$", lambda m: "\n" + stash("deqn")(m) + "\n", text, flags=re.S)
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: "\n" + stash("deqn")(m) + "\n", text, flags=re.S)
    text = re.sub(r"\\\((.+?)\\\)", stash("eqn"), text, flags=re.S)
    text = re.sub(r"(?<!\\)\$([^$\n]+?)\$", stash("eqn"), text)
    text = "\n".join(l for l in text.split("\n") if not re.fullmatch(r"\s*([-*_]\s*){3,}", l))
    text = re.sub(r"(?m)^\s*>\s?", "", text)
    links = []
    text = re.sub(r"\[([^\[\]]+)\]\((https?://[^)\s]+)\)",
                  lambda m: links.append(m.group(0)) or f"\x01{len(links) - 1}\x01", text)
    text = text.replace("[", "\\[").replace("]", "\\]").replace("%", "\\%")
    # LaTeX left outside $...$ in a docstring (e.g. "\\beta" in plain text):
    # escape the backslash and the braces, otherwise Rd reads it as an unknown
    # macro / "lost braces". (These docstrings should use $...$; see the
    # list printed by --report-latex.)
    text = re.sub(r"\\(?=[A-Za-z])", r"\\\\", text)
    text = re.sub(r"(?<!\\)([{}])", r"\\\1", text)
    text = re.sub(r"\x01(\d+)\x01", lambda m: links[int(m.group(1))], text)
    return re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], text)


def py_words_to_r(text):
    text = re.sub(r"\bTrue\b", "TRUE", text)
    text = re.sub(r"\bFalse\b", "FALSE", text)
    text = re.sub(r"\bNone\b", "py_none()", text)
    text = re.sub(r"(\w)=(TRUE|FALSE|py_none\(\))", r"\1 = \2", text)
    text = text.replace("jnp.ndarray", "numeric vector or array")
    text = re.sub(r"\btuple\b", "integer vector", text)
    text = re.sub(r"\bm\.dist\.", "bf.dist.", text)
    return text


INT_KWARGS = {"total_count", "num_steps", "dimension", "event_size", "num_categories",
              "rank", "high_int", "n"}


class _ToR:
    """Python expression (ast) -> R code, for the documentation examples."""

    def __init__(self, r_names):
        self.r_names = r_names              # python method -> R wrapper name

    def __call__(self, node, as_int=False):
        f = getattr(self, "_" + type(node).__name__, None)
        if f is None:
            raise ValueError(type(node).__name__)
        return f(node, as_int) if f.__code__.co_argcount == 3 else f(node)

    def _Constant(self, n, as_int=False):
        v = n.value
        if v is True:
            return "TRUE"
        if v is False:
            return "FALSE"
        if v is None:
            return "py_none()"
        if isinstance(v, str):
            return '"' + v.replace('"', '\\"') + '"'
        if isinstance(v, int):
            return f"{v}L" if as_int else str(v)
        return repr(v)

    def _Name(self, n):
        return n.id

    def _attr_path(self, n):
        parts = []
        while isinstance(n, ast.Attribute):
            parts.append(n.attr)
            n = n.value
        if isinstance(n, ast.Name):
            parts.append(n.id)
            return list(reversed(parts))
        return None

    def _Attribute(self, n):
        path = self._attr_path(n)
        if path is None:
            raise ValueError("attribute")
        if path[0] in ("jnp", "jax", "np"):
            root = "jnp" if path[0] in ("jnp", "np") else "jax"
            return root + "".join("$" + p for p in path[1:])
        raise ValueError("attribute " + ".".join(path))

    def _Call(self, n, as_int=False):
        path = self._attr_path(n.func) if isinstance(n.func, ast.Attribute) else None
        ints = False
        if path and path[:2] == ["m", "dist"] and len(path) == 3:
            if path[2] not in self.r_names:
                raise ValueError("no R wrapper for " + path[2])
            fn = "bf.dist." + self.r_names[path[2]]
        elif path and path[0] in ("jnp", "jax", "np"):
            fn = self._Attribute(n.func)
            ints = True                       # jnp$eye(2L), jnp$zeros(c(3L, 4L))
        else:
            raise ValueError("call")
        args = [self(a, ints) for a in n.args]
        for kw in n.keywords:
            if kw.arg is None:
                raise ValueError("**kwargs")
            args.append(f"{kw.arg} = {self(kw.value, ints or kw.arg in INT_KWARGS or kw.arg == 'shape')}")
        return f"{fn}({', '.join(args)})"

    def _List(self, n, as_int=False):
        if n.elts and all(isinstance(e, ast.List) for e in n.elts):      # 2-D -> matrix
            rows = [[self(x, as_int) for x in e.elts] for e in n.elts]
            return f"matrix(c({', '.join(v for r in rows for v in r)}), nrow = {len(rows)}L, byrow = TRUE)"
        items = ", ".join(self(e, as_int) for e in n.elts)
        literal = all(isinstance(e, (ast.Constant, ast.UnaryOp)) for e in n.elts)
        return f"c({items})" if literal else f"list({items})"    # objects: an R list

    def _Tuple(self, n, as_int=False):
        return "c(" + ", ".join(self(e, True) for e in n.elts) + ")"   # shapes: integers

    def _UnaryOp(self, n, as_int=False):
        op = {ast.USub: "-", ast.UAdd: "+", ast.Not: "!"}[type(n.op)]
        return op + self(n.operand, as_int)

    def _BinOp(self, n, as_int=False):
        op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "^",
              ast.MatMult: "%*%"}.get(type(n.op))
        if op is None:
            raise ValueError("operator")
        return f"{self(n.left, as_int)} {op} {self(n.right, as_int)}"

    def _Lambda(self, n):
        args = ", ".join(a.arg for a in n.args.args)
        body = n.body
        if isinstance(body, ast.Tuple):
            inner = ", ".join(self(e) for e in body.elts)
            return f"function({args}) reticulate::tuple({inner})"
        return f"function({args}) {self(body)}"


def _dedent_example(lines):
    """Dedent an example whose first line is less indented than the others."""
    lines = [l.rstrip() for l in lines]
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ""
    rest = [l for l in lines[1:] if l.strip()]
    ind = min((len(l) - len(l.lstrip()) for l in rest), default=0)
    first = lines[0].strip()
    return "\n".join([first] + [l[ind:] if len(l) >= ind else l.strip() for l in lines[1:]])


def example_to_r(example_lines, r_name, r_names):
    """R version of the Python example, statement by statement (assignments
    and m.dist.* calls; imports, m = bf(...) and definitions are left out).
    A statement that cannot be translated drops the rest of the example."""
    code = _dedent_example(example_lines)
    to_r = _ToR(r_names)
    out = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            continue
        try:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                tgt, val = node.targets[0], node.value
                if isinstance(val, ast.Call) and getattr(val.func, "id", None) == "bf":
                    continue                                  # m = bf(...)
                if isinstance(tgt, ast.Tuple) and isinstance(val, ast.Tuple):
                    out += [f"{t.id} <- {to_r(v, True)}" for t, v in zip(tgt.elts, val.elts)]
                else:
                    out.append(f"{to_r(tgt)} <- {to_r(val)}")
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                out.append(to_r(node.value))
        except (ValueError, KeyError):
            break
    if not any("bf.dist." in o for o in out):
        out = [f"bf.dist.{r_name}(sample = TRUE)"]
    return ["library(BayesForge)", "m <- importBF(platform = \"cpu\")"] + out


def _wrap_r_call(line, width=90, indent=""):
    """Split an R statement longer than `width` into shorter lines: an optional
    leading assignment (`lhs <- ` or, for an argument, `name = `) stays on the
    first line, the arguments of the outermost call each get a continuation
    line, and an argument that is itself too long is wrapped the same way.
    Commas inside nested (), [], {} or quoted strings are left alone; a plain
    c(...) of numbers is packed with as many numbers per line as fit."""
    if len(line) <= width:
        return [line]

    def top_level(s):
        """Yield (index, char) for characters outside quotes and brackets."""
        depth, quote, i = 0, None, 0
        while i < len(s):
            ch = s[i]
            if quote:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = None
            elif ch in "'\"":
                quote = ch
            elif ch in "([{":
                depth += 1
                if depth == 1 and ch == "(":
                    yield i, ch
            elif ch in ")]}":
                if depth and ch == ")" and depth == 1:
                    yield i, ch
                depth -= 1
            elif depth == 0:
                yield i, ch
            i += 1

    def split_args(s):
        """Split the inside of a call at its top-level commas."""
        args, start, depth, quote, i = [], 0, 0, None, 0
        while i < len(s):
            ch = s[i]
            if quote:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = None
            elif ch in "'\"":
                quote = ch
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "," and depth == 0:
                args.append(s[start:i].strip())
                start = i + 1
            i += 1
        args.append(s[start:].strip())
        return [a for a in args if a]

    stripped = line.rstrip()
    tail = "," if stripped.endswith(",") else ""
    body = stripped[:-1] if tail else stripped
    lead = line[:len(line) - len(line.lstrip())] or indent
    m = re.match(r"[A-Za-z._][A-Za-z0-9._$]*\s*(?:<-|=(?!=))\s+", body[len(lead):])
    prefix, rest = "", body[len(lead):]
    if m:
        prefix = body[len(lead):len(lead) + m.end()]
        rest = body[len(lead) + m.end():]
    if not rest.endswith(")"):
        return [line]
    open_ = close = -1
    for i, ch in top_level(rest):
        if ch == "(":
            if open_ >= 0:                   # first "(" is not the call's
                return [line]
            open_ = i
        elif ch == ")":
            close = i
            break
    if open_ < 0 or close != len(rest) - 1:
        return [line]
    head, args = rest[:open_], split_args(rest[open_ + 1:close])
    if not args:
        return [line]
    ind = lead + "    "
    out = [lead + prefix + head + "("]
    number = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?[L]?$")
    if head == "c" and all(number.match(a) for a in args):
        cur = ind
        for k, a in enumerate(args):
            piece = a + ("," if k < len(args) - 1 else "")
            if cur != ind and len(cur) + 1 + len(piece) > width:
                out.append(cur)
                cur = ind + piece
            else:
                cur = cur + (" " if cur != ind else "") + piece
        if cur != ind:
            out.append(cur)
    else:
        for k, a in enumerate(args):
            child = ind + a + ("," if k < len(args) - 1 else "")
            out += _wrap_r_call(child, width, ind) if len(child) > width else [child]
    out.append(lead + ")" + tail)
    return out


def _blocks(text):
    """Paragraphs, list items and \\deqn lines of a markdown text."""
    blocks, cur = [], []
    for line in text.split("\n"):
        st = line.strip()
        starts_item = re.match(r"^([-*+]|\d+\.)\s", st)
        if not st or st.startswith("\\deqn") or starts_item:
            if cur:
                blocks.append(" ".join(cur))
                cur = []
            if not st:
                blocks.append("")
            elif st.startswith("\\deqn"):
                blocks.append(st)
            else:
                cur = [st]
            continue
        cur.append(st)
    if cur:
        blocks.append(" ".join(cur))
    out = []
    for b in blocks:                      # collapse repeated blank lines
        if b == "" and (not out or out[-1] == ""):
            continue
        out.append(b)
    while out and out[-1] == "":
        out.pop()
    return out


def wrap(prefix, text, width=80):
    """Roxygen lines for text; never start a line with a markdown marker."""
    out = []
    for i, block in enumerate(_blocks(text)):
        if i == 0 and prefix:
            block = prefix + " " + block
        if not block:
            out.append("#'")
            continue
        if block.startswith("\\deqn"):
            out.append("#' " + block)
            continue
        lines = textwrap.wrap(block, width=width - 3, break_long_words=False,
                              break_on_hyphens=False)
        merged = []
        for ln in lines:
            # a continuation line starting with >, # or a list marker would be
            # read as markdown (block quote, heading, list): keep it on the
            # previous line instead
            if merged and re.match(r"^(>|#|[-*+]\s|\d+\.\s)", ln):
                merged[-1] += " " + ln
            else:
                merged.append(ln)
        indent = "  " if prefix else ""
        out.extend(("#' " if k == 0 else "#' " + indent) + ln for k, ln in enumerate(merged))
    return out or ["#'"]


def describe_arg(arg, parsed):
    if arg in STANDARD:
        return STANDARD[arg]
    if arg in parsed["args"]:
        return math_to_rd(py_words_to_r(" ".join(parsed["args"][arg]))).strip()
    if arg == "...":
        return "Further arguments passed to the Python method."
    return f"See the Python documentation of `{arg}`."


def roxygen_block(r_name, r_args, py_method, parsed, r_names):
    title = parsed["title"] or py_method.replace("_", " ").title()
    title = re.sub(r"\s+", " ", title).strip()
    if not title.lower().endswith("distribution"):
        title += " distribution"
    desc = math_to_rd(py_words_to_r("\n".join(l.strip() for l in parsed["description"]))).strip()
    desc = re.sub(r"\n{3,}", "\n\n", desc) or title + "."
    lines = [f"#' @title {title}", "#'", "#' @description"]
    lines += wrap("", desc)
    lines.append("#'")
    for a in r_args:
        lines += wrap(f"@param {a}", describe_arg(a, parsed))
    lines.append("#'")
    ret = [re.sub(r"^[-*+]\s+", "", py_words_to_r(r)).strip() for r in parsed["returns"]]
    if ret:
        lines.append("#' @return")
        lines.append("#' \\itemize{")
        for r in ret:
            lines += ["#'   \\item " + ln[3:] if i == 0 else "#'     " + ln[3:]
                      for i, ln in enumerate(wrap("", math_to_rd(r), width=74))]
        lines.append("#' }")
    else:
        lines.append("#' @return A random variable (model building) or draws (`sample = TRUE`).")
    lines.append("#'")
    lines.append("#' @examples")
    lines.append("#' \\donttest{")
    for ln in example_to_r(parsed["example"], r_name, r_names):
        lines += ["#' " + x for x in _wrap_r_call(ln)]
    lines.append("#' }")
    if parsed["wrapper"]:
        lines += ["#'", f"#' @seealso \\url{{{parsed['wrapper']}}}"]
    refs = [math_to_rd(r.lstrip("*- ").strip()) for r in parsed["references"]]
    if refs:
        lines += ["#'", "#' @references"]
        for r in refs:
            lines += wrap("", r)
            lines.append("#'")
        lines.pop()
    lines += ["#'", "#' @export"]
    return [re.sub(r"\s+$", "", ln) for ln in lines]


# ---------------------------------------------------------------------------- R files
FUN = re.compile(r"^(bf\.dist\.[A-Za-z0-9_.]+)\s*(?:=|<-)\s*function\(", re.M)


def r_formals(src, start):
    """Argument names of the function whose '(' ends at index start."""
    depth, i = 1, start
    while depth:
        depth += {"(": 1, ")": -1}.get(src[i], 0)
        i += 1
    sig = src[start:i - 1]
    names, depth, cur = [], 0, ""
    for ch in sig + ",":
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            nm = cur.split("=")[0].strip()
            if nm:
                names.append(nm)
            cur = ""
        else:
            cur += ch
    return names, i


def process_file(path, methods, report, r_names):
    src = open(path, encoding="utf-8").read()
    out, pos = [], 0
    for m in FUN.finditer(src):
        r_full = m.group(1)
        args, end = r_formals(src, m.end())
        body_end = src.find("\n}", end)
        call = re.search(r"\$dist\$([A-Za-z0-9_]+)\s*\(", src[end:body_end])
        if not call or args == ["..."] or call.group(1) not in methods:
            continue
        py_method = call.group(1)
        # The old roxygen block starts at the first #' line after the last line
        # of code above the function (blank lines inside it are dropped too:
        # they used to split a block in two, losing its first part).
        head = src[pos:m.start()]
        lines = head.split("\n")
        is_code = [bool(l.strip()) and not l.lstrip().startswith("#") for l in lines]
        last_code = max((i for i, c in enumerate(is_code) if c), default=-1)
        k = next((i for i in range(last_code + 1, len(lines)) if lines[i].startswith("#'")),
                 len(lines) - 1)
        block = roxygen_block(r_full[len("bf.dist."):], args, py_method,
                              parse_docstring(inspect.getdoc(methods[py_method])), r_names)
        out.append("\n".join(lines[:k] + block) + "\n")
        pos = m.start()
        report.append((r_full, py_method))
    out.append(src[pos:])
    return "".join(out), src


def wrapper_names(rdir, methods):
    """Python method -> name of the R wrapper calling it (bf.dist.<name>)."""
    names = {}
    for fname in sorted(os.listdir(rdir)):
        if not fname.endswith(".R"):
            continue
        src = open(os.path.join(rdir, fname), encoding="utf-8").read()
        for m in FUN.finditer(src):
            _, end = r_formals(src, m.end())
            call = re.search(r"\$dist\$([A-Za-z0-9_]+)\s*\(", src[end:src.find("\n}", end)])
            if call and call.group(1) in methods:
                names.setdefault(call.group(1), m.group(1)[len("bf.dist."):])
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if a file would change")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--report-latex", action="store_true",
                    help="list docstrings with LaTeX outside $...$ (to fix in Python)")
    args = ap.parse_args(argv)
    from BayesForge.Distributions.np_dists import UnifiedDist
    methods = {n: f for n, f in inspect.getmembers(UnifiedDist, inspect.isfunction)}
    if args.report_latex:
        for n, f in sorted(methods.items()):
            p = parse_docstring(inspect.getdoc(f))
            txt = "\n".join(p["description"] + sum(p["args"].values(), []) + p["returns"])
            txt = re.sub(r"\$\$.+?\$\$|\\\[.+?\\\]|\\\(.+?\\\)|\$[^$\n]+\$", "", txt, flags=re.S)
            bad = sorted(set(re.findall(r"\\[A-Za-z]+", txt)))
            if bad:
                print(f"{n}: {' '.join(bad)}")
        return 0
    changed, report = [], []
    rdir = os.path.join(args.root, "R")
    r_names = wrapper_names(rdir, methods)
    for fname in sorted(os.listdir(rdir)):
        if not fname.endswith(".R"):
            continue
        path = os.path.join(rdir, fname)
        new, old = process_file(path, methods, report, r_names)
        if new != old:
            changed.append(fname)
            if not args.check:
                open(path, "w", encoding="utf-8").write(new)
    print(f"{len(report)} wrappers documented from Python docstrings; "
          f"{len(changed)} file(s) {'would change' if args.check else 'rewritten'}.")
    if args.check and changed:
        print("\n".join("  " + c for c in changed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
