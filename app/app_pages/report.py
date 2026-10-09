import re

import streamlit as st

import lib

lib.require_artifacts()
report = (lib.ROOT / "reports" / "REPORT.md").read_text(encoding="utf-8")

st.caption("The full auto-generated analysis report (`reports/REPORT.md`). Every number is computed by the pipeline.")

IMG = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)\s*$")
buffer: list[str] = []


def flush() -> None:
    if buffer:
        st.markdown("\n".join(buffer))
        buffer.clear()


for line in report.splitlines():
    hit = IMG.match(line.strip())
    if hit:
        flush()
        st.image(str(lib.ROOT / "reports" / hit["src"]), caption=hit["alt"])
    else:
        buffer.append(line)
flush()
