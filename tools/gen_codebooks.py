#!/usr/bin/env python3
"""Regenerate the built-in trellis codebook block inside sj_kvarn.h.

The codebooks are the trained token-axis tables (3-bit L=9, 2-bit L=8) for K and V,
stored as IEEE fp16 bit patterns. The block between the two marker lines in
sj_kvarn.h is replaced; nothing else in the header is touched.

usage: gen_codebooks.py <source header with *_TOK_* init lists> <sj_kvarn.h>
"""
import re
import sys

BEGIN = "/* ---- BEGIN GENERATED CODEBOOKS (tools/gen_codebooks.py) ---- */"
END = "/* ---- END GENERATED CODEBOOKS ---- */"

TABLES = [
    # (source macro, output macro, expected entries)
    ("KVARN_CB3_TOK_K_INIT", "SJKVARN_CB3_K_INIT", 512),
    ("KVARN_CB3_TOK_V_INIT", "SJKVARN_CB3_V_INIT", 512),
    ("KVARN_CB2_TOK_K_INIT", "SJKVARN_CB2_K_INIT", 256),
    ("KVARN_CB2_TOK_V_INIT", "SJKVARN_CB2_V_INIT", 256),
]
SHAS = [("KVARN_CB3_TOK_SHA256", "SJKVARN_CB3_SHA256"), ("KVARN_CB2_TOK_SHA256", "SJKVARN_CB2_SHA256")]


def parse(src, name):
    m = re.search(r"#define\s+" + name + r"\s*\{(.*?)\}", src, re.S)
    if not m:
        sys.exit(f"missing {name}")
    vals = re.findall(r"0x[0-9a-fA-F]{4}", m.group(1))
    return [v.lower() for v in vals]


def main():
    src = open(sys.argv[1]).read()
    out = [BEGIN]
    for s, d in SHAS:
        m = re.search(r"#define\s+" + s + r"\s+\"([0-9a-f]+)\"", src)
        out.append(f'#define {d} "{m.group(1)}" /* sha256 of the uint16 K[] then V[] table file */')
    for s, d, n in TABLES:
        vals = parse(src, s)
        if len(vals) != n:
            sys.exit(f"{s}: {len(vals)} entries, expected {n}")
        out.append(f"#define {d} {{ \\")
        for i in range(0, n, 16):
            row = ", ".join(vals[i:i + 16])
            out.append(f"    {row}{',' if i + 16 < n else ''} \\")
        out.append("}")
    out.append(END)
    hdr_path = sys.argv[2]
    hdr = open(hdr_path).read()
    a, b = hdr.index(BEGIN), hdr.index(END) + len(END)
    open(hdr_path, "w").write(hdr[:a] + "\n".join(out) + hdr[b:])


if __name__ == "__main__":
    main()
