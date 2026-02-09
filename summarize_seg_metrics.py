import argparse, re, sys
from collections import OrderedDict

ANSI = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
NUM  = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?")

def strip_ansi(s: str) -> str:
    return ANSI.sub("", s)

def parse_args():
    ap = argparse.ArgumentParser("Summarize aAcc/mIoU/mAcc from log.")
    ap.add_argument("logfile")
    ap.add_argument("--datasets", default="", help="comma order filter")
    ap.add_argument("--delimiter", default="----------", help="block end marker")
    ap.add_argument("--digits", type=int, default=4)
    return ap.parse_args()

def is_number(tok: str) -> bool:
    return bool(NUM.fullmatch(tok))

def main():
    a = parse_args()
    want_order = [s.strip() for s in a.datasets.split(",") if s.strip()]
    delim = a.delimiter

    results = OrderedDict()   # ds -> dict(aAcc=..., mIoU=..., mAcc=...)
    order   = []              # output order
    curr    = None            # current dataset
    buf     = {"aAcc": None, "mIoU": None, "mAcc": None}

    with open(a.logfile, "rb") as f:
        for raw in f:
            line = strip_ansi(raw.decode("utf-8", "ignore")).replace("\r","").rstrip("\n")
            line = line.replace("\xa0", " ")

            if line.startswith("=== DATASET:"):
                if curr is not None and curr not in results:
                    results[curr] = buf.copy()
                    if curr not in order: order.append(curr)

                tmp = line.split("DATASET:", 1)[1].strip()
                if "|" in tmp:
                    ds = tmp.split("|", 1)[0].strip()
                else:
                    ds = tmp.strip()

                ds = ds.replace("=", "").strip()

                curr = ds
                buf = {"aAcc": None, "mIoU": None, "mAcc": None}
                continue

            if line.strip() == delim:
                if curr is not None:
                    results[curr] = buf.copy()
                    if curr not in order: order.append(curr)
                curr = None
                buf = {"aAcc": None, "mIoU": None, "mAcc": None}
                continue

            if curr is not None:
                for key in ("aAcc", "mIoU", "mAcc"):
                    pos = line.find(f"{key}:")
                    if pos >= 0:
                        tail = line[pos + len(key) + 1:]
                        for tok in tail.strip().split():
                            if is_number(tok):
                                buf[key] = float(tok)
                                break

    if curr is not None:
        results[curr] = buf.copy()
        if curr not in order: order.append(curr)

    if want_order:
        order = [ds for ds in want_order if ds in results]

    def fmt(x): return f"{x:.{a.digits}f}" if isinstance(x, float) else "N/A"
    print("=============== Evaluation Summary ===============")
    print(f"{'Dataset':16} {' aAcc':12} {' mIoU':12} {' mAcc':12}")
    print(f"{'-'*16:16} {'-'*7:12} {'-'*7:12} {'-'*7:12}")

    sa=si=sc=0.0; ca=ci=cc=0
    for ds in order:
        met = results.get(ds, {})
        va, vi, vc = met.get("aAcc"), met.get("mIoU"), met.get("mAcc")
        if isinstance(va, float): sa += va; ca += 1
        if isinstance(vi, float): si += vi; ci += 1
        if isinstance(vc, float): sc += vc; cc += 1
        print(f"{ds:16} {fmt(va):12} {fmt(vi):12} {fmt(vc):12}")

    print("-"*50)
    mean_a = sa/ca if ca else None
    mean_i = si/ci if ci else None
    mean_c = sc/cc if cc else None
    print(f"{'MEAN':16} {fmt(mean_a):12} {fmt(mean_i):12} {fmt(mean_c):12}")

if __name__ == "__main__":
    sys.exit(main())
