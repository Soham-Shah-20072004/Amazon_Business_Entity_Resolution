"""Generate docs/pipeline_overview.svg (hand-laid SVG, white background)."""
import sys
from html import escape

W, H = 1400, 910
C = {  # status palette: fill, stroke
    "done": ("#E6F4EA", "#2E7D32"),
    "built": ("#E3F2FD", "#1565C0"),
    "wip": ("#FFF4E0", "#C77700"),
    "next": ("#F1F3F4", "#80868B"),
    "panel": ("#FAFAFA", "#C4C7C5"),
    "neutral": ("#FFFFFF", "#9AA0A6"),
}
TXT, SUB = "#1F1F1F", "#4A4A4A"
out = []


def rect(x, y, w, h, kind, r=10, dash=False):
    f, s = C[kind]
    d = ' stroke-dasharray="6 4"' if dash else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{f}" stroke="{s}" stroke-width="2"{d}/>')


def text(x, y, s, size=13, weight="normal", color=TXT, anchor="start", italic=False):
    st = ' font-style="italic"' if italic else ""
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" '
               f'text-anchor="{anchor}"{st}>{escape(s)}</text>')


def lines(x, y, rows, size=13, lh=17, color=TXT, weight="normal", anchor="start"):
    for k, r in enumerate(rows):
        pad = len(r) - len(r.lstrip(" "))
        text(x + pad * 5, y + k * lh, r.lstrip(" "), size, weight, color, anchor)


def arrow(x1, y1, x2, y2, color="#5F6368", width=2.2):
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" '
               f'marker-end="url(#ah)"/>')


def path_arrow(d, color="#5F6368", small=True):
    m = "ahs" if small else "ah"
    out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8" marker-end="url(#{m})"/>')


def box(x, y, w, h, kind, title, rows, footer=None, title_size=15):
    rect(x, y, w, h, kind)
    text(x + 12, y + 24, title, title_size, "bold")
    lines(x + 12, y + 46, rows)
    if footer:
        f, s = C[kind]
        text(x + 12, y + h - 12, footer, 12, "bold", s)


# ---------------------------------------------------------------- header
text(20, 36, "Business entity resolution — our pipeline and where we are", 22, "bold")
for lx, kind, label in ((800, "done", "Done"), (880, "built", "Built, tested on fake data"),
                        (1100, "wip", "Building now"), (1230, "next", "Next")):
    f, s = C[kind]
    out.append(f'<rect x="{lx}" y="22" width="16" height="16" rx="3" fill="{f}" stroke="{s}" stroke-width="2"/>')
    text(lx + 22, 35, label, 13, color=SUB)

# ---------------------------------------------------------------- row A: main flow
Y, BH, BW, GAP = 64, 262, 172, 26
xs = [20 + k * (BW + GAP) for k in range(7)]
box(xs[0], Y, BW, BH, "done", "Raw data (TSV)",
    ["Train", "  2.2M S1, 10.3M S2+S3", "  + ground truth", "Test", "  1.7M S1, 10.0M S2+S3",
     "  France appears only", "  in test", "", "EDA on real data"], "done (Anumita)")
box(xs[1], Y, BW, BH, "built", "1 · Prepare",
    ["Clean every name and", "address into views:", "• ascii  (Hindi → Latin)", "• canonical", "   Pvt → private,",
     "   Gujarat → gj", "• sound skeleton", "   Delhi, दिल्ली → dl", "• numbers, PIN / ZIP"],
    "prepare_data.py")
box(xs[2], Y, BW, BH, "built", "2 · Blocking",
    ["Find likely matches", "without comparing", "every pair:", "• only inside the", "   same country",
     "• 5 search methods", "   (zoom below)", "", "≈100 candidates / S1"], "run_blocking.py")
box(xs[3], Y, BW, BH, "wip", "3 · Pre-ranker",
    ["Quick model on the", "cheap blocking scores", "keeps the best", "≈10 candidates / S1", "",
     "= candidate_pairs.tsv", "(smaller candidate", "sets rank higher)"], "building now")
box(xs[4], Y, BW, BH, "wip", "4 · Matcher",
    ["~80 similarity features", "per pair: name, address,", "numbers (conflict vs", "missing), word rarity,",
     "rank among candidates", "", "LightGBM →", "P(same business)"], "building now")
box(xs[5], Y, BW, BH, "wip", "5 · Decision",
    ["Keep pairs with P ≥ t", "(t tuned for macro F0.5)", "", "One owner: each S2/S3", "record joins ≤ 1 S1",
     "", "Empty list allowed", "(singletons score 1.0)"], "building now")
box(xs[6], Y, BW, BH, "next", "Outputs",
    ["candidate_pairs.tsv", "matching_results.tsv", "", "→ official validator", "→ leaderboard upload",
     "   (5 submissions)", "→ final zip + write-up"], "next")
for k in range(6):
    arrow(xs[k] + BW + 2, Y + BH / 2, xs[k + 1] - 4, Y + BH / 2)

# ---------------------------------------------------------------- row B left: blocking zoom
PY, PH = 356, 300
rect(20, PY, 750, PH, "panel", r=12)
text(36, PY + 28, "Zoom into step 2 · blocking", 16, "bold")
text(36, PY + 48, "Run separately for each country (US · India · France) and for each target source (S2, S3)",
     13, color=SUB)
# query card
rect(36, PY + 90, 190, 150, "neutral")
text(48, PY + 114, "One S1 record", 14, "bold")
lines(48, PY + 138, ["Smart Management", "Pvt Ltd", "1920 Purshottam Park", "Society, Vadodara,", "Gujarat"],
      13, 17, SUB)
methods = [("Nearest neighbours on name", "FAISS · top 15"),
           ("Nearest neighbours on address", "FAISS · top 15"),
           ("Nearest neighbours on skeleton", "catches Hindi / Gujarati · top 15"),
           ("Shared rare words", "e.g. “purshottam” · top 15"),
           ("Exact core name", "legal suffix dropped")]
my0, mh, mg = PY + 70, 40, 6
for k, (a, b) in enumerate(methods):
    y = my0 + k * (mh + mg)
    rect(282, y, 290, mh, "built", r=8)
    text(294, y + 17, a, 13, "bold")
    text(294, y + 33, b, 12, color=SUB)
    path_arrow(f"M226 {PY + 165} C 254 {PY + 165}, 254 {y + mh / 2}, 278 {y + mh / 2}")
    path_arrow(f"M572 {y + mh / 2} C 594 {y + mh / 2}, 594 {PY + 165}, 614 {PY + 165}")
rect(618, PY + 110, 136, 110, "built")
text(686, PY + 136, "Union", 15, "bold", anchor="middle")
lines(686, PY + 160, ["≈100 candidates", "per S1, each with", "all 5 scores"], 13, 17, SUB, anchor="middle")

# ---------------------------------------------------------------- row B right: training & tuning
rect(790, PY, 590, PH, "panel", r=12)
text(806, PY + 28, "Training and tuning (train set only)", 16, "bold")
steps = [("wip", "Label every candidate with the ground truth",
          "true match = 1, wrong look-alike = 0 → hard negatives for free"),
         ("wip", "5-fold cross-validation, grouped by S1",
          "every train S1 is scored by a model that never saw it"),
         ("wip", "Tune threshold t on those scores",
          "maximise macro F0.5, the exact competition metric"),
         ("wip", "Slice report: points lost per group",
          "singletons, multi-match, cross-script, number conflicts, …")]
sy, sh, sg = PY + 44, 48, 20
for k, (kind, a, b) in enumerate(steps):
    y = sy + k * (sh + sg)
    rect(806, y, 558, sh, kind, r=8)
    text(820, y + 21, a, 14, "bold")
    text(820, y + 38, b, 13, color=SUB)
    if k < len(steps) - 1:
        path_arrow(f"M1085 {y + sh + 2} L1085 {y + sh + sg - 2}")

# ---------------------------------------------------------------- row C: who does what
RY = 690
text(20, RY - 8, "Who does what", 16, "bold")
rect(20, RY + 4, 1360, 206, "panel", r=12)
cy, ch = RY + 34, 78
people = [(40, 250, "Claude", ["writes the code and tests it", "on fake data (no real data)"]),
          (340, 250, "GitHub · your fork", ["branch soham/pipeline-v1", "(code is already pushed)"]),
          (640, 330, "SageMaker Studio · you", ["git pull, then run the scripts", "on the real data"]),
          (1020, 330, "REPORT printed", ["you paste it back to Claude", "→ we decide the next change"])]
for x, w, t, rows in people:
    rect(x, cy, w, ch, "neutral")
    text(x + 14, cy + 25, t, 15, "bold")
    lines(x + 14, cy + 47, rows, 13, 17, SUB)
arrow(290 + 2, cy + ch / 2, 340 - 4, cy + ch / 2)
arrow(590 + 2, cy + ch / 2, 640 - 4, cy + ch / 2)
arrow(970 + 2, cy + ch / 2, 1020 - 4, cy + ch / 2)
rect(700, cy + ch + 24, 210, 44, "neutral", r=8)
text(805, cy + ch + 51, "S3 bucket: dataset zip", 13, "bold", anchor="middle")
arrow(805, cy + ch + 23, 805, cy + ch + 3)
path_arrow(f"M1185 {cy + ch + 2} L1185 {cy + ch + 84} L165 {cy + ch + 84} L165 {cy + ch + 5}", "#9AA0A6")
text(1160, cy + ch + 78, "loop until good, then submit", 12, color=SUB, anchor="end", italic=True)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Segoe UI, Helvetica, Arial, Noto Sans, Noto Sans Devanagari, sans-serif">'
       '<defs><marker id="ah" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">'
       '<path d="M0,0 L10,5 L0,10 z" fill="#5F6368"/></marker>'
       '<marker id="ahs" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto" markerUnits="userSpaceOnUse">'
       '<path d="M0,0 L7,3.5 L0,7 z" fill="#5F6368"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>' + "".join(out) + "</svg>")
open(sys.argv[1], "w", encoding="utf-8").write(svg)
print("wrote", sys.argv[1])
