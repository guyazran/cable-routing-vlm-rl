"""Build the self-contained robot/layout showcase page (HTML with embedded media).

Reads the routing videos and logs written by scripts/demo/record_video.py
(``--videos``), the physics-only outcome logs (``--dry``) and the layout
manifests, and writes one HTML file. Videos are re-encoded for the page
(scaled, sped up, lower bitrate); the full-resolution files stay where they are.

    python scripts/demo/build_showcase.py --videos runs/demo/robot --dry runs/demo/dry \\
        --out reports/robot_demo/showcase.html
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from cablerouting.config import ROOT, load_config
from cablerouting.scenes.generator import _stage_path, load_manifest

FAMILIES = {
    "primary10": ("Study family (for reference)", "Ten clips, 16-19 cm stages, 5-50° turns: trail-like arcs."),
    "hairpin6": ("Sharp turns", "Six clips, 15-22 cm arcs, 75-150° heading change per stage: U-turns and switchbacks."),
    "scatter5": ("Far moves off the trail", "Five clips, 28-40 cm jumps; the next clip's direction (25-80° off the axis) and "
                 "its heading (30-150°) are drawn independently, so stages are hooks and S-bends."),
    "zigzag5": ("Far and sharp", "Five clips, 24-34 cm legs with 60-120° mostly alternating turns."),
    "paper5": ("Paper-size family", "Five clips, same stage band as the study family."),
}
ORDER = ["hairpin6", "scatter5", "zigzag5", "primary10", "paper5"]
# With --media-dir, videos and images are written beside the page and linked by relative path
# (for hosts that serve a folder); otherwise they are embedded as data URIs.
MEDIA: dict = {"dir": None, "n": 0}


def emit(path: Path, mime: str, suffix: str) -> str:
    if MEDIA["dir"] is None:
        return b64(path, mime)
    import shutil

    MEDIA["n"] += 1
    name = f"media/{MEDIA['n']:02d}{suffix}"
    dst = Path(MEDIA["dir"]) / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dst)
    return name


def b64(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def page_video(src: Path, speed: float, width: int, crf: int) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "PYTHONPATH")}
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / "v.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf",
                        f"setpts=PTS/{speed},fps=30,scale={width}:-2", "-an", "-c:v", "libx264", "-preset", "slow",
                        "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
                       check=True, env=env)
        return emit(dst, "video/mp4", ".mp4")


def jpeg(path: Path, width: int = 1280, box: tuple | None = None) -> str:
    from PIL import Image

    im = Image.open(path)
    if box is not None:
        im = im.crop(box)
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "i.jpg"
        im.convert("RGB").save(p, quality=84)
        return emit(p, "image/jpeg", ".jpg")


# ------------------------------------------------------------------ figures

def layout_svg(scene, fam_cfg, w: int = 300) -> str:
    """Top-down schematic, robot at the bottom (image up = world +x, right = world -y)."""
    x0, x1, y0, y1 = -0.30, 0.86, -0.78, 0.78
    s = w / (y1 - y0)
    h = round((x1 - x0) * s)

    def P(p):
        return f"{(y1 - p[1]) * s:.1f},{(x1 - p[0]) * s:.1f}"

    parts = [f'<svg viewBox="0 0 {w} {h}" class="layout" role="img" aria-label="{html.escape(scene.scene_id)} layout">']
    cx, cy = P((0.0, 0.0)).split(",")
    for r, cls in ((0.78, "reach"), (0.20, "reach")):
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r * s:.1f}" class="{cls}"/>')
    lo, hi = (-0.154, -0.095), (0.072, 0.095)
    parts.append(f'<rect x="{(y1 - hi[1]) * s:.1f}" y="{(x1 - hi[0]) * s:.1f}" width="{(hi[1] - lo[1]) * s:.1f}" '
                 f'height="{(hi[0] - lo[0]) * s:.1f}" rx="3" class="base"/>')
    clips = scene.clip_objects()
    path = np.array(scene.initial_path)
    parts.append('<polyline class="cable" points="' + " ".join(P(p) for p in path) + '"/>')
    for a, b in zip(clips[:-1], clips[1:]):
        pts, _ = _stage_path(a.pose, b.pose, fam_cfg)
        parts.append('<polyline class="route" points="' + " ".join(P(p) for p in pts) + '"/>')
    for c in clips:
        u, n = c.pose.heading, np.array([-c.pose.heading[1], c.pose.heading[0]])
        hl, hd = c.half_length, c.dims.depth / 2 + 0.004
        corners = [c.pose.xy + sx * hd * u + sy * hl * n for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        parts.append('<polygon class="clip" points="' + " ".join(P(p) for p in corners) + '"/>')
        tip = c.pose.xy + 0.045 * u
        parts.append(f'<line class="axis" x1="{P(c.pose.xy).split(",")[0]}" y1="{P(c.pose.xy).split(",")[1]}" '
                     f'x2="{P(tip).split(",")[0]}" y2="{P(tip).split(",")[1]}"/>')
        lx, ly = P(c.pose.xy - 0.05 * n).split(",")
        parts.append(f'<text x="{lx}" y="{ly}" class="cid">{c.clip_id[1:]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def ending(l: dict) -> str:
    if l["complete"]:
        return "complete"
    d = l["decisions"][-1]["decision"] if l["decisions"] else {}
    where = f"{d.get('skill')}({d.get('target_clip')})"
    return {"collision": f"gripper hit a clip in {where}", "budget": f"decision cap during {where}"}.get(
        l["terminal"], l["terminal"])


def skill_mix(l: dict) -> str:
    n = {}
    for d in l["decisions"]:
        k = d["decision"]["skill"]
        n[k] = n.get(k, 0) + 1
    return " · ".join(f"{k} {n[k]}" for k in ("Insert", "Pull", "Flatten") if k in n)


def stage_points(cfg) -> dict[str, list[dict]]:
    out = {}
    for fam in ("primary10", "hairpin6", "scatter5", "zigzag5"):
        _, scenes = load_manifest(ROOT / "manifests" / f"{fam}_development.json")
        out[fam] = [{"scene": s.scene_id, "stage": st["stage"], "chord": round(100 * st["chord"], 1),
                     "turn": round(abs(st["relative_angle_deg"]), 1), "len": round(100 * st["arc_length"], 1)}
                    for s in scenes for st in s.stages]
    return out


# ------------------------------------------------------------------ page

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--series-1:#2a78d6;--ref:#c3c2b7;
--good:#0ca30c;--critical:#d03b3b;--cable:#c8261a;--clip:#3a3a38}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;--text-primary:#f0efec;
--text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--series-1:#3987e5;
--ref:#55544f;--cable:#e66767;--clip:#d8d7d2}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--text-primary);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1240px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 6px}h3{font-size:16px;margin:0 0 4px}
p.lead{color:var(--text-secondary);margin:0 0 18px;max-width:900px}.sub{color:var(--text-secondary)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:18px 0}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.tile .label{color:var(--text-secondary);font-size:13px}.tile .value{font-size:24px;font-weight:600}
.tile .note{color:var(--muted);font-size:12.5px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:14px;margin:14px 0}
video,img.still{width:100%;border-radius:8px;display:block;background:#000}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}@media(max-width:900px){.grid4{grid-template-columns:1fr 1fr}}
.grid4 .card{padding:8px;margin:0}
.meta{display:flex;flex-wrap:wrap;gap:6px 18px;color:var(--text-secondary);font-size:13.5px;margin:8px 0 2px}
.meta b{color:var(--text-primary);font-weight:600}
.ok::before{content:"✓ ";color:var(--good);font-weight:700}.bad::before{content:"✗ ";color:var(--critical);font-weight:700}
.speed{display:flex;gap:6px;align-items:center;font-size:13px;color:var(--text-secondary);margin-top:6px}
.speed button{font:inherit;border:1px solid var(--border);background:transparent;color:var(--text-primary);
border-radius:6px;padding:2px 9px;cursor:pointer}.speed button[aria-pressed=true]{background:var(--series-1);color:#fff;border-color:transparent}
.fam{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}@media(max-width:900px){.fam{grid-template-columns:repeat(2,1fr)}}
.lay{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:6px}
.lay .cap{font-size:12px;color:var(--text-secondary);text-align:center}
svg.layout{width:100%;height:auto}svg .reach{fill:none;stroke:var(--grid);stroke-width:1}
svg .base{fill:var(--ref)}svg .cable{fill:none;stroke:var(--cable);stroke-width:2;stroke-linecap:round;opacity:.8}
svg .route{fill:none;stroke:var(--series-1);stroke-width:2;stroke-linecap:round}
svg .clip{fill:var(--clip)}svg .axis{stroke:var(--clip);stroke-width:1.2}
svg .cid{font-size:9px;fill:var(--text-secondary);text-anchor:middle;dominant-baseline:middle}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--text-secondary);font-weight:600}td:nth-child(2){white-space:nowrap}td.num{text-align:right;font-variant-numeric:tabular-nums}
.chart{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}@media(max-width:900px){.chart{grid-template-columns:repeat(2,1fr)}}
.panel h4{margin:0 0 2px;font-size:13.5px}.panel svg{width:100%;height:auto;overflow:visible}
.panel .tick{font-size:10px;fill:var(--muted)}.panel .gl{stroke:var(--grid);stroke-width:1}
.panel .ax{stroke:var(--axis);stroke-width:1}.panel .d{fill:var(--series-1);stroke:var(--surface-1);stroke-width:2}
.panel .r{fill:var(--ref)}.panel .hit{fill:transparent;cursor:crosshair}
.legend{display:flex;gap:16px;font-size:13px;color:var(--text-secondary);margin:6px 0}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
#tip{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);border-radius:8px;
padding:6px 9px;font-size:12.5px;box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;z-index:9}
#tip b{font-size:14px}details{margin-top:8px}summary{cursor:pointer;color:var(--text-secondary);font-size:13.5px}
ul.notes li{margin:4px 0}code{font-size:13px}
"""

JS = """
document.querySelectorAll('.speed').forEach(box=>{const v=document.getElementById(box.dataset.for);
 box.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{v.playbackRate=+b.dataset.r;
 box.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',x===b));}));});
const tip=document.getElementById('tip');
function show(e,d){tip.replaceChildren();const v=document.createElement('b');v.textContent=d.turn+'° turn · '+d.chord+' cm chord';
 const s=document.createElement('div');s.textContent=d.scene+', stage to C'+String(d.stage).padStart(2,'0')+' (path '+d.len+' cm)';
 tip.append(v,s);tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px';}
document.querySelectorAll('.hit').forEach(h=>{const d=JSON.parse(h.dataset.d);
 h.addEventListener('pointermove',e=>show(e,d));h.addEventListener('pointerleave',()=>tip.style.display='none');
 h.addEventListener('focus',e=>{const r=h.getBoundingClientRect();show({clientX:r.x,clientY:r.y},d)});
 h.addEventListener('blur',()=>tip.style.display='none');});
"""


def chart_html(points: dict) -> str:
    W, H, ml, mb, mt, mr = 260, 200, 34, 28, 8, 8
    xmax, ymax = 45.0, 160.0

    def X(v):
        return ml + v / xmax * (W - ml - mr)

    def Y(v):
        return H - mb - v / ymax * (H - mb - mt)

    ref = points["primary10"]
    panels = []
    for fam in ("primary10", "hairpin6", "scatter5", "zigzag5"):
        g = [f'<svg viewBox="0 0 {W} {H}">']
        for t in (0, 50, 100, 150):
            g.append(f'<line class="gl" x1="{ml}" x2="{W - mr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                     f'<text class="tick" x="{ml - 5}" y="{Y(t) + 3:.1f}" text-anchor="end">{t}°</text>')
        for t in (0, 10, 20, 30, 40):
            g.append(f'<text class="tick" x="{X(t):.1f}" y="{H - mb + 14}" text-anchor="middle">{t}</text>')
        g.append(f'<line class="ax" x1="{ml}" x2="{W - mr}" y1="{H - mb}" y2="{H - mb}"/>')
        g.append(f'<text class="tick" x="{(ml + W) / 2:.0f}" y="{H - 1}" text-anchor="middle">chord between clips (cm)</text>')
        if fam != "primary10":
            g += [f'<circle class="r" cx="{X(d["chord"]):.1f}" cy="{Y(d["turn"]):.1f}" r="2.5"/>' for d in ref]
        for d in points[fam]:
            g.append(f'<circle class="d" cx="{X(d["chord"]):.1f}" cy="{Y(d["turn"]):.1f}" r="4"/>')
        for d in points[fam]:
            g.append(f'<circle class="hit" tabindex="0" cx="{X(d["chord"]):.1f}" cy="{Y(d["turn"]):.1f}" r="9" '
                     f"data-d='{html.escape(json.dumps(d))}'/>")
        g.append("</svg>")
        panels.append(f'<div class="panel"><h4>{fam}</h4>{"".join(g)}</div>')
    rows = "".join(f"<tr><td>{f}</td><td>{d['scene']}</td><td class=num>{d['stage']}</td><td class=num>{d['chord']}</td>"
                   f"<td class=num>{d['turn']}</td><td class=num>{d['len']}</td></tr>" for f in points for d in points[f])
    return (f'<div class="legend"><span><i style="background:var(--series-1)"></i>stages of the family named in the panel</span>'
            f'<span><i style="background:var(--ref)"></i>primary10 stages (reference)</span></div>'
            f'<div class="chart">{"".join(panels)}</div>'
            f'<details><summary>Table view ({sum(len(v) for v in points.values())} stages)</summary><table><thead><tr><th>Family</th>'
            f'<th>Scene</th><th>Stage</th><th>Chord (cm)</th><th>|Turn| (°)</th><th>Path (cm)</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></details>')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="runs/demo/robot")
    ap.add_argument("--dry", default="runs/demo/dry")
    ap.add_argument("--out", default="reports/robot_demo/showcase.html")
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--video-width", type=int, default=1440)
    ap.add_argument("--crf", type=int, default=30)
    ap.add_argument("--selection", default="", help="how the videos were chosen (shown on the page)")
    ap.add_argument("--media-dir", default=None, help="write media beside the page instead of embedding it")
    args = ap.parse_args()
    MEDIA["dir"] = args.media_dir
    cfg = load_config()
    vids = sorted(Path(args.videos).glob("*/log.json"))
    vlogs = {p.parent.name: json.loads(p.read_text()) for p in vids}
    dry = sorted(Path(args.dry).glob("*/log.json"))
    dlogs = [json.loads(p.read_text()) for p in dry]

    # ---- hero tiles
    done = [l for l in dlogs if l["complete"]]
    ik = max([l.get("ik_max_pos_err_mm", 0) for l in list(vlogs.values()) + dlogs
              if l["family"] in ("hairpin6", "scatter5", "zigzag5")] or [0])
    ik_p10 = max([l.get("ik_max_pos_err_mm", 0) for l in vlogs.values() if l["family"] == "primary10"] or [0])
    new = [l for l in dlogs if l["family"] in ("hairpin6", "scatter5", "zigzag5")]
    tiles = [
        ("Robot", "Franka Panda", "Isaac Sim asset at the virtual base, posed by IK on the grasp body"),
        ("Videos", str(len(vlogs)), f"{len({l['family'] for l in vlogs.values()})} layout families, wide view + close-up"),
        ("Physics-only completions", f"{sum(l['complete'] for l in new)} / {len(new)}",
         "new-family dev scenes, reference controller + heuristic insertion"),
        ("Largest IK residual, new families", f"{ik:.1f} mm" if ik else "n/a",
         f"hand vs grasp body over every frame; primary10 video up to {ik_p10:.0f} mm (targets beyond the arm's reach)"),
    ]
    out = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>Cable routing: Franka robot and new clip layouts</title><style>{CSS}</style></head><body><main>"]
    out.append("<h1>Cable routing in simulation: Franka robot and new clip layouts</h1>"
               "<p class=lead>The floating gripper is replaced by a Franka Emika Panda, as in the paper's simulation media. "
               "Each video shows the whole robot and every clip on the left, and a close-up of the clip being worked on on the right. "
               "Three new layout families need sharper turns and longer moves between clips than the trail-like study family. "
               "The controller is the ground-truth reference controller with the <b>heuristic stand-in</b> insertion policy, "
               "because no RL checkpoint has been trained yet. The VLM is not involved.</p>")
    out.append('<div class=tiles>' + "".join(
        f'<div class=tile><div class=label>{a}</div><div class=value>{b}</div><div class=note>{c}</div></div>'
        for a, b, c in tiles) + "</div>")

    # ---- videos by family
    out.append("<h2>Routing videos</h2><p class=sub>" + (html.escape(args.selection) + " " if args.selection else "") + "Embedded at "
               f"{args.speed:g}× real time and reduced resolution; the buttons change the playback speed. The poster frame is an "
               "insertion moment. Full-resolution real-time files are "
               f"in <code>{html.escape(args.videos)}/&lt;scene&gt;/routing.mp4</code>.</p>")
    for fam in ORDER:
        logs = [(sid, l) for sid, l in vlogs.items() if l["family"] == fam]
        if not logs:
            continue
        title, desc = FAMILIES[fam]
        out.append(f"<h3 style='margin-top:26px'>{title}: <code>{fam}</code></h3><p class=sub style='margin:0'>{desc}</p>")
        for sid, l in logs:
            vid = f"v_{sid.replace('-', '_')}"
            src = page_video(Path(args.videos) / sid / "routing.mp4", args.speed, args.video_width, args.crf)
            ins = sorted((Path(args.videos) / sid).glob("insert_*.jpg"))
            poster_path = ins[len(ins) // 2] if ins else Path(args.videos) / sid / "start.jpg"
            poster = jpeg(poster_path, args.video_width) if poster_path.exists() else ""
            status = "ok" if l["complete"] else "bad"
            turns = ", ".join(f"{abs(s['relative_angle_deg']):.0f}°" for s in l["stages"])
            chords = ", ".join(f"{100 * s['chord']:.0f}" for s in l["stages"])
            sp = args.speed
            out.append(f"<div class=card><video id={vid} controls muted playsinline preload=none poster='{poster}' "
                       f"src='{src}'></video>"
                       f"<div class=speed data-for={vid}>playback <button data-r={1 / sp:g}>real time</button>"
                       f"<button data-r=1 aria-pressed=true>{sp:g}×</button><button data-r=2>{2 * sp:g}×</button></div>"
                       f"<div class=meta><span class={status}><b>{sid}</b>: threaded {l['valid_prefix']}/{l['n_clips']}, "
                       f"{ending(l)}</span>"
                       f"<span><b>{l['decisions_used']}</b> skill calls ({skill_mix(l)}), <b>{l['physical_ops']}</b> pick-and-places</span>"
                       f"<span><b>{l['sim_time_s']:.0f} s</b> simulated</span>"
                       f"<span>turns {turns}</span><span>chords {chords} cm</span>"
                       f"<span>IK residual ≤ {l.get('ik_max_pos_err_mm', float('nan')):.2f} mm</span></div></div>")

    # ---- robot stills
    stills = []
    for sid, l in vlogs.items():
        for name in ("start", "final") + tuple(sorted(p.stem for p in (Path(args.videos) / sid).glob("insert_*.jpg"))):
            p = Path(args.videos) / sid / f"{name}.jpg"
            if p.exists():
                stills.append((sid, name, p))
    pick = []  # one insertion close-up per video, from the middle of its route, families in page order
    for fam in ORDER:
        for sid, l in vlogs.items():
            ins = [x for x in stills if x[0] == sid and x[1].startswith("insert_")]
            if l["family"] == fam and ins:
                pick.append(ins[len(ins) // 2])
    pick = pick[:4]
    if pick:
        out.append("<h2>Close-ups at the moment of insertion</h2><p class=sub>The close-up panel at the frame where the "
                   "hand is lowest and nearest the clip during a successful Insert. The camera looks from the side opposite "
                   "the hand, so the cable comes through the clip opening towards the viewer.</p><div class=grid4>")
        for sid, name, p in pick:
            out.append(f"<div class=card><img class=still src='{jpeg(p, 704, box=(1600, 0, 2304, 900))}' alt='{sid} {name}'>"
                       f"<div class=meta><b>{sid}</b> during {name.replace('insert_', 'Insert(')})</div></div>")
        out.append("</div>")

    # ---- layout families
    out.append("<h2>Layout families</h2><p class=sub>Top-down schematics, robot pedestal at the bottom and its "
               "0.20-0.78 m top-down reach as rings. Blue: the cable path the route requires; red: the initial cable; "
               "ticks mark each clip's insertion direction.</p>")
    for fam in ("hairpin6", "scatter5", "zigzag5", "primary10"):
        _, scenes = load_manifest(ROOT / "manifests" / f"{fam}_development.json")
        title, desc = FAMILIES[fam]
        out.append(f"<h3 style='margin-top:18px'>{title}: <code>{fam}</code></h3><p class=sub style='margin:0 0 6px'>{desc}</p>"
                   "<div class=fam>")
        for s in scenes[:4]:
            out.append(f"<div class=lay>{layout_svg(s, cfg.family(fam))}<div class=cap>{s.scene_id}</div></div>")
        out.append("</div>")
    out.append("<h3 style='margin-top:22px'>Stage geometry by family</h3><p class=sub style='margin:0 0 4px'>"
               "Every development stage: distance between consecutive clips against the heading change the cable must make. "
               "The new families sit outside the study family's band.</p>")
    out.append(chart_html(stage_points(cfg)))

    # ---- outcomes table
    if dlogs:
        n_new = sum(l["family"] in ("hairpin6", "scatter5", "zigzag5") for l in dlogs)
        n_old = {f: sum(l["family"] == f for l in dlogs) for f in ("primary10", "paper5")}
        out.append("<h2>Physics-only outcomes</h2><p class=sub>"
                   f"All {n_new} new-family development scenes, plus {n_old['primary10']} primary10 and {n_old['paper5']} paper5 "
                   "scenes, simulated without rendering, each as the first episode after loading. Same controller and heuristic "
                   "insertion as the videos, with a cap of 6 skill calls per clip for the new families. These are development "
                   "checks, not study results.</p>")
        fams = [f for f in ORDER if any(l["family"] == f for l in dlogs)]
        summ = []
        for f in fams:
            ls = [l for l in dlogs if l["family"] == f]
            ins = sum(1 for l in ls if l["terminal"] == "collision" and l["decisions"][-1]["decision"]["skill"] == "Insert")
            flat = sum(1 for l in ls for d in l["decisions"] if d["decision"]["skill"] == "Flatten")
            summ.append(f"<li><b>{f}</b>: {sum(l['complete'] for l in ls)}/{len(ls)} complete; "
                        f"{ins} ended with the heuristic Insert's gripper hitting a clip; {flat} Flatten calls in total</li>")
        out.append("<ul class=notes>" + "".join(summ) + "</ul>")
        rows = []
        for l in sorted(dlogs, key=lambda l: (ORDER.index(l["family"]) if l["family"] in ORDER else 9, l["scene_id"])):
            rows.append(f"<tr><td>{l['family']}</td><td>{l['scene_id']}</td><td class={'ok' if l['complete'] else 'bad'}>"
                        f"{l['valid_prefix']}/{l['n_clips']}</td><td>{ending(l)}</td><td>{skill_mix(l)}</td>"
                        f"<td class=num>{l['decisions_used']}</td>"
                        f"<td class=num>{l['physical_ops']}</td><td class=num>{l['sim_time_s']:.0f}</td>"
                        f"<td class=num>{l.get('ik_max_pos_err_mm', float('nan')):.2f}</td></tr>")
        out.append("<div class=card><table><thead><tr><th>Family</th><th>Scene</th><th>Threaded</th><th>How it ended</th>"
                   "<th>Skill mix</th><th>Skill calls</th><th>Pick-and-places</th><th>Sim s</th><th>IK residual mm</th></tr></thead>"
                   f"<tbody>{''.join(rows)}</tbody></table></div>")

    ends = {}
    for l in dlogs:
        k = "complete" if l["complete"] else f"{l['terminal']}:{l['decisions'][-1]['decision']['skill']}"
        ends[k] = ends.get(k, 0) + 1
    fails = len(dlogs) - ends.get("complete", 0)
    coll_other = sum(v for k, v in ends.items() if k.startswith("collision:") and k != "collision:Insert")
    cap = sum(v for k, v in ends.items() if k.startswith("budget:"))
    why = (f"Of the {fails} physics-only runs that did not complete, {ends.get('collision:Insert', 0)} ended when the "
           f"heuristic Insert's gripper overlapped a clip, {coll_other} with a gripper-clip collision in Pull or Flatten, "
           f"and {cap} hit the decision cap while Insert or Flatten kept repeating.") if dlogs else ""
    p10 = [l for l in vlogs.values() if l["family"] == "primary10"]
    reach_note = (f"In the primary10 video the hand is up to {max(l.get('ik_max_pos_err_mm', 0) for l in p10):.0f} mm "
                  "off the grasp body at moments when the skills ask for a grasp outside the arm's reach "
                  "(the IK returns the closest pose); elsewhere it is exact.") if p10 else ""
    out.append(f"<h2>What changed and what to keep in mind</h2><ul class=notes>"
               f"{NOTES.format(why=why, reach_note=reach_note)}</ul>")
    out.append(f"<div id=tip role=tooltip></div><script>{JS}</script></main></body></html>")
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(out))
    print(f"[showcase] {dst} {dst.stat().st_size / 1e6:.1f} MB", flush=True)


NOTES = """
<li><b>The arm follows the existing grasp body, so physics is unchanged.</b> Before every rendered frame, IK places the Panda's
fingertips on the kinematic grasp body (top-down, closing axis = gripper yaw); the arm has no colliders or joints in PhysX.
FK matches the Isaac asset to 0.4 µm. A static box at the pedestal keeps the cable out of the base.</li>
<li><b>Free-space moves are shown, not simulated.</b> The executor teleports the gripper between grasps because those moves
cannot touch the cable. The video interpolates them around the base, and the fingers open and close over 0.3 s. Every
cable-holding motion is simulated step by step and played in real time (before the page's speed-up).</li>
<li><b>Observation images are unchanged.</b> The planner's top-down image and crops hide the arm, as a wrist camera at the
observation pose would not see it.</li>
<li><b>Reach.</b> A real Panda reaches 0.20-0.78 m top-down at table height. The new families are generated inside that band.
The study family (primary10) uses a 0.15-0.90 m reach model, and none of its development scenes fits fully inside the real
reach once the 16 cm insertion regions are counted. {reach_note}</li>
<li><b>Controller.</b> The reference controller with the heuristic insertion stand-in, not the trained RL policy or the VLM.
Failures on the new families show where these scripted skills break, not where the study system would.</li>
<li><b>Why runs fail.</b> {why} The heuristic Insert grasps the head even when the head lies against the clip,
or slides its fingers past the clip at table height, and the gripper then overlaps a clip (a terminal collision under the
study rule). At sharp turns Flatten often has to repeat: the bent cable springs back and leaves the head misaligned.
These are the limits of the scripted stand-ins on the new layouts, and a target for the RL insertion policy.</li>
<li><b>Outcomes are sensitive to tiny state differences.</b> Rendering neither steps nor perturbs physics (checked:
identical cable states with and without rendering). But the first episode after a scene is loaded ends about 0.1 mm away
from later replays of the same episode, which are bitwise identical to each other, and the cable dynamics amplify such
differences. A video's run can therefore end differently from the physics-only run of the same scene; each video reports
its own outcome.</li>
<li><b>Scope.</b> The new families are development/demo families (deviation L5) and do not enter the five-vs-ten screen.</li>
"""


if __name__ == "__main__":
    main()
