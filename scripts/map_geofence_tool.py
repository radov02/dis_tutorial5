#!/usr/bin/env python3

import argparse
import math
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml

# defaults:
DEFAULT_YAML = Path('/home/erik/rins/maps/maps.yaml')
_ALT_YAML    = Path(__file__).resolve().parent / "maps/maps.yaml"
COMMANDER_PY = Path(__file__).resolve().parent / "robot_commander.py"
GEOFENCE_SCRIPTS: list[Path] = [
    Path(__file__).resolve().parent / "autonomous_sweep.py",
    Path(__file__).resolve().parent / "halfautonomous_search.py",
]
CURRENT_POLYGON: list[tuple[float, float]] = [
    (-8.0, -8.5),
    ( 0.0, -8.5),
    ( 0.0,  8.5),
    (-8.0,  8.5),
]

def parse_map(yaml_path: str) -> dict:
    """
    Read a ROS map YAML + the referenced PGM/PNG and return a dict with:
      origin_x, origin_y  - map origin in metres (bottom-left corner)
      resolution           - metres per pixel
      width, height        - image size in pixels
      world_x_min/max, world_y_min/max  - world-coordinate extents
      image                - HxW numpy array (grayscale, 0-255)
      pgm_path             - resolved path to the image file
    """
    yaml_path = Path(yaml_path).resolve()
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    res  = float(meta["resolution"])
    orig = meta["origin"]  # [x, y, theta]
    ox, oy = float(orig[0]), float(orig[1])

    img_file = meta.get("image", "maps.pgm")
    if not Path(img_file).is_absolute():
        img_file = yaml_path.parent / img_file
    img_file = Path(img_file).resolve()

    # --- read PGM/PNG ---
    with open(img_file, "rb") as f:
        magic = f.readline().strip()
        if magic not in (b"P5", b"P2"):
            raise ValueError(f"Unsupported image format: {magic}")
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        maxval = int(f.readline())
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    img = raw.reshape((h, w))

    x_min = ox
    x_max = ox + w * res
    y_min = oy
    y_max = oy + h * res

    return dict(
        origin_x=ox, origin_y=oy,
        resolution=res,
        width=w, height=h,
        world_x_min=x_min, world_x_max=x_max,
        world_y_min=y_min, world_y_max=y_max,
        image=img,
        pgm_path=str(img_file),
    )


def print_map_info(m: dict):
    print("=" * 55)
    print("MAP COORDINATE INFORMATION")
    print("=" * 55)
    print(f"  Image size   : {m['width']} x {m['height']} px")
    print(f"  Resolution   : {m['resolution']} m/px")
    print(f"  Origin       : ({m['origin_x']:.4f}, {m['origin_y']:.4f})  [bottom-left]")
    print(f"  X range      : {m['world_x_min']:.4f}  ->  {m['world_x_max']:.4f}  m")
    print(f"  Y range      : {m['world_y_min']:.4f}  ->  {m['world_y_max']:.4f}  m")
    print(f"  Map image    : {m['pgm_path']}")
    print("=" * 55)


def world_to_pixel(m: dict, wx: float, wy: float) -> tuple[float, float]:
    """World (wx, wy) -> image (col, row).  Row 0 = top = y_max."""
    col = (wx - m["origin_x"]) / m["resolution"]
    row = m["height"] - 1 - (wy - m["origin_y"]) / m["resolution"]
    return col, row


def pixel_to_world(m: dict, col: float, row: float) -> tuple[float, float]:
    """Image (col, row) -> world (wx, wy)."""
    wx = m["origin_x"] + col * m["resolution"]
    wy = m["origin_y"] + (m["height"] - 1 - row) * m["resolution"]
    return wx, wy


def interactive_geofence(m: dict, initial_polygon=None, apply_to_file: Path | None = None):
    """
    Opens a matplotlib window showing the map.
    - Left-click   - add a vertex (orange dot + coordinate label)
    - Backspace    - remove the last vertex
    - Right-click  - print coordinates of that point without adding a vertex
    - Enter        - finish: print the ALLOWED_AREA_POLYGON snippet (and patch
                     robot_commander.py if apply_to_file is set)
    - Escape       - quit without output
    """
    print_map_info(m)
    print()
    print("INTERACTIVE GEOFENCE EDITOR")
    print("  Left-click   - add a vertex")
    print("  Backspace    - remove last vertex")
    print("  Enter        - finish & print polygon" +
          (" + patch robot_commander.py" if apply_to_file else ""))
    print("  Right-click  - print coords (no vertex added)")
    print("  Escape       - quit")
    if initial_polygon:
        print(f"\n  Loaded {len(initial_polygon)} existing vertices. Edit as needed.")
    print()

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.canvas.manager.set_window_title("Map Geofence Tool")

    # Map image with correct world-coordinate extent
    extent = [m["world_x_min"], m["world_x_max"], m["world_y_min"], m["world_y_max"]]
    ax.imshow(m["image"], cmap="gray", origin="lower", extent=extent, aspect="equal")

    # 1-metre grid
    x_ticks = range(math.floor(m["world_x_min"]), math.ceil(m["world_x_max"]) + 1)
    y_ticks = range(math.floor(m["world_y_min"]), math.ceil(m["world_y_max"]) + 1)
    for xv in x_ticks:
        ax.axvline(xv, color="red", lw=0.4, alpha=0.5)
    for yv in y_ticks:
        ax.axhline(yv, color="blue", lw=0.4, alpha=0.5)

    ax.set_xticks(list(x_ticks))
    ax.set_yticks(list(y_ticks))
    ax.tick_params(labelsize=7)
    ax.grid(False)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Left-click = add vertex  |  Backspace = undo  |  Enter = done  |  Esc = quit")

    # ---- mutable drawing state ----
    vertices: list[tuple[float, float]] = list(initial_polygon) if initial_polygon else []

    # We store every drawn artist in one flat list so redraw() can wipe them all cleanly.
    drawn_artists: list = []
    patch_artist: list = [None]   # separate because patches need ax.add_patch

    def redraw():
        # Clear all previously drawn artists
        for a in drawn_artists:
            try:
                a.remove()
            except Exception:
                pass
        drawn_artists.clear()
        if patch_artist[0] is not None:
            try:
                patch_artist[0].remove()
            except Exception:
                pass
            patch_artist[0] = None

        if not vertices:
            fig.canvas.draw_idle()
            return

        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]

        # Semi-transparent filled polygon (drawn first so dots sit on top)
        if len(vertices) >= 3:
            from matplotlib.patches import Polygon as MplPolygon
            poly_arr = np.array(vertices)
            p = MplPolygon(poly_arr, closed=True,
                           facecolor="orange", alpha=0.18, edgecolor="none", zorder=3)
            ax.add_patch(p)
            patch_artist[0] = p

        # Closing edge line (drawn before dots so dots overlap it)
        if len(vertices) >= 2:
            lx = xs + [xs[0]]
            ly = ys + [ys[0]]
            ln, = ax.plot(lx, ly, "-", color="orange", lw=2, zorder=4)
            drawn_artists.append(ln)

        # Individual vertex dots + coordinate labels
        for idx, (vx, vy) in enumerate(vertices):
            dot, = ax.plot(vx, vy, "o", color="orange", markersize=8, zorder=6)
            drawn_artists.append(dot)
            # Label: index + coords, slightly offset so it doesn't overlap the dot
            txt = ax.text(vx + 0.06, vy + 0.06,
                          f"{idx+1}: ({vx:.2f}, {vy:.2f})",
                          fontsize=7, color="yellow",
                          bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55),
                          zorder=7)
            drawn_artists.append(txt)

        fig.canvas.draw_idle()

    # Load initial polygon if given
    redraw()

    def on_click(event):
        if event.inaxes != ax:
            return
        wx, wy = event.xdata, event.ydata
        if wx is None:
            return
        if event.button == 3:   # right-click -> report coords only
            print(f"  Probe: ({wx:.3f}, {wy:.3f})")
            return
        if event.button == 1:
            vertices.append((wx, wy))
            print(f"  + vertex {len(vertices):2d}: ({wx:.3f}, {wy:.3f})")
            redraw()

    def on_key(event):
        if event.key in ("enter", "return"):
            if len(vertices) < 3:
                print("Need at least 3 vertices - keep clicking.")
                return
            snippet = polygon_snippet(vertices)
            print(snippet)
            if apply_to_file:
                patch_robot_commander(apply_to_file, vertices)
            plt.close(fig)
        elif event.key == "backspace":
            if vertices:
                removed = vertices.pop()
                print(f"  - removed vertex {len(vertices)+1}: ({removed[0]:.3f}, {removed[1]:.3f})")
                redraw()
            else:
                print("  (no vertices to remove)")
        elif event.key == "escape":
            print("Cancelled - no output written.")
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.tight_layout()
    plt.show()


def polygon_snippet(vertices: list[tuple[float, float]]) -> str:
    """Return the Python source block ready to paste into robot_commander.py."""
    lines = [
        "",
        "=" * 60,
        "Copy-paste this block into robot_commander.py:",
        "=" * 60,
        "ALLOWED_AREA_POLYGON: list[tuple[float, float]] = [",
    ]
    for x, y in vertices:
        lines.append(f"    ({x:.2f}, {y:.2f}),")
    lines += ["]", "=" * 60]
    return "\n".join(lines)


def patch_robot_commander(filepath: Path, vertices: list[tuple[float, float]]):
    """Replace the ALLOWED_AREA_POLYGON definition in robot_commander.py in-place."""
    if not filepath.exists():
        print(f"  [apply] File not found: {filepath}")
        return
    src = filepath.read_text()

    # Build the replacement block
    tuple_lines = "\n".join(f"    ({x:.2f}, {y:.2f})," for x, y in vertices)
    new_block = (
        "ALLOWED_AREA_POLYGON: list[tuple[float, float]] = [\n"
        + tuple_lines + "\n"
        "]"
    )

    # Match the existing ALLOWED_AREA_POLYGON = [...] block (multiline)
    pattern = re.compile(
        r"ALLOWED_AREA_POLYGON\s*:\s*list\[.*?\]\s*=\s*\[.*?\]",
        re.DOTALL,
    )
    if not pattern.search(src):
        print("  [apply] Could not locate ALLOWED_AREA_POLYGON in file - paste manually.")
        return

    new_src = pattern.sub(new_block, src)
    filepath.write_text(new_src)
    print(f"  [apply] Patched {filepath}  ({len(vertices)} vertices written)")
    print("  Rebuild with: cd /home/erik/rins && colcon build --packages-select dis_tutorial5 --symlink-install")


def patch_geofence_enabled(enabled: bool, filepaths: list[Path] | None = None):
    """Set GEOFENCE_ENABLED = True/False in each target script."""
    if filepaths is None:
        filepaths = GEOFENCE_SCRIPTS
    pattern = re.compile(r"^GEOFENCE_ENABLED\s*:\s*bool\s*=\s*(True|False)", re.MULTILINE)
    value_str = "True" if enabled else "False"
    rebuild_needed = False
    for fp in filepaths:
        if not fp.exists():
            print(f"  [geofence] File not found: {fp}")
            continue
        src = fp.read_text()
        if not pattern.search(src):
            print(f"  [geofence] Could not locate GEOFENCE_ENABLED in {fp.name} - skipping.")
            continue
        new_src = pattern.sub(f"GEOFENCE_ENABLED: bool = {value_str}", src)
        if new_src == src:
            print(f"  [geofence] {fp.name}: already {'enabled' if enabled else 'disabled'}, no change.")
        else:
            fp.write_text(new_src)
            print(f"  [geofence] {fp.name}: geofencing {'ENABLED' if enabled else 'DISABLED'}.")
            rebuild_needed = True
    if rebuild_needed:
        print("  Rebuild with: cd /home/erik/rins && colcon build --packages-select dis_tutorial5 --symlink-install")


def find_yaml(explicit: str | None) -> str:
    if explicit:
        return explicit
    if DEFAULT_YAML.exists():
        return str(DEFAULT_YAML)
    if _ALT_YAML.exists():
        return str(_ALT_YAML)
    sys.exit(
        f"Could not find maps.yaml. Pass --map /path/to/maps.yaml explicitly.\n"
        f"Tried:\n  {DEFAULT_YAML}\n  {_ALT_YAML}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map",     metavar="PATH",  help="Path to maps.yaml (auto-detected if omitted)")
    parser.add_argument("--info",    action="store_true", help="Print map coordinate info and exit (no plot)")
    parser.add_argument("--preview", action="store_true", help="Load current CURRENT_POLYGON for editing")
    parser.add_argument("--apply",   action="store_true", help="Auto-patch robot_commander.py after Enter")
    parser.add_argument("--turnoff", action="store_true", help="Disable geofencing in autonomous_sweep.py and halfautonomous_search.py")
    parser.add_argument("--turnon",  action="store_true", help="Re-enable geofencing in autonomous_sweep.py and halfautonomous_search.py")
    args = parser.parse_args()

    if args.turnoff:
        patch_geofence_enabled(False)
        return
    if args.turnon:
        patch_geofence_enabled(True)
        return

    yaml_path = find_yaml(args.map)
    m = parse_map(yaml_path)

    if args.info:
        print_map_info(m)
        return

    initial = CURRENT_POLYGON if args.preview else None
    apply_path = COMMANDER_PY if args.apply else None
    interactive_geofence(m, initial_polygon=initial, apply_to_file=apply_path)


if __name__ == "__main__":
    main()
