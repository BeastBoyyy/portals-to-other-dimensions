"""config.yaml -> td/config.json (TouchDesigner's Python may not ship PyYAML; JSON always works).

    python tools/export_td_config.py
"""
import json, pathlib, sys, dataclasses

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from handcore import config as hcfg   # noqa: E402

TD = {
    "repo": str(ROOT),
    "frame_divider": 2,          # project fps 60 / 2 = 30 Hz frame writer
    "frame_delay_s": 1 / 60,     # numpyArray(delayed=True) returns the previous cook
    "camera_fps": 30,
    "scene_fade_s": 0.3,
    "parallax_pos": 0.06,
    "parallax_tilt": 0.05,
    "udp_port": 9000,
    "quality_cell_px": 3.0,      # 3 px at 1280x720 = 102,400 points (4 = 57,600; 6 = ~25k)
}


def main():
    cfg = hcfg.load(str(ROOT / "config.yaml"))
    out = dataclasses.asdict(cfg)
    out["td"] = TD
    p = ROOT / "td" / "config.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
