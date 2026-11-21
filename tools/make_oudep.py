"""Build the Diffical model dependency package (.oudep, OpenUtau format)."""
import argparse, hashlib, json, pathlib, zipfile

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="dir with assembler.onnx/vocoder.onnx/hnsep.onnx")
    ap.add_argument("--id", default="diffical-models")
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    models = pathlib.Path(args.models)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = ["assembler.onnx", "vocoder.onnx", "hnsep.onnx"]
    for f in files:
        if not (models / f).exists():
            raise SystemExit(f"missing {models / f}")
    manifest = {}
    for f in files:
        h = hashlib.sha256((models / f).read_bytes()).hexdigest()
        manifest[f] = {"sha256": h, "size": (models / f).stat().st_size}
    yaml = f"""id: {args.id}
version: {args.version}
description: Diffical model weights (assembler + vocoder + HN-SEP separator)
@class: DifficalModels
"""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("oudep.yaml", yaml)
        z.writestr("models/manifest.json", json.dumps(manifest, indent=2))
        for f in files:
            z.write(models / f, f"models/{f}")
        for extra in ("LICENSE-WEIGHTS.md", "NOTICE", "LICENSE"):
            p = pathlib.Path(__file__).resolve().parents[1] / extra
            if p.exists():
                z.write(p, extra)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
