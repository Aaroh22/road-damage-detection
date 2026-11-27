"""Optional live smoke check. Run with the same Python environment as the app."""
from tempfile import TemporaryDirectory
from pathlib import Path
from engine import ROOT, photo
with TemporaryDirectory() as tmp:
    result = photo(ROOT/'assets/huelva-road.jpg', Path(tmp))
    print('YOLO26s · Huelva photo · confidence ≥ 0.25')
    for b in result['detections']:
        print(b['label'], f"{b['score']:.3f}")
    if not result['detections']:
        raise SystemExit('No boxes returned. Compare your runtime with the documented sample run.')
