import argparse
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


API_ROOT = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "Prostate-Anatomical-Edge-Cases"


def api_json(endpoint: str, parameters: dict[str, str]) -> list[dict]:
    url = f"{API_ROOT}/{endpoint}?{urllib.parse.urlencode({**parameters, 'format': 'json'})}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download_series(series_uid: str, destination: Path) -> None:
    url = f"{API_ROOT}/getImage?{urllib.parse.urlencode({'SeriesInstanceUID': series_uid})}"
    archive_path = destination.with_suffix(".zip")
    with urllib.request.urlopen(url, timeout=120) as response, archive_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe archive member: {member.filename}")
        archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one public TCIA prostate CT and RTSTRUCT")
    parser.add_argument("patient_id")
    parser.add_argument("--output-root", type=Path, default=Path("data/tcia"))
    args = parser.parse_args()
    rows = api_json("getSeries", {"Collection": COLLECTION, "PatientID": args.patient_id})
    selected = [row for row in rows if row.get("Modality") in {"CT", "RTSTRUCT"}]
    modalities = {row["Modality"] for row in selected}
    if modalities != {"CT", "RTSTRUCT"}:
        raise ValueError(f"expected CT and RTSTRUCT series; found {sorted(modalities)}")
    subject_dir = args.output_root / args.patient_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "collection": COLLECTION,
        "patient_id": args.patient_id,
        "series": selected,
    }
    (subject_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for row in selected:
        destination = subject_dir / row["Modality"].lower()
        download_series(row["SeriesInstanceUID"], destination)
        print(f"downloaded {row['Modality']} to {destination}", flush=True)


if __name__ == "__main__":
    main()
