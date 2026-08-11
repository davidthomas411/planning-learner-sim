import argparse
import http.client
import json
import shutil
import time
import urllib.parse
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path


API_ROOT = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "Prostate-Anatomical-Edge-Cases"

DOWNLOAD_STATUS_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TCIA cohort download</title><style>
body{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#202124;background:#fff}
h1{font-size:24px;font-weight:500}.track{height:28px;background:#e8eaed;border-radius:4px;overflow:hidden}
.bar{height:100%;width:0;background:#1a73e8;transition:width .35s ease}.line{display:flex;justify-content:space-between;margin:10px 0}
.detail{color:#5f6368}.complete{background:#188038}.failed{background:#d93025}
@media(prefers-color-scheme:dark){body{color:#e8eaed;background:#202124}.track{background:#3c4043}.detail{color:#bdc1c6}}
</style></head><body><h1>TCIA prostate cohort download</h1>
<div class="line"><strong id="phase">Starting</strong><span id="percent">0.0%</span></div>
<div class="track" role="progressbar" aria-label="Download progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><div class="bar" id="bar"></div></div>
<div class="line detail"><span id="count">0 / 0 patients</span><span id="eta">Estimating remaining time</span></div>
<p class="detail" id="case">Waiting for first patient.</p><p class="detail" id="updated"></p>
<script>
async function refresh(){try{const response=await fetch('progress.json?'+Date.now(),{cache:'no-store'});const p=await response.json();
const value=Math.max(0,Math.min(100,p.percent_complete||0));document.getElementById('bar').style.width=value+'%';
document.querySelector('.track').setAttribute('aria-valuenow',value.toFixed(1));document.getElementById('percent').textContent=value.toFixed(1)+'%';
document.getElementById('phase').textContent=p.status==='complete'?'Complete':p.status==='failed'?'Failed':'Running';
document.getElementById('count').textContent=p.completed+' / '+p.total+' patients';
document.getElementById('eta').textContent=p.status==='complete'?'Finished in '+format(p.elapsed_seconds):p.estimated_seconds_remaining==null?'Estimating remaining time':format(p.estimated_seconds_remaining)+' remaining';
document.getElementById('case').textContent=p.last_case||'Waiting for first patient.';
document.getElementById('updated').textContent='Local update: '+new Date().toLocaleTimeString();
document.getElementById('bar').className='bar '+(p.status==='complete'?'complete':p.status==='failed'?'failed':'');}catch(error){document.getElementById('updated').textContent='Waiting for progress file...';}}
function format(seconds){seconds=Math.max(0,Math.round(seconds||0));const minutes=Math.floor(seconds/60);const remainder=seconds%60;return minutes?minutes+'m '+remainder+'s':remainder+'s';}
refresh();setInterval(refresh,2000);
</script></body></html>"""


def write_download_progress(
    status_dir: Path | None,
    completed: int,
    total: int,
    started: float,
    last_case: str,
    status: str = "running",
) -> None:
    if status_dir is None:
        return
    elapsed = time.perf_counter() - started
    rate = completed / elapsed if completed > 0 and elapsed > 0 else 0.0
    payload = {
        "status": status,
        "completed": completed,
        "total": total,
        "percent_complete": 100.0 * completed / max(total, 1),
        "elapsed_seconds": elapsed,
        "estimated_seconds_remaining": (total - completed) / rate if rate > 0 else None,
        "last_case": last_case,
    }
    temporary = status_dir / "progress.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    destination = status_dir / "progress.json"
    for attempt in range(20):
        try:
            temporary.replace(destination)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def api_json(endpoint: str, parameters: dict[str, str]) -> list[dict]:
    url = f"{API_ROOT}/{endpoint}?{urllib.parse.urlencode({**parameters, 'format': 'json'})}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download_series(series_uid: str, destination: Path, maximum_attempts: int = 4) -> None:
    if any(destination.rglob("*.dcm")):
        print(f"using existing files in {destination}", flush=True)
        return
    url = f"{API_ROOT}/getImage?{urllib.parse.urlencode({'SeriesInstanceUID': series_uid})}"
    archive_path = destination.with_suffix(".zip")
    retry_errors = (OSError, http.client.HTTPException, urllib.error.URLError, zipfile.BadZipFile)
    for attempt in range(1, maximum_attempts + 1):
        archive_path.unlink(missing_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=120) as response, archive_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            with zipfile.ZipFile(archive_path) as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise zipfile.BadZipFile(f"CRC check failed for {bad_member}")
            break
        except retry_errors as error:
            archive_path.unlink(missing_ok=True)
            if attempt == maximum_attempts:
                raise
            delay = min(2 ** attempt, 10)
            print(
                f"download attempt {attempt} failed for {destination.name}: {error}; "
                f"retrying in {delay} seconds",
                flush=True,
            )
            time.sleep(delay)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe archive member: {member.filename}")
        archive.extractall(destination)
    archive_path.unlink()


def collection_series() -> dict[str, list[dict]]:
    rows = api_json("getSeries", {"Collection": COLLECTION})
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("Modality") in {"CT", "RTSTRUCT"}:
            grouped[str(row["PatientID"])].append(row)
    return dict(grouped)


def stratified_patient_ids(patient_ids: list[str], count: int) -> list[str]:
    """Select subjects at uniform positions in the sorted collection list."""

    if not 1 <= count <= len(patient_ids):
        raise ValueError(f"count must be from 1 to {len(patient_ids)}")
    if count == 1:
        return [patient_ids[len(patient_ids) // 2]]
    indices = [round(index * (len(patient_ids) - 1) / (count - 1)) for index in range(count)]
    return [patient_ids[index] for index in indices]


def download_subject(patient_id: str, series: list[dict], output_root: Path) -> dict:
    selected = [row for row in series if row.get("Modality") in {"CT", "RTSTRUCT"}]
    modalities = {row["Modality"] for row in selected}
    if modalities != {"CT", "RTSTRUCT"} or len(selected) != 2:
        raise ValueError(f"{patient_id}: expected one CT and one RTSTRUCT; found {sorted(modalities)}")
    subject_dir = output_root / patient_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "collection": COLLECTION,
        "patient_id": patient_id,
        "series": selected,
    }
    (subject_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    for row in selected:
        destination = subject_dir / row["Modality"].lower()
        download_series(row["SeriesInstanceUID"], destination)
        print(f"ready {row['Modality']} in {destination}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public TCIA prostate CT and RTSTRUCT subjects")
    parser.add_argument("patient_ids", nargs="*")
    parser.add_argument(
        "--count",
        type=int,
        help="Select this many subjects at uniform positions in the collection list",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/tcia"))
    parser.add_argument("--status-dir", type=Path)
    args = parser.parse_args()
    grouped = collection_series()
    collection_ids = sorted(grouped)
    requested_ids = list(args.patient_ids)
    if args.count is not None:
        requested_ids.extend(stratified_patient_ids(collection_ids, args.count))
    requested_ids = list(dict.fromkeys(requested_ids))
    if not requested_ids:
        parser.error("give at least one patient ID or use --count")
    unknown = sorted(set(requested_ids) - set(collection_ids))
    if unknown:
        raise ValueError(f"unknown patient IDs: {unknown}")
    started = time.perf_counter()
    if args.status_dir is not None:
        args.status_dir.mkdir(parents=True, exist_ok=True)
        (args.status_dir / "status.html").write_text(DOWNLOAD_STATUS_PAGE, encoding="utf-8")
    write_download_progress(args.status_dir, 0, len(requested_ids), started, "Preparing collection metadata")
    manifests = []
    for index, patient_id in enumerate(requested_ids, start=1):
        print(f"[{index}/{len(requested_ids)}] {patient_id}", flush=True)
        write_download_progress(
            args.status_dir,
            index - 1,
            len(requested_ids),
            started,
            f"Downloading {patient_id}",
        )
        try:
            manifests.append(download_subject(patient_id, grouped[patient_id], args.output_root))
        except Exception:
            write_download_progress(
                args.status_dir,
                index - 1,
                len(requested_ids),
                started,
                f"Failed while downloading {patient_id}",
                status="failed",
            )
            raise
        write_download_progress(
            args.status_dir,
            index,
            len(requested_ids),
            started,
            f"Ready: {patient_id}",
        )
    cohort_manifest = {
        "collection": COLLECTION,
        "selection": "explicit" if args.count is None else "uniform_collection_positions",
        "patient_ids": requested_ids,
        "subjects": manifests,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "cohort_download_manifest.json").write_text(
        json.dumps(cohort_manifest, indent=2), encoding="utf-8"
    )
    write_download_progress(
        args.status_dir,
        len(requested_ids),
        len(requested_ids),
        started,
        f"Ready: {requested_ids[-1]}",
        status="complete",
    )


if __name__ == "__main__":
    main()
