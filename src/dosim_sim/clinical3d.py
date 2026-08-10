"""Import a small clinical-anatomy test set into the synthetic planner interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.path import Path as PolygonPath

from .volume3d import SyntheticCase3D


ROI_ALIASES = {
    "prostate": "prostate",
    "bladder": "bladder",
    "rectum": "rectum",
    "femur_head_l": "left_femoral_head",
    "femur_head_r": "right_femoral_head",
}


def _normalized_roi_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _pad_to_physical_cube(
    arrays: list[np.ndarray],
    spacing: tuple[float, float, float],
) -> list[np.ndarray]:
    """Pad aligned z-row-column arrays to a common physical field of view."""

    shape = np.asarray(arrays[0].shape)
    physical = shape * np.asarray(spacing)
    cube_length = float(physical.max())
    target_shape = np.ceil(cube_length / np.asarray(spacing)).astype(int)
    padding = []
    for current, target in zip(shape, target_shape, strict=True):
        total = int(target - current)
        padding.append((total // 2, total - total // 2))
    return [np.pad(array, padding, mode="constant") for array in arrays]


def _resample_masks(masks: list[np.ndarray], output_size: int) -> list[np.ndarray]:
    try:
        from scipy.ndimage import zoom
    except ImportError as exc:  # pragma: no cover - optional clinical dependency
        raise RuntimeError("Install the clinical dependencies to import DICOM data") from exc
    factors = tuple(output_size / value for value in masks[0].shape)
    return [zoom(mask.astype(np.uint8), factors, order=0, prefilter=False).astype(bool) for mask in masks]


def load_tcia_prostate_case(root: Path, grid_size: int = 64, ptv_margin_mm: float = 5.0) -> SyntheticCase3D:
    """Load one TCIA prostate CT/RTSTRUCT subject as a contour-only planning case."""

    if grid_size < 24:
        raise ValueError("grid_size must be at least 24")
    try:
        import pydicom
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - optional clinical dependency
        raise RuntimeError("Install the clinical dependencies to import DICOM data") from exc

    datasets = [pydicom.dcmread(path) for path in Path(root).rglob("*.dcm")]
    ct = [dataset for dataset in datasets if dataset.Modality == "CT"]
    rtstruct = [dataset for dataset in datasets if dataset.Modality == "RTSTRUCT"]
    if not ct or len(rtstruct) != 1:
        raise ValueError("the subject must contain one CT series and one RTSTRUCT")

    orientation = np.asarray(ct[0].ImageOrientationPatient, dtype=float)
    column_direction = orientation[:3]
    row_direction = orientation[3:]
    normal = np.cross(column_direction, row_direction)
    ct.sort(key=lambda dataset: float(np.dot(np.asarray(dataset.ImagePositionPatient, dtype=float), normal)))
    positions = np.asarray([np.asarray(dataset.ImagePositionPatient, dtype=float) for dataset in ct])
    slice_coordinates = positions @ normal
    row_spacing, column_spacing = map(float, ct[0].PixelSpacing)
    slice_spacing = float(np.median(np.diff(slice_coordinates)))
    spacing = (abs(slice_spacing), row_spacing, column_spacing)
    ct_series = str(ct[0].SeriesInstanceUID)
    referenced_series = str(
        rtstruct[0]
        .ReferencedFrameOfReferenceSequence[0]
        .RTReferencedStudySequence[0]
        .RTReferencedSeriesSequence[0]
        .SeriesInstanceUID
    )
    if referenced_series != ct_series:
        raise ValueError("RTSTRUCT does not reference the downloaded CT series")

    pixels = np.stack([
        dataset.pixel_array.astype(np.float32) * float(dataset.RescaleSlope) + float(dataset.RescaleIntercept)
        for dataset in ct
    ])
    threshold_body = pixels > -500.0
    body = threshold_body

    roi_number_to_name = {
        int(item.ROINumber): ROI_ALIASES.get(_normalized_roi_name(str(item.ROIName)))
        for item in rtstruct[0].StructureSetROISequence
    }
    masks = {name: np.zeros_like(body) for name in ROI_ALIASES.values()}
    for roi in rtstruct[0].ROIContourSequence:
        name = roi_number_to_name.get(int(roi.ReferencedROINumber))
        if name is None:
            continue
        for contour in getattr(roi, "ContourSequence", []):
            points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            plane = points @ normal
            slice_index = int(np.argmin(np.abs(slice_coordinates - float(plane.mean()))))
            delta = points - positions[slice_index]
            columns = delta @ column_direction / column_spacing
            rows = delta @ row_direction / row_spacing
            row_min = max(0, int(np.floor(rows.min())))
            row_max = min(body.shape[1] - 1, int(np.ceil(rows.max())))
            column_min = max(0, int(np.floor(columns.min())))
            column_max = min(body.shape[2] - 1, int(np.ceil(columns.max())))
            grid_rows, grid_columns = np.mgrid[row_min : row_max + 1, column_min : column_max + 1]
            samples = np.column_stack((grid_columns.ravel() + 0.5, grid_rows.ravel() + 0.5))
            polygon = PolygonPath(np.column_stack((columns, rows))).contains_points(samples)
            polygon = polygon.reshape(grid_rows.shape)
            view = masks[name][slice_index, row_min : row_max + 1, column_min : column_max + 1]
            view ^= polygon

    missing = [name for name, mask in masks.items() if not mask.any()]
    if missing:
        raise ValueError(f"required contours are missing or empty: {missing}")
    roi_union = np.logical_or.reduce(list(masks.values()))
    roi_center = np.argwhere(roi_union).mean(axis=0)[1:]
    body_sections = []
    for section in threshold_body:
        labels, count = ndimage.label(section)
        if count:
            seed_row, seed_column = np.rint(roi_center).astype(int)
            selected = int(labels[seed_row, seed_column])
            if selected == 0:
                objects = ndimage.find_objects(labels)
                centers = []
                for index, bounds in enumerate(objects, start=1):
                    if bounds is None:
                        continue
                    center = np.array([(item.start + item.stop - 1) / 2 for item in bounds])
                    size = int(np.count_nonzero(labels[bounds] == index))
                    if size >= 500:
                        centers.append((float(np.linalg.norm(center - roi_center)), index))
                selected = min(centers)[1] if centers else 0
            section = labels == selected if selected else np.zeros_like(section)
        body_sections.append(ndimage.binary_fill_holes(section))
    body = np.stack(body_sections)
    prostate = masks["prostate"]
    distance = ndimage.distance_transform_edt(~prostate, sampling=spacing)
    target = prostate | (distance <= ptv_margin_mm)
    target &= body
    femoral_heads = masks["left_femoral_head"] | masks["right_femoral_head"]

    union = target | masks["bladder"] | masks["rectum"] | femoral_heads
    indices = np.argwhere(union)
    margin = np.ceil(np.array([40.0, 80.0, 80.0]) / np.asarray(spacing)).astype(int)
    lower = np.maximum(indices.min(axis=0) - margin, 0)
    upper = np.minimum(indices.max(axis=0) + margin + 1, np.asarray(body.shape))
    slices = tuple(slice(int(low), int(high)) for low, high in zip(lower, upper, strict=True))
    cropped = [body[slices], target[slices], masks["bladder"][slices], masks["rectum"][slices], femoral_heads[slices]]
    resampled = _resample_masks(_pad_to_physical_cube(cropped, spacing), grid_size)
    body_out, target_out, bladder_out, rectum_out, femoral_out = [
        np.transpose(mask, (2, 1, 0)) for mask in resampled
    ]
    axis = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    return SyntheticCase3D(
        case_id=f"tcia-{Path(root).name}",
        seed=0,
        axis=axis,
        body=body_out,
        target=target_out,
        oars=(bladder_out, rectum_out, femoral_out),
        oar_limits=(0.48, 0.44, 0.34),
        structure_names=("bladder", "rectum", "femoral_heads"),
        anatomy="tcia_prostate",
        difficulty="external",
        available_beams=tuple(range(12)),
    )
