"""HD-BET brain masking module for FORGE Studio.

Runs HD-BET (High Definition Brain Extraction Tool) on reconstructed MRI
magnitude images and produces a binary brain mask NIfTI file.

Expects config keys:
  - input_paths: list of NIfTI files (magnitude images); uses the first one
  - output_dir:  directory for output mask file
  - params:      optional overrides (use_tta, device)

Returns: [output_mask_path]
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from forge_studio import forge_log, forge_read_config, forge_param


def _robust_nii_load(nii_path: str) -> nib.Nifti1Image:
    """Load a NIfTI file, falling back to explicit Nifti1 format if auto-detect fails."""
    try:
        return nib.load(nii_path)
    except nib.filebasedimages.ImageFileError:
        # MATLAB's save_untouch_nii sometimes writes files nibabel can't auto-detect.
        # Try loading as explicit Nifti1.
        return nib.Nifti1Image.from_filename(nii_path)


def _ensure_nii_gz(nii_path: str, tmp_dir: str) -> str:
    """HD-BET requires .nii.gz input. Convert .nii → .nii.gz if needed."""
    if nii_path.endswith(".nii.gz"):
        return nii_path
    img = _robust_nii_load(nii_path)
    gz_path = os.path.join(tmp_dir, Path(nii_path).stem + ".nii.gz")
    nib.save(img, gz_path)
    return gz_path


def _select_magnitude_volume(nii_path: str, tmp_dir: str) -> str:
    """If the input is 4-D, extract the first volume (frame) for masking."""
    img = _robust_nii_load(nii_path)
    data = np.asanyarray(img.dataobj)
    if data.ndim <= 3:
        return nii_path
    forge_log("info", "Input is %d-D — extracting first volume for masking", data.ndim)
    vol = data[..., 0]
    out = os.path.join(tmp_dir, "mag_vol0.nii.gz")
    nib.save(nib.Nifti1Image(vol, img.affine, img.header), out)
    return out


def run(config_path: str) -> list[str]:
    config = forge_read_config(config_path)
    input_paths: list[str] = config["input_paths"]
    output_dir: str = config["output_dir"]
    params = config.get("params", {})

    # Find the best input for brain extraction. Priority:
    #   1. Overlay image (_overlay.nii) from the preparation stage
    #   2. T2-weighted stack (_t2stack) from wave extraction
    #   3. Magnitude image (_mag)
    #   4. Any NIfTI that isn't a wave/phase/mask/params file
    #   5. First NIfTI file as last resort

    # Build candidate list: start with chained input_paths, then scan the
    # job output directory for overlay images from earlier stages.
    all_candidates: list[str] = list(input_paths)

    # The job output dir is the parent of this stage's output_dir
    # (output_dir = {jobDir}/stage_N_xxx/, so parent = jobDir).
    job_dir = os.path.dirname(output_dir)
    if os.path.isdir(job_dir):
        for stage_dir_name in sorted(os.listdir(job_dir)):
            stage_dir = os.path.join(job_dir, stage_dir_name)
            if not os.path.isdir(stage_dir):
                continue
            for fname in os.listdir(stage_dir):
                fpath = os.path.join(stage_dir, fname)
                if fpath not in all_candidates and (
                    fname.endswith(".nii") or fname.endswith(".nii.gz")
                ):
                    all_candidates.append(fpath)

    nifti_paths = [p for p in all_candidates
                   if p.lower().endswith(".nii") or p.lower().endswith(".nii.gz")]

    mag_path: str | None = None
    for priority_patterns in [
        ["overlay"],
        ["t2stack"],
        ["mag"],
    ]:
        for p in nifti_paths:
            bn = os.path.basename(p).lower()
            if any(pat in bn for pat in priority_patterns):
                mag_path = p
                break
        if mag_path:
            break

    if mag_path is None:
        # Fall back: pick first NIfTI that isn't a wave/phase/mask file
        skip_patterns = ["wave_", "_real", "_imag", "phase", "mask"]
        for p in nifti_paths:
            bn = os.path.basename(p).lower()
            if not any(pat in bn for pat in skip_patterns):
                mag_path = p
                break

    if mag_path is None and nifti_paths:
        mag_path = nifti_paths[0]
    if mag_path is None:
        raise FileNotFoundError("No input NIfTI files provided for brain masking")

    forge_log("info", "HD-BET input: %s", os.path.basename(mag_path))

    # Options
    use_tta = forge_param(config, "use_tta", default=False)
    device_str = forge_param(config, "device", default="cpu")

    with tempfile.TemporaryDirectory(prefix="hdbet_") as tmp_dir:
        # Prepare input: ensure .nii.gz + extract single volume if 4-D
        gz_input = _ensure_nii_gz(mag_path, tmp_dir)
        gz_input = _select_magnitude_volume(gz_input, tmp_dir)

        # HD-BET output paths (in tmp first, then copy to output_dir)
        hdbet_output = os.path.join(tmp_dir, "brain.nii.gz")

        forge_log("info", "Running HD-BET (device=%s, tta=%s)...", device_str, use_tta)

        # Import HD-BET (heavy import, deferred to avoid slow startup)
        import torch
        from HD_BET.checkpoint_download import maybe_download_parameters
        from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict

        maybe_download_parameters()

        device = torch.device(device_str)
        predictor = get_hdbet_predictor(
            use_tta=bool(use_tta),
            device=device,
            verbose=False,
        )

        hdbet_predict(
            input_file_or_folder=gz_input,
            output_file_or_folder=hdbet_output,
            predictor=predictor,
            keep_brain_mask=True,
            compute_brain_extracted_image=False,
        )

        # HD-BET writes mask as {output}_bet.nii.gz
        mask_gz = hdbet_output.replace(".nii.gz", "_bet.nii.gz")
        if not os.path.isfile(mask_gz):
            # Fallback: check for the file without _bet suffix
            mask_gz = hdbet_output
        if not os.path.isfile(mask_gz):
            raise FileNotFoundError(f"HD-BET did not produce expected mask file in {tmp_dir}")

        # Copy mask to output directory as uncompressed .nii (consistent with pipeline)
        os.makedirs(output_dir, exist_ok=True)
        basename = Path(mag_path).name.replace(".nii.gz", "").replace(".nii", "")
        final_mask_path = os.path.join(output_dir, f"{basename}_brain_mask.nii")

        # Decompress .nii.gz → .nii
        mask_img = nib.load(mask_gz)
        nib.save(mask_img, final_mask_path)

        forge_log("info", "Brain mask saved: %s", os.path.basename(final_mask_path))

    return [final_mask_path]
