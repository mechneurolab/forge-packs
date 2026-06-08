function output_paths = prepEmpire(config_path)
%PREPEMPIRE  Prepare EMPIRE 3D MRE data for reconstruction.
%
%  output_paths = prepEmpire(CONFIG_PATH)
%
%  Reads a Siemens TWIX .dat file from a 3D EMPIRE MRE acquisition and
%  produces a prepared ISMRMRD .h5 file suitable for forgeSense.
%
%  Pipeline steps:
%    1. Parse TWIX file and extract k-space, trajectory, and headers
%    2. Extract prescan calibration data for coil sensitivity estimation
%    3. Compute coil sensitivity maps (ESPIRiT or prescan-based)
%    4. Compute trajectory and density compensation weights
%    5. Write prepared ISMRMRD .h5 with all required fields
%
%  Config fields (from _step_config_prep_empire.json):
%    config.input_paths{1}  - Path to TWIX .dat file
%    config.output_dir      - Directory for output files
%    config.params.output_format - 'ismrmrd' (only supported value)
%
%  FORGE Studio MATLAB module interface.
%  See: specs/Phase7/forge-matlab-backend-spec.md §5

  config     = forge_read_config(config_path);
  output_dir = config.output_dir;
  twix_path  = config.input_paths{1};

  % Scan dimensions from header (populated by FORGE Studio from TWIX ASCCONV)
  nSlices      = field_or_default(config, 'slices', 1);
  nRepetitions = field_or_default(config, 'repetitions', 1);
  nPhases      = field_or_default(config, 'phases', 1);
  nEchoes      = field_or_default(config, 'echoes', 1);
  nAverages    = field_or_default(config, 'averages', 1);
  nChannels    = field_or_default(config, 'channels', 0);

  forge_log('info', 'prepEmpire: starting preparation for %s', twix_path);
  forge_log('info', 'prepEmpire: dimensions — slices=%d reps=%d phases=%d echoes=%d avg=%d channels=%d', ...
    nSlices, nRepetitions, nPhases, nEchoes, nAverages, nChannels);

  % ── Step 1: Parse TWIX ──────────────────────────────────────────────────
  forge_log('info', 'prepEmpire: parsing TWIX file');
  forge_progress('parse', 1, 5, 'Preparation', 'Parsing TWIX');

  % parse data
  rInfo = recoInfo(twix_path);

  % ── Step 2: Extract prescan calibration ─────────────────────────────────
  forge_log('info', 'prepEmpire: extracting prescan calibration data');
  forge_progress('parse', 2, 5, 'Preparation', 'Extracting calibration');

  cImagesCal = gridCoilImagesCal(rInfo, 'echoesToRecon', 1, 'phasesToRecon', 1);
  imSOSCal = sqrt(sum(abs(cImagesCal).^2, 10));

  % ── Step 3: Compute coil sensitivity maps ───────────────────────────────
  forge_log('info', 'prepEmpire: computing coil sensitivity maps');
  forge_progress('parse', 3, 5, 'Preparation', 'Computing SENSE maps');

  level = multithresh(abs(imSOSCal));

  mask = abs(imSOSCal) > level;

  %mask = mask .* mask_circ;
  mask = (mask > 0);

  mask = imclose(mask, strel('cube',round(rInfo.N/5)));
  mask = imdilate(mask, strel('sphere',2));

  mask = reshape(mask, rInfo.N, rInfo.N, rInfo.Nz, rInfo.nSlices);

  sen = combineCoilsInati3D(squeeze(cImagesCal), 10, 9);
  sen = reshape(sen, rInfo.N,rInfo.N,rInfo.Nz,rInfo.nSlices,rInfo.nCoils);

  % ── Step 4: Compute overlay image ──────────────────────────────────
  forge_log('info', 'prepEmpire: computing overlay image');
  forge_progress('parse', 4, 5, 'Preparation', 'Computing trajectory');

  FMzeros = zeros(rInfo.N,rInfo.N,rInfo.Nz);
  imgOverlay = fieldCorrectedReconCal(rInfo, sen, mask, FMzeros, 'echoesToRecon', 1, 'phasesToRecon',1);

  % ── Step 5: Write prepared ISMRMRD and overlay image ────────────────────
  forge_log('info', 'prepEmpire: writing prepared ISMRMRD file');
  forge_progress('parse', 5, 5, 'Preparation', 'Writing ISMRMRD');

  % Construct output filename from input basename
  [~, basename, ~] = fileparts(twix_path);
  output_h5 = fullfile(output_dir, sprintf('%s_prepared.h5', basename));

  convertRecoInfoToIsmrmrd(sprintf('%s',output_h5),rInfo,permute(sen,[1 2 3 5 4]), FMzeros);

  % Save overlay image (magnitude) as NIfTI with orientation header
  overlay_path = fullfile(output_dir, sprintf('%s_overlay.nii', basename));
  overlay_mag = single(abs(squeeze(imgOverlay)));

  % Build NIfTI sform affine from rInfo orientation data.
  % rInfo stores direction cosines in LPS (DICOM/ISMRMRD convention).
  % NIfTI uses RAS, so flip the first two axes (L->R, P->A).
  lps2ras = diag([-1, -1, 1]);
  voxSize = [rInfo.FOV/rInfo.N, rInfo.FOV/rInfo.N, rInfo.sliceThickness];
  R = lps2ras * [rInfo.readDir(:), rInfo.phaseDir(:), rInfo.sliceDir(:)];
  T = lps2ras * rInfo.slicePosition(:, 1);
  affine = eye(4);
  affine(1:3, 1:3) = R .* voxSize;
  affine(1:3, 4) = T;

  nii = make_nii(overlay_mag, voxSize, [0 0 0], 16, 'EMPIRE overlay');
  nii.hdr.hist.sform_code = 1;  % Scanner anatomical coordinates
  nii.hdr.hist.srow_x = affine(1, :);
  nii.hdr.hist.srow_y = affine(2, :);
  nii.hdr.hist.srow_z = affine(3, :);
  save_nii(nii, overlay_path);
  forge_log('info', 'prepEmpire: saved overlay image → %s', overlay_path);

  forge_progress_done('parse');
  forge_log('info', 'prepEmpire: preparation complete → %s', output_h5);
  output_paths = {output_h5, overlay_path};
end

function val = field_or_default(s, name, default_val)
%FIELD_OR_DEFAULT  Return struct field value, or default if absent.
  if isfield(s, name)
    val = s.(name);
  else
    val = default_val;
  end
end
