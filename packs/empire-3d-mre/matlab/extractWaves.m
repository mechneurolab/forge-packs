function output_paths = extractWaves(config_path)
%EXTRACTWAVES  Extract harmonic wave displacements from reconstructed MRE images.
%
%  output_paths = extractWaves(CONFIG_PATH)
%
%  Takes the reconstructed MRE images from forgeSense (per-rep mag+phase
%  NIfTI pairs) and extracts harmonic wave displacement fields via echo
%  ratio, Laplacian phase unwrapping, and temporal Fourier analysis.
%
%  Pipeline steps:
%    1. Scan recon output directory for mag/phase NIfTI pairs
%    2. Load and combine into complex volumes
%    3. Reshape reps into [echoes, MEG directions, phase offsets]
%    4. Compute echo ratio per MEG direction (phase difference encoding)
%    5. Laplacian phase unwrapping (no external dependencies)
%    6. FFT along phase offset dimension, extract 1st harmonic
%    7. Write X/Y/Z wave displacement NIfTIs
%
%  Config fields (from _step_config_wave_extract.json):
%    config.input_paths{1}                  - Path to recon output directory
%    config.output_dir                      - Directory for output files
%    config.params.frequency_hz             - Vibration frequency in Hz
%    config.params.num_harmonics            - Number of harmonics to extract
%    config.params.displacement_component   - 'all','magnitude','x','y','z'
%    config.params.n_echoes                 - Number of echoes (default: 2)
%    config.params.n_meg_directions         - Number of MEG directions (default: 3)
%    config.params.phase_to_disp_scale      - Phase-to-displacement scale (default: 1.764)
%    config.params.meg_order                - MEG dir ordering [Z,Y,X] (default: [3,1,2])
%
%  FORGE Studio MATLAB module interface.
%  See: specs/Phase7/forge-matlab-backend-spec.md §5

  config     = forge_read_config(config_path);
  output_dir = config.output_dir;
  recon_path = config.input_paths{1};

  % The pipeline chains individual file paths from the previous stage.
  % If recon_path points to a file rather than a directory, use its parent.
  if isfile(recon_path)
    recon_path = fileparts(recon_path);
  end

  % Parameters
  freq_hz        = forge_param(config, 'frequency_hz', 60.0);
  num_harmonics  = forge_param(config, 'num_harmonics', 1);
  disp_comp      = forge_param(config, 'displacement_component', 'all');
  n_echoes       = forge_param(config, 'n_echoes', 2);
  n_meg          = forge_param(config, 'n_meg_directions', 3);
  disp_scale     = forge_param(config, 'phase_to_disp_scale', 1.764);
  meg_order      = forge_param(config, 'meg_order', [3 1 2]); % [Z Y X] MEG indices

  forge_log('info', 'extractWaves: recon_dir=%s freq=%.1f Hz harmonics=%d', ...
    recon_path, freq_hz, num_harmonics);
  forge_log('info', 'extractWaves: n_echoes=%d n_meg=%d scale=%.4f', ...
    n_echoes, n_meg, disp_scale);

  % ── Step 1: Scan and load reconstructed images ─────────────────────────
  forge_log('info', 'extractWaves: scanning recon directory for NIfTI pairs');
  forge_progress('waves', 1, 7, 'Wave Extraction', 'Loading images');

  % Discover slices and reps from the file listing
  mag_files = dir(fullfile(recon_path, 'img_Slice*_Rep*_*_mag.nii'));
  assert(~isempty(mag_files), ...
    'extractWaves:noFiles', 'No magnitude NIfTIs found in %s', recon_path);

  % Parse slice and rep indices from filenames
  slice_indices = [];
  rep_indices   = [];
  for k = 1:numel(mag_files)
    tokens = regexp(mag_files(k).name, 'img_Slice(\d+)_Rep(\d+)_', 'tokens');
    if ~isempty(tokens)
      slice_indices(end+1) = str2double(tokens{1}{1}); %#ok<AGROW>
      rep_indices(end+1)   = str2double(tokens{1}{2}); %#ok<AGROW>
    end
  end

  n_slices = numel(unique(slice_indices));
  n_reps   = numel(unique(rep_indices));
  n_phase_offsets = n_reps / (n_echoes * n_meg);

  forge_log('info', 'extractWaves: found %d slice(s), %d rep(s) → %d echoes x %d MEG x %d phases', ...
    n_slices, n_reps, n_echoes, n_meg, n_phase_offsets);

  assert(mod(n_reps, n_echoes * n_meg) == 0, ...
    'extractWaves:repMismatch', ...
    'n_reps (%d) is not divisible by n_echoes*n_meg (%d*%d=%d)', ...
    n_reps, n_echoes, n_meg, n_echoes * n_meg);

  % Load first file to get volume dimensions and voxel sizes
  first_mag = fullfile(mag_files(1).folder, mag_files(1).name);
  tmp_nii = load_untouch_nii(first_mag);
  vol_size = size(tmp_nii.img);
  nx = vol_size(1); ny = vol_size(2); nz = vol_size(3);

  % Voxel sizes from NIfTI header (pixdim: [qfac dx dy dz ...])
  dx = double(tmp_nii.hdr.dime.pixdim(2));
  dy = double(tmp_nii.hdr.dime.pixdim(3));
  dz = double(tmp_nii.hdr.dime.pixdim(4));

  forge_log('info', 'extractWaves: volume size = [%d %d %d], voxel = [%.2f %.2f %.2f] mm', ...
    nx, ny, nz, dx, dy, dz);

  % Load all mag+phase pairs into complex array
  % img(x, y, z, slice, rep) — complex-valued
  img = zeros(nx, ny, nz, n_slices, n_reps, 'single');

  unique_slices = sort(unique(slice_indices));
  unique_reps   = sort(unique(rep_indices));

  total_files = n_slices * n_reps;
  file_count = 0;
  for si = 1:n_slices
    s = unique_slices(si);
    for ri = 1:n_reps
      r = unique_reps(ri);

      mag_name = sprintf('img_Slice%d_Rep%d_Avg0_Echo0_Phase0_mag.nii', s, r);
      phs_name = sprintf('img_Slice%d_Rep%d_Avg0_Echo0_Phase0_phs.nii', s, r);

      mag_path = fullfile(recon_path, mag_name);
      phs_path = fullfile(recon_path, phs_name);

      assert(isfile(mag_path), 'extractWaves:missingFile', 'Missing: %s', mag_path);
      assert(isfile(phs_path), 'extractWaves:missingFile', 'Missing: %s', phs_path);

      mag_nii = load_untouch_nii(mag_path);
      phs_nii = load_untouch_nii(phs_path);
      img(:,:,:,si,ri) = single(mag_nii.img) .* exp(1i * single(phs_nii.img));

      file_count = file_count + 1;
      if mod(file_count, 10) == 0
        forge_progress('waves', 1, 7, 'Wave Extraction', ...
          sprintf('Loading %d/%d', file_count, total_files));
      end
    end
  end
  forge_log('info', 'extractWaves: loaded %d file pairs', total_files);

  % ── Step 2: Reshape into echo/MEG/phase structure ──────────────────────
  forge_log('info', 'extractWaves: reshaping into echo/MEG/phase structure');
  forge_progress('waves', 2, 7, 'Wave Extraction', 'Reshaping data');

  % Squeeze out singleton slice dimension if single-slab 3D
  img = squeeze(img); % → (nx, ny, nz, n_reps) for single slice

  % Reshape: reps → (n_echoes, n_meg, n_phase_offsets)
  imgraw = reshape(img, [nx, ny, nz, n_echoes, n_meg, n_phase_offsets]);

  % ── Step 3: Compute echo ratio per MEG direction ───────────────────────
  forge_log('info', 'extractWaves: computing echo ratios');
  forge_progress('waves', 3, 7, 'Wave Extraction', 'Echo ratios');

  % Echo ratio: echo1 ./ echo2 → phase difference encodes motion
  % meg_order maps MEG acquisition index → [Z, Y, X] output order
  echo_ratio = zeros(nx, ny, nz, 3, n_phase_offsets, 'single');
  for di = 1:3
    meg_idx = meg_order(di);
    echo_ratio(:,:,:,di,:) = squeeze( ...
      imgraw(:,:,:,1,meg_idx,:) ./ imgraw(:,:,:,2,meg_idx,:));
  end
  % di=1→Z, di=2→Y, di=3→X

  % Replace NaN/Inf from division (zero signal regions)
  echo_ratio(~isfinite(echo_ratio)) = 0;

  % ── Step 4: Phase unwrapping (Laplacian method) ────────────────────────
  forge_log('info', 'extractWaves: Laplacian phase unwrapping');
  forge_progress('waves', 4, 7, 'Wave Extraction', 'Phase unwrapping');

  % Create brain mask from magnitude for masking
  mag_avg = mean(abs(img), ndims(img)); % average magnitude across reps
  mask_thresh = multithresh(abs(mag_avg));
  mask = abs(mag_avg) > mask_thresh;

  phase_unwrapped = zeros(nx, ny, nz, 3, n_phase_offsets, 'single');
  total_unwraps = 3 * n_phase_offsets;
  unwrap_count = 0;
  for di = 1:3
    for pi_idx = 1:n_phase_offsets
      phi_wrapped = angle(echo_ratio(:,:,:,di,pi_idx));
      phi_wrapped = phi_wrapped .* single(mask); % mask before unwrapping
      phase_unwrapped(:,:,:,di,pi_idx) = single(laplacian_unwrap_3d(double(phi_wrapped)));

      unwrap_count = unwrap_count + 1;
      if mod(unwrap_count, 3) == 0
        forge_progress('waves', 4, 7, 'Wave Extraction', ...
          sprintf('Unwrapping %d/%d', unwrap_count, total_unwraps));
      end
    end
  end

  % Apply mask to unwrapped phase
  phase_unwrapped = phase_unwrapped .* single(mask);

  % ── Step 5: FFT along phase offsets → harmonic extraction ──────────────
  forge_log('info', 'extractWaves: extracting harmonics via FFT');
  forge_progress('waves', 5, 7, 'Wave Extraction', 'Harmonic extraction');

  % Scale phase to displacement and flip phase offset dimension
  disp_img = flip(phase_unwrapped, 5) * disp_scale;

  % FFT along phase offset dimension (dim 5), normalize
  fft_img = fft(disp_img, [], 5) / (n_phase_offsets / 2);

  % Extract harmonics: bin 2 = 1st harmonic, bin 3 = 2nd, etc.
  wave_fields = fft_img(:,:,:,:,2:(1 + num_harmonics));

  % For single harmonic (most common), squeeze out that dimension
  if num_harmonics == 1
    wave_fields = squeeze(wave_fields); % → (nx, ny, nz, 3)
  end

  % Separate directional components (Z, Y, X)
  Zmotion = wave_fields(:,:,:,1);
  Ymotion = wave_fields(:,:,:,2);
  Xmotion = wave_fields(:,:,:,3);

  forge_log('info', 'extractWaves: wave field range — X:[%.3g, %.3g] Y:[%.3g, %.3g] Z:[%.3g, %.3g]', ...
    min(abs(Xmotion(:))), max(abs(Xmotion(:))), ...
    min(abs(Ymotion(:))), max(abs(Ymotion(:))), ...
    min(abs(Zmotion(:))), max(abs(Zmotion(:))));

  % ── Step 6: OSS-SNR filtering and MRE params ──────────────────────────
  forge_log('info', 'extractWaves: computing OSS-SNR and saving MRE params');
  forge_progress('waves', 6, 7, 'Wave Extraction', 'OSS-SNR filtering');

  % Load reference NIfTI for output header/geometry (used by steps 6 and 7)
  ref_nii = load_untouch_nii(first_mag);

  % Compute OSS-SNR from unwrapped phase images
  OSS_SNR = oss_snr_filter(double(phase_unwrapped), [dx dy dz], freq_hz, double(mask));

  % Compute t2stack (mean magnitude across reps) for anatomical reference
  t2stack = mean(abs(squeeze(img)), ndims(squeeze(img)));

  % Build mreParams struct for downstream inversion
  [~, basename, ~] = fileparts(recon_path);
  mreParams.subj   = basename;
  mreParams.FOVx   = nx * dx;
  mreParams.FOVy   = ny * dy;
  mreParams.FOVz   = nz * dz;
  mreParams.nx     = nx;
  mreParams.ny     = ny;
  mreParams.nz     = nz;
  mreParams.dx     = dx;
  mreParams.dy     = dy;
  mreParams.dz     = dz;
  mreParams.freq   = freq_hz;
  mreParams.oss_snr = OSS_SNR;

  % Save MRE parameters as JSON (backend-agnostic for inversion stage)
  mre_json_path = fullfile(output_dir, sprintf('%s_mre_params.json', basename));
  fid = fopen(mre_json_path, 'w');
  fprintf(fid, '%s', jsonencode(mreParams));
  fclose(fid);
  forge_log('info', 'extractWaves: saved MRE params → %s', mre_json_path);

  % Save mask and t2stack as NIfTIs (readable by any backend)
  mask_path = fullfile(output_dir, sprintf('%s_mask.nii', basename));
  write_scalar_nifti(mask_path, single(mask), ref_nii);
  forge_log('info', 'extractWaves: saved mask → %s', mask_path);

  t2_path = fullfile(output_dir, sprintf('%s_t2stack.nii', basename));
  write_scalar_nifti(t2_path, single(t2stack), ref_nii);
  forge_log('info', 'extractWaves: saved t2stack → %s', t2_path);

  % --- Sentinel-ready handoff (.mat in read_mre_uiuc layout) ---
  sentinel_mat = fullfile(output_dir, sprintf('%s_mre_sentinel.mat', basename));
  save(sentinel_mat, 'Xmotion', 'Ymotion', 'Zmotion', 't2stack', 'mask', 'mreParams', '-v7.3');
  fprintf('[extractWaves] wrote Sentinel handoff: %s\n', sentinel_mat);

  % ── Step 7: Write output NIfTIs ────────────────────────────────────────
  forge_log('info', 'extractWaves: writing output (%s)', disp_comp);
  forge_progress('waves', 7, 7, 'Wave Extraction', 'Writing output');

  output_paths = {};

  switch disp_comp
    case 'all'
      % Write each component as real + imag NIfTI pairs
      names = {'Z', 'Y', 'X'};
      motions = {Zmotion, Ymotion, Xmotion};
      for di = 1:3
        paths = write_complex_nifti(output_dir, basename, names{di}, motions{di}, ref_nii);
        output_paths = [output_paths, paths]; %#ok<AGROW>
        forge_log('info', 'extractWaves: wrote %s real+imag', names{di});
      end

    case 'magnitude'
      wave_mag = sqrt(abs(Xmotion).^2 + abs(Ymotion).^2 + abs(Zmotion).^2);
      out_file = fullfile(output_dir, sprintf('%s_wave_mag.nii', basename));
      write_scalar_nifti(out_file, wave_mag, ref_nii);
      output_paths = {out_file};

    case 'x'
      output_paths = write_complex_nifti(output_dir, basename, 'X', Xmotion, ref_nii);

    case 'y'
      output_paths = write_complex_nifti(output_dir, basename, 'Y', Ymotion, ref_nii);

    case 'z'
      output_paths = write_complex_nifti(output_dir, basename, 'Z', Zmotion, ref_nii);

    otherwise
      error('extractWaves:badComponent', 'Unknown displacement_component: %s', disp_comp);
  end

  % Include auxiliary outputs (MRE params JSON, mask, t2stack, Sentinel .mat)
  output_paths = [output_paths, {mre_json_path, mask_path, t2_path, sentinel_mat}];

  forge_progress_done('waves');
  forge_log('info', 'extractWaves: complete — %d output file(s)', numel(output_paths));
end

% ═══════════════════════════════════════════════════════════════════════════
% Local functions
% ═══════════════════════════════════════════════════════════════════════════

function phi_unwrapped = laplacian_unwrap_3d(phi_wrapped)
%LAPLACIAN_UNWRAP_3D  3D phase unwrapping via Laplacian method.
%
%  Schofield & Zhu, "Fast phase unwrapping algorithm for interferometric
%  applications," Optics Letters 28(14), 2003.
%
%  Computes the true Laplacian of the unwrapped phase from the wrapped
%  phase using the identity:
%    nabla^2(phi) = cos(phi)*nabla^2(sin(phi)) - sin(phi)*nabla^2(cos(phi))
%  then solves the Poisson equation in Fourier space.
%
%  Note: the global phase offset is undefined (DC term set to zero).
%  This is irrelevant for MRE since the subsequent FFT across phase
%  offsets removes any constant offset.

  [n1, n2, n3] = size(phi_wrapped);

  % Discrete Laplacian kernel in Fourier space
  [K1, K2, K3] = ndgrid(0:n1-1, 0:n2-1, 0:n3-1);
  lap_kernel = 2*(cos(2*pi*K1/n1) - 1) + ...
               2*(cos(2*pi*K2/n2) - 1) + ...
               2*(cos(2*pi*K3/n3) - 1);

  cos_phi = cos(phi_wrapped);
  sin_phi = sin(phi_wrapped);

  % Forward Laplacian of sin(phi) and cos(phi)
  lap_sin = real(ifftn(fftn(sin_phi) .* lap_kernel));
  lap_cos = real(ifftn(fftn(cos_phi) .* lap_kernel));

  % True Laplacian of the unwrapped phase
  rho = cos_phi .* lap_sin - sin_phi .* lap_cos;

  % Inverse Laplacian (Poisson solve via FFT)
  inv_lap = 1 ./ lap_kernel;
  inv_lap(1, 1, 1) = 0; % DC undefined — set to zero

  phi_unwrapped = real(ifftn(fftn(rho) .* inv_lap));
end

function paths = write_complex_nifti(output_dir, basename, comp_name, complex_data, ref_nii)
%WRITE_COMPLEX_NIFTI  Write complex wave data as separate real + imag NIfTI files.
  real_file = fullfile(output_dir, sprintf('%s_wave_%s_real.nii', basename, comp_name));
  imag_file = fullfile(output_dir, sprintf('%s_wave_%s_imag.nii', basename, comp_name));
  write_scalar_nifti(real_file, real(complex_data), ref_nii);
  write_scalar_nifti(imag_file, imag(complex_data), ref_nii);
  paths = {real_file, imag_file};
end

function write_scalar_nifti(filepath, data, ref_nii)
%WRITE_SCALAR_NIFTI  Write a real-valued 3D volume as a NIfTI file.
%  Uses the reference NIfTI struct for header/geometry, replaces the image data.
  out_nii = ref_nii;
  out_nii.img = single(data);
  sz = size(single(data));
  out_nii.hdr.dime.dim(1) = numel(sz);
  out_nii.hdr.dime.dim(2:1+numel(sz)) = sz;
  out_nii.hdr.dime.datatype = 16;  % FLOAT32
  out_nii.hdr.dime.bitpix = 32;
  out_nii.hdr.hist.descrip = 'MRE wave displacement';
  save_untouch_nii(out_nii, filepath);
end
