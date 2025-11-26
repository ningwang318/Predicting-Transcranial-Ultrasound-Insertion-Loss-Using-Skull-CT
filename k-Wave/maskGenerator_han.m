function [rho, v, alpha_coeff] = maskGenerator_han(hounsfieldunit)
    % maskGenerator_han
    %
    % Input:
    %   hounsfieldunit : 3D CT Hounsfield unit array [depth, nRows, nCols]
    %
    % Output:
    %   rho           : density distribution (kg/m^3)
    %   v             : speed of sound distribution (m/s)
    %   alpha_coeff   : alpha0 for k-Wave (dB/(MHz^2·cm))
    %
    % Description:
    %   - First, SkullLayerSegmentation is used to segment layers based on HU:
    %       mask == 0 : non-skull (water/background)
    %       mask == 1 : outer cortical
    %       mask == 2 : trabecular
    %       mask == 3 : inner cortical
    %   - Only voxels with mask ~= 0 are converted using hounsfield2density to obtain rho, then
    %       v = 0.70 * rho + 1730 (Li et al. 2025)
    %   - Voxels with mask == 0 are directly assigned as water:
    %       rho = 1000 kg/m^3, v = 1500 m/s
    %   - alpha_coeff is nonzero only inside bone and set to zero in water

    %--------------------------------------------------------------
    % 0. Get dimension & initialize outputs
    %--------------------------------------------------------------
    [depth, nRows, nCols] = size(hounsfieldunit);

    rho = zeros(size(hounsfieldunit));
    v   = zeros(size(hounsfieldunit));

    %--------------------------------------------------------------
    % 1. Segment skull layers based on HU to obtain a mask of 0/1/2/3
    %--------------------------------------------------------------
    mask = SkullLayerSegmentation(hounsfieldunit);

    %--------------------------------------------------------------
    % 2. Directly assign values for water/background (mask == 0)
    %--------------------------------------------------------------
    water_idx = (mask == 0);
    rho(water_idx) = 1000;   % water density
    v(water_idx)   = 1500;   % water speed of sound

    %--------------------------------------------------------------
    % 3. Compute rho and v only for skull voxels (mask ~= 0)
    %--------------------------------------------------------------
    skull_idx = (mask ~= 0);

    if any(skull_idx(:))
        % HU values for skull voxels
        hu_skull = hounsfieldunit(skull_idx);

        % HU → density (k-Wave hounsfield2density supports vector input)
        rho_skull = hounsfield2density(hu_skull);

        % density → speed of sound (Li et al. 2025)
        v_skull = 0.70 .* rho_skull + 1730;

        % write back
        rho(skull_idx) = rho_skull;
        v(skull_idx)   = v_skull;
    end

    %--------------------------------------------------------------
    % 4. Compute alpha_coeff (nonzero only inside bone)
    %    Using the SDR relation:
    %       alpha0_cort = 4
    %       alpha0_trab = 4 + 17*(1 - SDR)
    %    Unit: dB/(MHz^2·cm), directly compatible with k-Wave alpha_coeff
    %--------------------------------------------------------------
    alpha_coeff = zeros(size(hounsfieldunit));

    % Process column by column
    for p1 = 1:nRows
        for p2 = 1:nCols

            m_col  = mask(:, p1, p2);
            hu_col = hounsfieldunit(:, p1, p2)-1000;

            % Bone voxel indices
            cort_idx = find(m_col == 1 | m_col == 3);   % outer + inner cortical
            trab_idx = find(m_col == 2);                % trabecular

            % If no bone appears in this column, skip
            if isempty(cort_idx) && isempty(trab_idx)
                continue;
            end

            % Mean HU of cortical / trabecular regions
            if ~isempty(cort_idx)
                hu_cort = mean(hu_col(cort_idx));
            else
                hu_cort = mean(hu_col(trab_idx));   % fallback
            end

            if ~isempty(trab_idx)
                hu_trab = mean(hu_col(trab_idx));
            else
                hu_trab = hu_cort;                  % no trabecular → SDR = 1
            end

            % SDR = HU_trab / HU_cort, clamp to [0, 1]
            if hu_cort ~= 0
                SDR = hu_trab / hu_cort;
            else
                SDR = 1;
            end
            SDR = max(0, min(1, SDR));

            % alpha0 values (k-Wave alpha_coeff)
            alpha0_cort = 4;
            alpha0_trab = 4 + 17 * (1 - SDR);

            % write into alpha_coeff (water remains 0)
            if ~isempty(cort_idx)
                alpha_coeff(cort_idx, p1, p2) = alpha0_cort;
            end
            if ~isempty(trab_idx)
                alpha_coeff(trab_idx, p1, p2) = alpha0_trab;
            end
        end
    end

%     --------------------------------------------------------------
%     5. Padding to match the simulation grid
%     --------------------------------------------------------------

    offset   = 5;
    padFront = 20 + offset;
    totalPad = 272 + offset + 2 - size(v,1);
    padBack  = totalPad - padFront;
    
    rho = padarray(rho, [padFront, 0, 0], 1000, 'pre');
    rho = padarray(rho, [padBack,  0, 0], 1000, 'post');
    
    v   = padarray(v,   [padFront, 0, 0], 1500, 'pre');
    v   = padarray(v,   [padBack,  0, 0], 1500, 'post');
    
    alpha_coeff = padarray(alpha_coeff, [padFront, 0, 0], 0, 'pre');
    alpha_coeff = padarray(alpha_coeff, [padBack,  0, 0], 0, 'post');

end
