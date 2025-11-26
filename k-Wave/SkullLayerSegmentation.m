function mask = SkullLayerSegmentation(hu_vol)
% SkullLayerSegmentation
% Input:
%   hu_vol : 3D HU volume, size = [depth, nRows, nCols]
%
% Output:
%   mask   : 3D mask with the same size
%            0 = background / no bone
%            1 = outer cortical
%            2 = trabecular
%            3 = inner cortical
    
    [depth, nRows, nCols] = size(hu_vol);
    mask = zeros(size(hu_vol), 'like', hu_vol);  % double or int both acceptable

    for p1 = 1:nRows
        for p2 = 1:nCols

            profile = hu_vol(:, p1, p2);
            nz = find(profile ~= 1000);

            %----------------------------------------------------------
            % 1. Handle the case where the entire column is 1000: borrow a neighboring column
            %----------------------------------------------------------
            if isempty(nz)
                % If the current column is all 1000, copy the HU line from a neighboring column
                if p1 == 1
                    srcCol = min(p1 + 10, nRows);  % original code used p1+10, here we add protection
                else
                    srcCol = p1 - 1;
                end

                profile = hu_vol(:, srcCol, p2);
                nz = find(profile ~= 1000);

                if isempty(nz)
                    % Neighboring column is also all 1000 → skip
                    continue;
                end
            end

            firstIdx = nz(1);
            lastIdx  = nz(end);
            line = profile(firstIdx:lastIdx);
            L = numel(line);

            % If only a few scattered bone points exist, later logic will handle them
            outer_idx_rel = [];
            trab_idx_rel  = [];
            inner_idx_rel = [];

            %----------------------------------------------------------
            % 2. Three-layer segmentation logic
            %----------------------------------------------------------
            if L > 8
                % ---- Case 1: Thick skull → segmentation using "min HU + 0.8·max threshold"
                if L > 4  % Ensure 3:end-2 is valid
                    midMin = min(line(3:end-2));
                else
                    midMin = min(line);  % fallback for extreme cases
                end
                midIdxRel = find(line ./ midMin == 1, 1, 'first');
                if isempty(midIdxRel)
                    midIdxRel = round(L/2);
                end

                cort1 = line(1:midIdxRel);
                cort2 = line(min(midIdxRel+1, L):end);  % avoid overflow

                % Outer cortical: trailing segment of cort1 where HU > 0.8 * max(cort1)
                if ~isempty(cort1)
                    c1EndRel = find(cort1 > max(cort1) * 0.8, 1, 'last');
                    if isempty(c1EndRel)
                        c1EndRel = numel(cort1);
                    end
                    outer_idx_rel = 1:c1EndRel;
                end

                % Inner cortical: leading segment of cort2 where HU > 0.8 * max(cort2)
                if ~isempty(cort2)
                    c2StartInC2 = find(cort2 > max(cort2) * 0.8, 1, 'first');
                    if isempty(c2StartInC2)
                        c2StartInC2 = 1;
                    end
                    c2StartRel  = midIdxRel - 1 + c2StartInC2;  % index relative to "line"
                    inner_idx_rel = c2StartRel:L;
                end

                lenOuter = numel(outer_idx_rel);
                lenInner = numel(inner_idx_rel);

                if lenOuter + lenInner < L
                    trab_idx_rel = (lenOuter + 1) : (L - lenInner);
                else
                    % Similar to original logic: ensure at least 1 voxel for trabecular region
                    if lenOuter > 4
                        trab_idx_rel = outer_idx_rel(end);
                        outer_idx_rel = outer_idx_rel(1:end-1);
                    elseif lenInner > 0
                        trab_idx_rel = inner_idx_rel(1);
                        inner_idx_rel = inner_idx_rel(2:end);
                    else
                        % If still impossible, treat everything as trabecular
                        trab_idx_rel = 1:L;
                        outer_idx_rel = [];
                        inner_idx_rel = [];
                    end
                end

            elseif L >= 5
                % ---- Case 2: Moderate skull thickness → first 2 & last 2 as cortical
                outer_idx_rel = 1:2;
                inner_idx_rel = (L-1):L;
                if L > 4
                    trab_idx_rel  = 3:(L-2);
                else
                    trab_idx_rel  = [];
                end

            else
                % ---- Case 3: Very thin skull (L < 5)
                % Simplified handling: classify the whole region as trabecular
                trab_idx_rel = 1:L;
                outer_idx_rel = [];
                inner_idx_rel = [];
            end

            %----------------------------------------------------------
            % 3. Write segmentation results back to the 3-D mask
            %----------------------------------------------------------
            if ~isempty(outer_idx_rel)
                outer_idx_abs = firstIdx - 1 + outer_idx_rel;
                mask(outer_idx_abs, p1, p2) = 1;
            end
            if ~isempty(trab_idx_rel)
                trab_idx_abs = firstIdx - 1 + trab_idx_rel;
                mask(trab_idx_abs, p1, p2) = 2;
            end
            if ~isempty(inner_idx_rel)
                inner_idx_abs = firstIdx - 1 + inner_idx_rel;
                mask(inner_idx_abs, p1, p2) = 3;
            end

        end
    end
end
