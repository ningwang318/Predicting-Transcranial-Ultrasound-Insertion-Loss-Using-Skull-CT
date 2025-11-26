function mask = SkullLayerSegmentation(hu_vol)
% SkullLayerSegmentation
% 输入:
%   hu_vol : 3D HU 体数据, size = [depth, nRows, nCols]
%
% 输出:
%   mask   : 同尺寸 3D 掩膜
%            0 = 背景/无骨
%            1 = outer cortical
%            2 = trabecular
%            3 = inner cortical

    [depth, nRows, nCols] = size(hu_vol);
    mask = zeros(size(hu_vol), 'like', hu_vol);  % 用 double/int 都可

    for p1 = 1:nRows
        for p2 = 1:nCols

            profile = hu_vol(:, p1, p2);
            nz = find(profile ~= 1000);

            %----------------------------------------------------------
            % 1. 处理"全 1000"的情况：用邻居来填
            %----------------------------------------------------------
            if isempty(nz)
                % 若当前列全 1000，复制邻居列的 HU 线来做分割
                if p1 == 1
                    srcCol = min(p1 + 10, nRows);  % 原代码是 p1+10，这里做一下保护
                else
                    srcCol = p1 - 1;
                end

                profile = hu_vol(:, srcCol, p2);
                nz = find(profile ~= 1000);

                if isempty(nz)
                    % 邻居也全 1000，就跳过
                    continue;
                end
            end

            firstIdx = nz(1);
            lastIdx  = nz(end);
            line = profile(firstIdx:lastIdx);
            L = numel(line);

            % 如果只有零星几点骨，后面会用简单规则
            outer_idx_rel = [];
            trab_idx_rel  = [];
            inner_idx_rel = [];

            %----------------------------------------------------------
            % 2. 三层分割逻辑
            %----------------------------------------------------------
            if L > 8
                % ---- 情况一：较厚骨层，使用“最小 HU + 0.8·max 阈值”
                if L > 4  % 保证 3:end-2 有意义
                    midMin = min(line(3:end-2));
                else
                    midMin = min(line);  % 极端情况兜底
                end
                midIdxRel = find(line ./ midMin == 1, 1, 'first');
                if isempty(midIdxRel)
                    midIdxRel = round(L/2);
                end

                cort1 = line(1:midIdxRel);
                cort2 = line(min(midIdxRel+1, L):end);  % 防止越界

                % 外板：> 0.8 * max(cort1) 的后一段
                if ~isempty(cort1)
                    c1EndRel = find(cort1 > max(cort1) * 0.8, 1, 'last');
                    if isempty(c1EndRel)
                        c1EndRel = numel(cort1);
                    end
                    outer_idx_rel = 1:c1EndRel;
                end

                % 内板：> 0.8 * max(cort2) 的前一段
                if ~isempty(cort2)
                    c2StartInC2 = find(cort2 > max(cort2) * 0.8, 1, 'first');
                    if isempty(c2StartInC2)
                        c2StartInC2 = 1;
                    end
                    c2StartRel  = midIdxRel - 1 + c2StartInC2;  % 相对 line 的下标
                    inner_idx_rel = c2StartRel:L;
                end

                lenOuter = numel(outer_idx_rel);
                lenInner = numel(inner_idx_rel);

                if lenOuter + lenInner < L
                    trab_idx_rel = (lenOuter + 1) : (L - lenInner);
                else
                    % 和原逻辑类似，强行给 trab 留一个像素
                    if lenOuter > 4
                        trab_idx_rel = outer_idx_rel(end);
                        outer_idx_rel = outer_idx_rel(1:end-1);
                    elseif lenInner > 0
                        trab_idx_rel = inner_idx_rel(1);
                        inner_idx_rel = inner_idx_rel(2:end);
                    else
                        % 真没办法，就全当 trab
                        trab_idx_rel = 1:L;
                        outer_idx_rel = [];
                        inner_idx_rel = [];
                    end
                end

            elseif L >= 5
                % ---- 情况二：中等厚度，固定前 2 / 后 2 为 cortical
                outer_idx_rel = 1:2;
                inner_idx_rel = (L-1):L;
                if L > 4
                    trab_idx_rel  = 3:(L-2);
                else
                    trab_idx_rel  = [];
                end

            else
                % ---- 情况三：非常薄的骨层（L < 5）
                % 这里简化处理：全部视为 trabecular
                trab_idx_rel = 1:L;
                outer_idx_rel = [];
                inner_idx_rel = [];
            end

            %----------------------------------------------------------
            % 3. 写入 mask（转换回绝对 depth index）
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
