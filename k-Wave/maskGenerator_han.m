function [rho, v, alpha_coeff] = maskGenerator_han(hounsfieldunit)
    % maskGenerator_han
    %
    % 输入：
    %   hounsfieldunit : 3D CT Hounsfield unit 数组 [depth, nRows, nCols]
    %
    % 输出：
    %   rho           : 密度分布 (kg/m^3)
    %   v             : 声速分布 (m/s)
    %   alpha_coeff   : k-Wave 用的 alpha0 (dB/(MHz^2·cm))
    %
    % 说明：
    %   - 先用 SkullLayerSegmentation 基于 HU 分割出：
    %       mask == 0 : 非头骨（水/背景）
    %       mask == 1 : outer cortical
    %       mask == 2 : trabecular
    %       mask == 3 : inner cortical
    %   - 仅对 mask~=0 的体素通过 hounsfield2density 计算 rho，再用
    %       v = 0.70 * rho + 1730 (Li et al. 2025)
    %   - 对 mask==0 的体素直接赋值为水：
    %       rho = 1000 kg/m^3, v = 1500 m/s
    %   - alpha_coeff 只在骨内非零，水中为 0

    %--------------------------------------------------------------
    % 0. 获取尺寸 & 初始化
    %--------------------------------------------------------------
    [depth, nRows, nCols] = size(hounsfieldunit);

    rho = zeros(size(hounsfieldunit));
    v   = zeros(size(hounsfieldunit));

    %--------------------------------------------------------------
    % 1. 基于 HU 分层，得到 0/1/2/3 的 mask
    %--------------------------------------------------------------
    mask = SkullLayerSegmentation(hounsfieldunit);

    %--------------------------------------------------------------
    % 2. 对水/背景 (mask == 0) 直接赋值
    %--------------------------------------------------------------
    water_idx = (mask == 0);
    rho(water_idx) = 1000;   % 水密度
    v(water_idx)   = 1500;   % 水声速

    %--------------------------------------------------------------
    % 3. 只对头骨区域 (mask ~= 0) 计算 rho 和 v
    %--------------------------------------------------------------
    skull_idx = (mask ~= 0);

    if any(skull_idx(:))
        % 取出头骨体素对应的 HU
        hu_skull = hounsfieldunit(skull_idx);

        % HU -> 密度 (k-Wave 的 hounsfield2density 支持向量输入)
        rho_skull = hounsfield2density(hu_skull);

        % 密度 -> 声速 (Li et al. 2025)
        v_skull = 0.70 .* rho_skull + 1730;

        % 写回对应位置
        rho(skull_idx) = rho_skull;
        v(skull_idx)   = v_skull;
    end

    %--------------------------------------------------------------
    % 4. 计算 alpha_coeff（只在骨内非零）
    %    使用 SDR 关系:
    %       alpha0_cort = 4
    %       alpha0_trab = 4 + 17*(1 - SDR)
    %    单位即为 dB/(MHz^2·cm)，可直接给到 k-Wave 的 alpha_coeff
    %--------------------------------------------------------------
    alpha_coeff = zeros(size(hounsfieldunit));
    % 对整幅图逐列处理
    for p1 = 1:nRows
        for p2 = 1:nCols

            m_col  = mask(:, p1, p2);
            hu_col = hounsfieldunit(:, p1, p2)-1000;

            % 骨内索引
            cort_idx = find(m_col == 1 | m_col == 3);   % outer + inner cortical
            trab_idx = find(m_col == 2);                % trabecular

            % 若这一列没有骨，跳过
            if isempty(cort_idx) && isempty(trab_idx)
                continue;
            end

            % cortical / trabecular 的 HU 均值
            if ~isempty(cort_idx)
                hu_cort = mean(hu_col(cort_idx));
            else
                hu_cort = mean(hu_col(trab_idx));  % 兜底
            end

            if ~isempty(trab_idx)
                hu_trab = mean(hu_col(trab_idx));
            else
                hu_trab = hu_cort;                 % 无 trab 时 SDR=1
            end

            % SDR = HU_trab / HU_cort，限制在 [0,1]
            if hu_cort ~= 0
                SDR = hu_trab / hu_cort;
            else
                SDR = 1;
            end
            SDR = max(0, min(1, SDR));

            % alpha0（k-Wave 的 alpha_coeff），单位 dB/(MHz^2·cm)
            alpha0_cort = 4;
            alpha0_trab = 4 + 17 * (1 - SDR);

            % 写入 alpha_coeff：水区保持 0，不用管
            if ~isempty(cort_idx)
                alpha_coeff(cort_idx, p1, p2) = alpha0_cort;
            end
            if ~isempty(trab_idx)
                alpha_coeff(trab_idx, p1, p2) = alpha0_trab;
            end
        end
    end

%     --------------------------------------------------------------
%     5.  padding 到仿真域，
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
