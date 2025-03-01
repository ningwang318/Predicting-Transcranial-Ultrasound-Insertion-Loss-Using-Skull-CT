function result = FeatureExtract(s_normal_3_2)

    [depth, nRows, nCols] = size(s_normal_3_2);
    

    result = zeros(6, nRows, nCols);
    

    for p1 = 1:nRows
        for p2 = 1:nCols

            line_density_1 = s_normal_3_2(:, p1, p2);
            line_density_1 = line_density_1(find(line_density_1 ~= 0, 1, 'first'):find(line_density_1 ~= 0, 1, 'last'), 1);
 
            line_density_all = zeros(depth, nRows, nCols);
            line_density_all(1:length(line_density_1), p1, p2) = line_density_1;
            line_density = line_density_all(1:length(line_density_1), p1, p2);


            if size(line_density, 1) == 0
                if p1 == 1
                    line_density_1 = s_normal_3_2(:, p1 + 10, p2);
                    line_density_1 = line_density_1(find(line_density_1 ~= 0, 1, 'first'):find(line_density_1 ~= 0, 1, 'last'), 1);
                    line_density_all(1:length(line_density_1), p1, p2) = line_density_1;
                    line_density = line_density_all(1:length(line_density_1), p1, p2);
                else
                    line_density_all(:, p1, p2) = line_density_all(:, p1 - 1, p2);
                    line_density = line_density_all(1:find(line_density_all(:, p1 - 1, p2) ~= 0, 1, 'last'), p1 - 1, p2);
                end
            end


            cort_1_x = [];
            cort_2_x = [];
            trab_1 = [];
            if size(line_density, 1) > 8
                cort_1 = line_density(1:find(line_density ./ min(line_density(3:end - 2)) == 1, 1, 'first'));
                cort_2 = line_density(find(line_density ./ min(line_density(3:end - 2)) == 1, 1, 'first') + 1:end);

                cort_1_x = line_density(1:find(cort_1 > max(cort_1) * 0.8, 1, 'last'));
                cort_1_x(1:find((cort_1_x) ./ max(cort_1_x) == 1)) = max(cort_1_x);

                cort_2_x = line_density(find(cort_2 > max(cort_2) * 0.8, 1, 'first') + length(cort_1):end);
                cort_2_x(find((cort_2_x) ./ max(cort_2_x) == 1):end) = max(cort_2_x);

                if length(cort_2_x) + length(cort_1_x) < length(line_density)
                    trab_1 = line_density(length(cort_1_x) + 1:length(line_density) - length(cort_2_x));
                else
                    if length(cort_1_x) > 4
                        trab_1 = cort_1_x(end);
                        cort_1_x = cort_1_x(1:end - 1);
                    else
                        trab_1 = cort_2_x(1);
                        cort_1_x = cort_1_x(2:end);
                    end
                end
            end
            

            if size(line_density, 1) <= 8 && size(line_density, 1) >= 5
                cort_1_x = line_density(1:2);
                cort_1_x(1:2) = max(cort_1_x);
                cort_2_x = line_density(end - 1:end);
                cort_2_x(1:2) = max(cort_2_x);
                trab_1 = line_density(3:end - 2);
                
                if max(cort_1_x) < max(trab_1)
                    cort_1_x(1:2) = max(trab_1);
                end
                if max(cort_2_x) < max(trab_1)
                    cort_2_x(1:2) = max(trab_1);
                end
            end
            

            if size(line_density, 1) < 5
                cort_1_x = max(line_density);
                trab_1 = max(line_density);
                cort_2_x = max(line_density);
            end
            

            [cort_out_d, cort_out_v] = hounsfield2density(cort_1_x);
            [trab_d, trab_v] = hounsfield2density(trab_1);
            [cort_inn_d, cort_inn_v] = hounsfield2density(cort_2_x);


            rho_4_pixel = mean(cort_out_d);
            rho_3_pixel = mean(trab_d);
            rho_2_pixel = mean(cort_inn_d);

            v_4_pixel = mean(cort_out_v);
            v_3_pixel = mean(trab_v);
            v_2_pixel = mean(cort_inn_v);

            th_4_pixel_2 = length(cort_out_d);
            th_3_pixel_2 = length(trab_d);
            th_2_pixel_2 = length(cort_inn_d);
            %th_pixel_2 = th_2_pixel_2 + th_3_pixel_2 + th_4_pixel_2;


            result(1, p1, p2) = th_4_pixel_2; %
            result(2, p1, p2) = th_3_pixel_2; % 
            result(3, p1, p2) = th_2_pixel_2; % 
            result(4, p1, p2) = rho_4_pixel;  % 
            result(5, p1, p2) = rho_3_pixel;  % 
            result(6, p1, p2) = rho_2_pixel;  % 

        end
    end
end
