function [v,rho,alpha_coeff] = maskGenerator(v_test, rho_test, alpha_coeff_input)

    v = v_test;
    rho = rho_test;
    alpha_coeff = alpha_coeff_input;

    alpha_coeff(alpha_coeff == 2.7) = 4;
    alpha_coeff(alpha_coeff == 11.5) = 8;
 
    offset = 5;
    padFront = 20 + offset; % dx=0.44mm， 10mm between skull and transducer，5 grids offset 
    totalPad = 272 + offset + 2 - size(v, 1); 
    padBack = totalPad - padFront; 
  
    v = padarray(v, [padFront, 0, 0], 1500, 'pre'); 
    v = padarray(v, [padBack, 0, 0], 1500, 'post'); 
    rho = padarray(rho, [padFront, 0, 0], 1000, 'pre'); 
    rho = padarray(rho, [padBack, 0, 0], 1000, 'post');     
    alpha_coeff = padarray(alpha_coeff, [padFront, 0, 0], 0, 'pre'); 
    alpha_coeff = padarray(alpha_coeff, [padBack, 0, 0], 0, 'post'); 

    
    % % 生成 skullMask
    % skullMask = false(size(paddedMatrix)); 
    % skullMask(paddedMatrix > threshold) = true; 
    % % 初始化 brainMask
    % brainMask = false(size(skullMask)); 

    % for i = 1:size(skullMask, 2) 
    %     for j = 1:size(skullMask, 3)

    %         x_indices = find(skullMask(:, i, j), 1, 'last');
    %         if ~isempty(x_indices)

    %             brainMask(x_indices+1:end, i, j) = true;
    %         end
    %     end
    % end

%     figure;
%     sliceViewer(skullMask);
%     title('3D Skull Mask');
% 
%     figure;
%     sliceViewer(brainMask);
%     title('3D Brain Mask');
end
