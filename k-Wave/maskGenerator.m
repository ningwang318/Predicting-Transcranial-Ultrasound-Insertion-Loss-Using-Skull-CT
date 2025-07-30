function [v,rho,alpha_coeff] = maskGenerator(v_test, rho_test, alpha_coeff_input)
    % maskGenerator - Generates 3D masks for skull and brain
    % Inputs:
    %   v_test - 3D matrix of size (x, 136, 136)
    %   rho_test - 3D matrix of size (x, 136, 136)
    %   alpha_coeff_input - 3D matrix of size (x, 136, 136)
    % Outputs:
    %   v - Processed matrix
    %   rho - Processed matrix
    %   alpha_coeff - Processed matrix
    % Ensure the input matrices are of double type
    v = v_test;
    rho = rho_test;
    alpha_coeff = alpha_coeff_input;

    % Find indices where alpha_coeff == 2.7, and replace them
    mask_2_7 = (alpha_coeff == 2.7);
    rho(mask_2_7) = 1850;
    v(mask_2_7) = 2800;
    alpha_coeff(mask_2_7) = 16;
    
    % Find indices where alpha_coeff == 11.5, and replace them
    mask_11_5 = (alpha_coeff == 11.5);
    rho(mask_11_5) = 1700;
    v(mask_11_5) = 2300;
    alpha_coeff(mask_11_5) = 32;
    
    % Padding for front and back positions
    offset = 5;
    padFront = 20 + offset; % dx=0.44mm, 10mm between skull and transducer, 5 grids offset
    totalPad = 272 + offset + 5 - size(v, 1); % Total padding count
    padBack = totalPad - padFront; % Remaining padding at the back
    % Padding the matrices
    v = padarray(v, [padFront, 0, 0], 1500, 'pre'); 
    v = padarray(v, [padBack, 0, 0], 1500, 'post'); 
    rho = padarray(rho, [padFront, 0, 0], 1000, 'pre'); 
    rho = padarray(rho, [padBack, 0, 0], 1000, 'post');     
    alpha_coeff = padarray(alpha_coeff, [padFront, 0, 0], 0, 'pre'); 
    alpha_coeff = padarray(alpha_coeff, [padBack, 0, 0], 0, 'post'); 
end
