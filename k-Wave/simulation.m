%clearvars;
% medium parameters
% c0              = 1500;     % sound speed [m/s]
% rho0            = 1000;     % density [kg/m^3]
% source parameters
source_frequencies = [220000,650000,1000000];     % source frequency [Hz]
source_roc      = 0.075;    % bowl radius of curvature [m]
source_diameter = 0.06;    % bowl aperture diameter [m]
source_mag      = 60000;      % source pressure [Pa]
% grid parameters
axial_size      = 0.12;    % total grid size in the axial dimension [m]
lateral_size    = 0.06;  % total grid size in the lateral dimension [m]
horizenal_size  = 0.06;
% computational parameters
ppw             = 6;        % number of points per wavelength
t_end           = 1.2e-4;    % total compute time [s] (this must be long enough to reach steady state)
record_periods  = 1;        % number of periods to record
cfl             = 0.06;     % CFL number
source_x_offset = 5;      % grid points to offset the source
bli_tolerance   = 0.1;     % tolerance for truncation of the off-grid source points
upsampling_rate = 8;       % density of integration points relative to grid

folderPath ='E:\Ning\Simulation\Simulation\skull-ct';
fileList = dir(fullfile(folderPath, '*.mat')); 
fprintf('find %d  .mat file：\n', length(fileList));
for i = 1:length(fileList)
    fprintf('%d: %s\n', i, fileList(i).name);
end
for fileIdx = 1:length(fileList)
    filePath = fullfile(folderPath, fileList(fileIdx).name);
    data = load(filePath);    
    [v, rho, alpha_coeff] = maskGenerator(data.v_test, data.rho_test, data.alpha_coeff); 

    for f_index = 1:length(source_frequencies)
        source_f0 = source_frequencies(f_index); % Update source frequency
        
        % Calculate the grid spacing based on the PPW and F0
        dx = 4.4e-4;  % [m], pre-defined spacing
        
        % Compute the size of the grid (offset + pml_size)
        Nx = roundEven(axial_size / dx) + source_x_offset + 2 ;
        Ny = roundEven(lateral_size / dx) ;
        Nz = roundEven(horizenal_size / dx) ;
        % Create the computational grid
        kgrid = kWaveGrid(Nx, dx, Ny, dx, Nz, dx);
        
        % Compute points per temporal period
        PPP = round(ppw / cfl);
        dt = 1 / (PPP * source_f0); % Time spacing
        Nt = round(t_end / dt);
        kgrid.setTime(Nt, dt);
        
        % Display calculated parameters
        disp(['Simulating for f0 = ' num2str(source_f0) ' Hz']);
        
        % Create time varying source
        source_sig = createCWSignals(kgrid.t_array, source_f0, source_mag, 0);
    % Create empty kWaveArray and add a bowl element
        karray = kWaveArray('BLITolerance', bli_tolerance, 'UpsamplingRate', upsampling_rate);
        bowl_pos = [source_x_offset * dx - 0.06, 0.0, 0.0];
        focus_pos = [0.06, 0, 0];
        karray.addBowlElement(bowl_pos, source_roc, source_diameter, focus_pos);
        
        % Set the source properties
        source.p_mask = karray.getArrayBinaryMask(kgrid);
        source.p = karray.getDistributedSourceSignal(kgrid, source_sig);
        
        % Set medium properties
        
        medium.sound_speed = v;
        medium.density = rho;
        medium.alpha_coeff = alpha_coeff;
        medium.alpha_power = 2;
        % Assign medium properties for different masks
        % medium.sound_speed(skull_mask) = 2800;
        % medium.density(skull_mask) = 1850;
        % medium.alpha_coeff(skull_mask) = 16;
        % 
        % medium.sound_speed(brain_mask) = 1560;
        % medium.density(brain_mask) = 1040;
        % medium.alpha_coeff(brain_mask) = 1.2;
        
        % Set sensor mask and recording settings
        sensor.mask = zeros(Nx, Ny, Nz);
        sensor.mask(source_x_offset + 2:end, :, :) = 1;
        sensor.record = {'p'};
        sensor.record_start_index = kgrid.Nt - record_periods * PPP + 1;
        
        % Run the simulation
        input_args = {
            'PMLSize', [24, 16, 16], 'PMLAlpha', 1.5, ...
            'PMLInside', false, ...
            'PlotPML', false, ...
            'DisplayMask', 'off'
        };
        
        sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor, ...
                    input_args{:}, ...
                    'DataCast', 'gpuArray-single', ...
                    'PlotScale', [-1, 1] * source_mag);
        
        % Reshape and process the results
        sensor_data_reshaped = reshape(sensor_data.p, 273, 136, 136, []);
        amp = extractAmpPhase(sensor_data_reshaped, 1 / kgrid.dt, source_f0, ...
            'Dim', 4, 'Window', 'Rectangular', 'FFTPadding', 1);
        amp = reshape(amp, 273, 136, []);
        amp = gather(amp); % Bring data from GPU to CPUclear
        % Save the result for each frequency
        [~, skullFileName, ~] = fileparts(fileList(fileIdx).name);
        save(['E:\Ning\Simulation\Simulation\result\amp_result_' skullFileName '_f0_' num2str(source_f0) '.mat'], 'amp');
        disp(['simulation complete，frequency f0 = ' num2str(source_f0) ' Hz，file ' fileList(fileIdx).name 'results saved']);
    end
end

