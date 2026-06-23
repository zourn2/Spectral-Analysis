function [X, weight, coords, tri] = import_pyfr(t_i, t_e, filename)

% Incompressible wake_cylinder from PyFR
filepath = fullfile('data', filename);

if exist(filepath, 'file') == 2
    load(filepath)
    disp('Load exist file: X, weight, coords, tri')

else
    %% Read csv and sort
    dataDir = 'raw_cyl/csv'; % Path of the csv folder
    warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');
    filePattern = fullfile(dataDir, '*.csv');
    csvFiles = dir(filePattern);

    time_values = zeros(length(csvFiles), 1);
    for i = 1:length(csvFiles)
        % Extract time index from filename (e.g. 'result_0.05.csv' -> '0.05')
        numStr = regexp(csvFiles(i).name, '\d+\.?\d*', 'match');

        if ~isempty(numStr)
            time_values(i) = str2double(numStr{1});
        else
            time_values(i) = NaN;
        end
    end

    % Extract the data inside the time interval
    target_idx = (time_values >= t_i) & (time_values <= t_e);
    csvFiles = csvFiles(target_idx);
    time_values = time_values(target_idx);
    [~, sortIdx] = sort(time_values);
    csvFiles = csvFiles(sortIdx);

    %% Remove duplicate coordinates
    firstFilePath = fullfile(csvFiles(1).folder, csvFiles(1).name);
    firstData = readtable(firstFilePath);
    nt = length(csvFiles);

    % Resolved using the 'Clean to Grid' filter in ParaView
    % If you can't use that filter, uncomment below for remove duplicates

    % raw_coords = [firstData.Points_0, firstData.Points_1];
    % [coords, ia, ~] = unique(raw_coords, 'rows', 'stable');
    coords = [firstData.Points_0, firstData.Points_1];
    n_points = size(coords, 1);

    x = coords(:, 1); 
    y = coords(:, 2);

    % Delaunay triangulation
    DT = delaunayTriangulation(x, y);
    tri = DT.ConnectivityList;

    % Calculate dV
    P1 = DT.Points(tri(:,1), :);
    P2 = DT.Points(tri(:,2), :);
    P3 = DT.Points(tri(:,3), :);
    tri_areas = 0.5 * abs(P1(:,1).*(P2(:,2)-P3(:,2)) + ...
        P2(:,1).*(P3(:,2)-P1(:,2)) + ...
        P3(:,1).*(P1(:,2)-P2(:,2)));

    I = tri(:);                              
    J = repmat(tri_areas / 3, 3, 1);         
    dV_vector = accumarray(I, J, [n_points, 1]); 
    dV_vector(dV_vector <= 0) = mean(dV_vector);
    dV     = dV_vector;
    weight = repmat(dV, [1, 3]);


    %% Preallocate X and data loading
    if isempty(gcp('nocreate'))
        disp('Start CPU Parallel Pool');
        parpool; 
    end
    
    disp('Reading .csv -> Constructing data matrix X...');
    X = zeros(nt, n_points, 3);

    tic;
    parfor ti = 1:nt
        warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');
        filePath = fullfile(csvFiles(ti).folder, csvFiles(ti).name);
        currentData = readtable(filePath);

        u_ti = currentData.Velocity_0(:);
        v_ti = currentData.Velocity_1(:);
        p_ti = currentData.Pressure(:);

        X_ti = zeros(1, n_points, 3);
        X_ti(1, :, 1) = u_ti;
        X_ti(1, :, 2) = v_ti;
        X_ti(1, :, 3) = p_ti

        X(ti, :, :) = X_ti;
    end
    toc;
    disp('Constructing data matrix X complete');

    % Save data to .mat
    save(filepath, 'X', 'weight', 'coords', 'tri', '-v7.3');
end