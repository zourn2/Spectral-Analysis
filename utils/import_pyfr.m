function [X, weight, coords, tri] = import_pyfr(t_i, t_e, filename)

% Incompressible wake_cylinder from PyFR
filepath = fullfile('data', filename);

if exist(filepath, 'file') == 2
    load(filepath)
    disp('Load exist file: X, weight, coords, tri')

else
    %% 1. CSV 파일 목록 읽기 및 시간순 정렬
    dataDir = 'raw_cyl/csv'; % CSV 폴더 경로
    warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');
    filePattern = fullfile(dataDir, '*.csv');
    csvFiles = dir(filePattern);

    % 파일명에서 추출한 숫자를 담을 배열 미리 생성
    time_values = zeros(length(csvFiles), 1);

    for i = 1:length(csvFiles)
        % 정규표현식을 사용하여 파일명에서 숫자 추출 (예: 'snap_0.05.csv' -> '0.05')
        % '\d+\.?\d*' : 숫자(\d)가 반복(+)되고, 소수점(\.?)이 있을 수도 있으며, 뒤에 숫자(\d*)가 붙는 패턴
        numStr = regexp(csvFiles(i).name, '\d+\.?\d*', 'match');

        if ~isempty(numStr)
            % 추출된 문자열 중 첫 번째 값을 실제 숫자로 변환
            time_values(i) = str2double(numStr{1});
        else
            time_values(i) = NaN; % 숫자를 찾지 못한 경우 예외 처리
        end
    end

    % 원하는 시간 범위 내의 데이터만 추출
    target_idx = (time_values >= t_i) & (time_values <= t_e);
    csvFiles = csvFiles(target_idx);
    time_values = time_values(target_idx);

    % 추출한 숫자(time_values)를 기준으로 오름차순 정렬하여 인덱스(sortIdx) 획득
    [~, sortIdx] = sort(time_values);

    % csvFiles 구조체를 정렬된 인덱스에 맞게 완벽하게 재배치
    csvFiles = csvFiles(sortIdx);

    %% 2. 첫 번째 파일에서 중복 좌표 제거 및 dV(면적) 계산
    firstFilePath = fullfile(csvFiles(1).folder, csvFiles(1).name);
    firstData = readtable(firstFilePath);
    nt = length(csvFiles); % 시간 스냅샷 수

    % 전체 좌표 추출 및 중복 제거 (고유 격자점 추출) 
    % -> Paraview의 Clean to Grid로 해결
    % raw_coords = [firstData.Points_0, firstData.Points_1];
    % [coords, ia, ~] = unique(raw_coords, 'rows', 'stable');
    coords = [firstData.Points_0, firstData.Points_1];
    n_points = size(coords, 1); % 고유한 격자점(Node)의 총 개수

    x = coords(:, 1); 
    y = coords(:, 2);

    % 델로네 삼각분할 수행 (중복이 제거되어 경고가 뜨지 않습니다)
    DT = delaunayTriangulation(x, y);
    tri = DT.ConnectivityList;

    % 각 삼각형의 면적 계산
    P1 = DT.Points(tri(:,1), :);
    P2 = DT.Points(tri(:,2), :);
    P3 = DT.Points(tri(:,3), :);
    tri_areas = 0.5 * abs(P1(:,1).*(P2(:,2)-P3(:,2)) + ...
        P2(:,1).*(P3(:,2)-P1(:,2)) + ...
        P3(:,1).*(P1(:,2)-P2(:,2)));

    % 각 노드에 면적 분배 (dV 계산)
    I = tri(:);                              
    J = repmat(tri_areas / 3, 3, 1);         
    dV_vector = accumarray(I, J, [n_points, 1]); 
    dV_vector(dV_vector <= 0) = mean(dV_vector); % 외곽 노드 예외 보정
    dV     = dV_vector;
    weight = repmat(dV, [1, 2]); % u, v 성분을 위한 공간 가중치 [n_points x 2]


    %% 3. BMD/SPOD용 메인 행렬 X 사전 할당 및 데이터 적재
    % [u, v] 2개의 변수만 사용하므로 마지막 차원은 2입니다.
    if isempty(gcp('nocreate'))
        disp('Start CPU Parallel Pool');
        parpool; 
    end
    
    disp('Reading .csv -> Constructing data matrix X...');
    X = zeros(nt, n_points, 2);

    tic;
    parfor ti = 1:nt
        warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');
        filePath = fullfile(csvFiles(ti).folder, csvFiles(ti).name);
        currentData = readtable(filePath);

        % 속도 성분 분리한 후 한번에 적재
        u_ti = currentData.Velocity_0(:);
        v_ti = currentData.Velocity_1(:);

        X_ti = zeros(1, n_points, 2);
        X_ti(1, :, 1) = u_ti;
        X_ti(1, :, 2) = v_ti;

        X(ti, :, :) = X_ti;
    end
    toc;

    disp('Constructing data matrix X complete');

    % Save data to .mat
    save(filepath, 'X', 'weight', 'coords', 'tri', '-v7.3');
end