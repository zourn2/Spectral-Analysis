% SPOD PRACTICE 1
% Incompressible wake_cylinder from PyFR
clc, clear variables, close all;
addpath('../utils')
addpath('../data')
addpath('../')

%% Load data
t_i = 75.0;
t_e = 280.0;
filename = ['cyl_pyfr_', num2str(t_i), 'to', num2str(t_e), '.mat'];

% Load or extract unstructured grid data, spatial weights, and connectivity (tri)
[X, weight, coords, tri] = import_pyfr(t_i, t_e, filename);

%% SPOD Parameters & Options
x = coords(:, 1); 
y = coords(:, 2);

u = X(:, :, 1);
v = X(:, :, 2);

dt        = 0.05;
nt        = size(u,1);
nDFT      = 512;
nOvlp     = 256;
window    = hann(nDFT);

opts.conflvl    = 0.95;    % confidence interval


%%%%%%%%%%
%% SPOD %%
%%%%%%%%%%

[L,P,f, Lc] = spod(X,window,weight,nOvlp,dt, opts);
fprintf('SPOD with df = %6g Complete.\n', 1/dt/nDFT)


%% Display n-th peak of the frequency
num_peaks = 5;
num_modes = 2;

if ~exist('nDFT', 'var') || isempty(nDFT)
    nDFT = 256;
end

% Display independent n-th mode of peaks
for mi = 1:min(num_modes, size(L, 2))
    fprintf(' < Mode %d >  \n', mi);
    
    L_vec = real(L(:, mi));
    L_vec(isnan(L_vec)) = -Inf;
    
    % 2D local maxima (compare with the two adjacent values on each side)
    is_peak = [false; false; ...
              (L_vec(3:end-2) > L_vec(2:end-3)) & (L_vec(3:end-2) > L_vec(1:end-4)) & ...
              (L_vec(3:end-2) > L_vec(4:end-1)) & (L_vec(3:end-2) > L_vec(5:end)); ...
              false; false];

    % Extract peak and sort
    loc_max_idx = find(is_peak);
    loc_max_val = L_vec(loc_max_idx);

    % Sort by descend in the local maxima group
    [sorted_peaks, sorted_order] = sort(loc_max_val, 'descend');
    sorted_locs = loc_max_idx(sorted_order);

    % Display the top ranking peaks and their corresponding frequencies
    for j = 1:num_peaks
        val_peak = sorted_peaks(j); 
        idx_peak = sorted_locs(j); 
        f_val    = f(idx_peak); 
        
        fprintf('  Rank %d: \t λ = %4.7g \t idx = %4d  ->  freq = %7.4f Hz\n', ...
            j, val_peak, idx_peak, f_val);
    end
end

%% Mode energy spectrum with confidence interval
figure
for mi = 1:size(L,2)
    lh = loglog(f,L(:,mi),'LineWidth',1); hold on
    loglog(f,Lc(:,mi,1),'LineWidth',0.1,'Color',get(lh,'Color'),'LineStyle','--'); % lower confidence level
    loglog(f,Lc(:,mi,2),'LineWidth',0.1,'Color',get(lh,'Color'),'LineStyle','--'); % upper confidence level
end
set(gca,'XScale','log','YScale','log');
xlabel('frequency'), ylabel('SPOD mode energy')
title(sprintf('%2g%% CI for each mode', opts.conflvl*100))

%% Visualize the 1st and 2nd SPOD modes at three frequencies
freq_indices = [6,11,16]; 
modes_to_plot = [1, 2];

% Set component
for var_idx = 1:length(X(1,1,:))  % 1: u, 2: v, 3: p
    figure
    count = 1;
    for fi = freq_indices
        for mi = modes_to_plot
            subplot(3, 2, count)
            mode_shape = squeeze(P(fi, :, var_idx, mi)); % Dimension of P: [nFreq, n_points, n_vars, nModes]
            
            % use trisurf for rendering unstructured data
            patch('Faces', tri, 'Vertices', coords, ...
                  'FaceVertexCData', real(mode_shape(:)), ...
                  'FaceColor', 'interp', 'EdgeColor', 'none');
                  
            axis equal tight; colormap default;
            
            % Set zero as the center of colorbar
            c_max = max(abs(real(mode_shape(:))));
            if c_max > 0
                caxis([-c_max, c_max]);
            end
            
            xlabel('x'), ylabel('y')
            if var_idx == 1
                title(sprintf('u'' : f = %.4f Hz, Mode %d, \\lambda = %.4g', f(fi), mi, L(fi,mi)))
            elseif var_idx == 2
                title(sprintf('v'' : f = %.4f Hz, Mode %d, \\lambda = %.4g', f(fi), mi, L(fi,mi)))
            elseif var_idx == 3
                title(sprintf('p'' : f = %.4f Hz, Mode %d, \\lambda = %.4g', f(fi), mi, L(fi,mi)))
            end
            count = count + 1;
        end
    end
end
