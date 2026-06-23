% BMD PRACTICE 2
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

%% BMD Parameters & Options
dt              = 0.05;
nt              = size(X,1);
nDFT        = 1024;
nOvlp        = 512;
window     = hann(nDFT);

% opts.regions= 1:8;  % BMD regions -> 1:8 means entire region of BMD
opts.regions = [1];

% Restrict max frequency index: f1, f2 < opts.nfreq / dt / nDFT
limit_f = 4.0;
opts.nfreq  = ceil(limit_f * dt * nDFT);


%%%%%%%%%
%% BMD %%
%%%%%%%%%
%[B,P,f,idx] = bmd(X);
%[B,P,f_idx,idx] = bmd_par(X);
[B,P,f,idx] = bmd_gpu(X);

[f1,f2]    = ndgrid(f);    % Create a 2D frequency grid
fprintf('BMD with df = %6g Complete.\n', 1/dt/nDFT)
disp(' ')


%% Display n-th peak of the frequency triad
num_peaks = 5;

B_abs = abs(B);
B_abs(isnan(B_abs)) = -1;

% 2D local maxima
is_peak = false(size(B_abs));
B_c = B_abs(2:end-1, 2:end-1);

% Compare with surroundings
is_peak_inner = (B_c > B_abs(1:end-2, 1:end-2)) & ... % upper left
                (B_c > B_abs(1:end-2, 2:end-1)) & ...            % upper
                (B_c > B_abs(1:end-2, 3:end))   & ...             % upper right
                (B_c > B_abs(2:end-1, 1:end-2)) & ...            % left
                (B_c > B_abs(2:end-1, 3:end))   & ...             % right
                (B_c > B_abs(3:end,   1:end-2)) & ...             % lower left
                (B_c > B_abs(3:end,   2:end-1)) & ...             % lower
                (B_c > B_abs(3:end,   3:end));                      % lower right

is_peak(2:end-1, 2:end-1) = is_peak_inner;

% Extract peak and sort
loc_max_idx = find(is_peak);
loc_max_val = B_abs(loc_max_idx);

% Sort by descend in the local maxima group
[sorted_peaks_abs, sorted_order] = sort(loc_max_val, 'descend');
sorted_locs = loc_max_idx(sorted_order);

n_found = min(num_peaks, length(sorted_peaks_abs));

% Display the top ranking peaks and their corresponding frequencies
for k = 1:n_found
    idx_peak = sorted_locs(k);
    
    % Compute log()
    val_peak_log = log(sorted_peaks_abs(k));    
    
    % Map coordinate and physical frequency
    [row, col] = ind2sub(size(B), idx_peak);
    
    % Extract frequency
    f_k = f(row); f_l = f(col);
    f_kl = f_k + f_l;

    % Set the index of zero frequency as zero-offset
    [~, zero_idx] = min(abs(f)); 
    i_k = row - zero_idx; i_l = col - zero_idx; 
    i_kl = i_k + i_l;
    
    fprintf('  Rank %2d:  λ_1 = %8.4f \t idx = (%2d, %2d, %2d)  ->  f = {%7.4f, %7.4f, %7.4f} Hz\n', ...
        k, val_peak_log, i_k, i_l, i_kl, f_k, f_l, f_kl);
end

%% Mode magnitude bispectrum
spec_fig = figure;
subplot(3,4,[1 2 5 6 9 10]);

contourf(f1, f2, log(abs(B)), 100, 'linecolor', 'none'); 
axis equal tight; colormap default;
c = colorbar('Location', 'east'); 
c.Label.String = '|\lambda_1|';
xlabel('f_1'), ylabel('f_2');
xlim([min(f1(idx)) max(f1(idx))]); ylim([min(f2(idx)) max(f2(idx))]);
title('Mode bispectrum');

%% Interactive visualization of modes (Unstructured)
dcm             = datacursormode(spec_fig);
dcm.Enable      = 'on';
dcm.UpdateFcn   = @displayTriplet;
disp('Click any point in the mode bispecrtum to plot modes, or ESC to exit visualization mode!');

while 1
    waitforbuttonpress;

    % Exit while loop if ESC is pressed
    key = get(gcf,'currentcharacter');
    if key==27, break, end

    point = getCursorInfo(dcm);
    if isempty(point), continue; end

    % Find the linear index of the frequency triad {f2,f2,f1+f2}
    [~, min_idx] = min((f1(:) - point.Position(1)).^2 + (f2(:) - point.Position(2)).^2);
    triadIdx     = find(idx == min_idx);  

    if isempty(triadIdx)
        disp('No interactions here. Click another point!');
        continue;
    end

    % Use trisurf instead of pcolor
    % u-component bispectral modes (k+l)
    subplot(3,4,3)
    mode  = squeeze(P(1, triadIdx, :, 1)); % Dimension of P: [2, triadIdx, n_points, # of parameters]
    trisurf(tri, coords(:, 1), coords(:, 2), zeros(size(coords(:, 1))), real(mode), 'EdgeColor', 'none'); view(2);
    axis equal tight; caxis(max(abs(mode(:)))*0.5*[-1 1]);
    xlabel('x'), ylabel('y'), title('\phi^u_{k+l}');

    % v-component bispectral modes (k+l)
    subplot(3,4,4)
    mode  = squeeze(P(1, triadIdx, :, 2));
    trisurf(tri, coords(:, 1), coords(:, 2), zeros(size(coords(:, 1))), real(mode), 'EdgeColor', 'none'); view(2);
    axis equal tight; caxis(max(abs(mode(:)))*0.5*[-1 1]);
    xlabel('x'), ylabel('y'), title('\phi^v_{k+l}');

    % u-component cross-frequency field (k o l)
    subplot(3,4,7)
    mode  = squeeze(P(2, triadIdx, :, 1));
    trisurf(tri, coords(:, 1), coords(:, 2), zeros(size(coords(:, 1))), real(mode), 'EdgeColor', 'none'); view(2);
    axis equal tight; caxis(max(abs(mode(:)))*0.5*[-1 1]);
    xlabel('x'), ylabel('y'), title('\phi^u_{k\circ l}');

    % v-component cross-frequency field (k o l)
    subplot(3,4,8)
    mode  = squeeze(P(2, triadIdx, :, 2));
    trisurf(tri, coords(:, 1), coords(:, 2), zeros(size(coords(:, 1))), real(mode), 'EdgeColor', 'none'); view(2);
    axis equal tight; caxis(max(abs(mode(:)))*0.5*[-1 1]);
    xlabel('x'), ylabel('y'), title('\phi^v_{k\circ l}');

    % u-component interaction map
    subplot(3,4,11)
    mode  = abs(squeeze(P(1, triadIdx, :, 1)) .* squeeze(P(2, triadIdx, :, 1)));
    trisurf(tri, coords(:, 1), coords(:, 2), zeros(size(coords(:, 1))), real(mode), 'EdgeColor', 'none'); view(2);
    axis equal tight; 
    xlabel('x'), ylabel('y'), title('\phi^u_{k\circ l}\circ\phi^u_{k+l}');

    % v-component interaction map
    subplot(3,4,12)
    mode  = abs(squeeze(P(1, triadIdx, :, 2)) .* squeeze(P(2, triadIdx, :, 2)));
    trisurf(tri, coords(:, 1), coords(:, 2), zeros(size(coords(:, 1))), real(mode), 'EdgeColor', 'none'); view(2);
    axis equal tight; 
    xlabel('x'), ylabel('y'), title('\phi^v_{k\circ l}\circ\phi^v_{k+l}');

    drawnow;   
    figure(spec_fig);
end


%% Data tooltip display function
function txt = displayTriplet(~,info)
x = info.Position(1);
y = info.Position(2);
txt = ['{f1: ' num2str(x,'%.4g') ', f2: ' num2str(y,'%.4g') ', f1+f2: ' num2str(x+y,'%.4g') '}'];
end
