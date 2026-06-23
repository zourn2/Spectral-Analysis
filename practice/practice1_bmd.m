% BMD PRACTICE 1 
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


%% Inspect wake data
u = X(:, :, 1);
v = X(:, :, 2);
p = X(:, :, 3);

u_mean = mean(u, 1);
v_mean = mean(v, 1);
p_mean = mean(p, 1);

dt = 0.05;
nt = size(u,1);

% u-velocity fluctuation
subplot(3,1,1)
h_u = patch('Faces', tri, 'Vertices', coords, 'FaceVertexCData', (u(1,:) - u_mean)', ...
    'FaceColor', 'interp', 'EdgeColor', 'none');
axis equal tight; colormap default; caxis([-0.5 0.5]);
xlabel('x'), ylabel('y'); 
title_u = title(['u'' (t = ', num2str(dt), ')']); % Store title handle for fast graphical updates

% v-velocity fluctuation
subplot(3,1,2)
h_v = patch('Faces', tri, 'Vertices', coords, 'FaceVertexCData', (v(1,:) - v_mean)', ...
    'FaceColor', 'interp', 'EdgeColor', 'none');
axis equal tight; colormap default; caxis([-1 1]);
xlabel('x'), ylabel('y'); title('v''');

% pressure fluctuation
subplot(3,1,3)
h_p = patch('Faces', tri, 'Vertices', coords, 'FaceVertexCData', p(1,:)', ...
    'FaceColor', 'interp', 'EdgeColor', 'none');
axis equal tight; colormap default; caxis([0 1.5]);
xlabel('x'), ylabel('y'); title('p');

% Render Animation Loop
for ti =2500:2600
    % Update u' (Fluctuation)
    h_u.FaceVertexCData = (u(ti,:) - u_mean)';
    title_u.String      = ['u'' (t = ', num2str(t_i + ti*dt), ')']; % Update time string
    
    % Update v' (Fluctuation)
    h_v.FaceVertexCData = (v(ti,:) - v_mean)';
    
    % Update p' (Fluctuation)
    h_p.FaceVertexCData = p(ti,:)';
    
    drawnow;
    % GIF saving block (Uncomment to export)
    % frame = getframe(fig_anim);
    % [im, map] = rgb2ind(frame2im(frame), 256);
    % if ti == 1300
    %     imwrite(im, map, 'wake.gif', 'gif', 'LoopCount', Inf, 'DelayTime', 0.001);
    % else
    %     imwrite(im, map, 'wake.gif', 'gif', 'WriteMode', 'append', 'DelayTime', 0.001);
    % end 
end


%%%%%%%%%
%% BMD %%
%%%%%%%%%

%[B,P,f_idx,idx] = bmd(X);
%[B,P,f_idx,idx] = bmd_par(X);
[B,P,f_idx,idx] = bmd_gpu(p);

if ~exist('nDFT', 'var') || isempty(nDFT)
    nDFT = 256;             % Default length(f_idx);
end
[i1,i2] = ndgrid(f_idx);    % Create a 2D frequency index grid

f = f_idx / (nDFT * dt);
[f1,f2]     = ndgrid(f);    % Create a 2D frequency grid
fprintf('BMD with df = %6g Complete \n', 1/dt/nDFT)
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

% Display the top ranking triad interactions and their corresponding frequencies
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

%% Plot mode magnitude bispectrum
figure
subplot(2,2,[1 3])
contourf(i1, i2, log(abs(B)), 100, 'linecolor', 'none'); 
axis equal tight; colormap default;
xlim([0 max(f_idx)]); 
ylim([min(f_idx) max(f_idx)/2]); 
xlabel('k'), ylabel('l');
title('Mode bispectrum');
hold on;

%% Plot bispectral mode and interaction map for triad (k,l,k+l)
target_k = 5;  
target_l = -3; 

% Map targets to the closest actual frequencies in the array to prevent floating-point errors
[~, k_idx] = min(abs(f_idx - target_k));
[~, l_idx] = min(abs(f_idx - target_l));
k = f_idx(k_idx);
l = f_idx(l_idx);
fk = f(k_idx);
fl = f(l_idx);

% Convert 2D target grid coordinates to a linear index to query the valid idx array
linear_target = sub2ind(size(f1), k_idx, l_idx);
triadIdx = find(idx == linear_target);

if isempty(triadIdx)
    warning(['The selected frequency index pair (%.4f, %.4f) is outside the valid (or symmetric) region. ' ...
            'Please re-adjust the target peaks based on the Bispectrum plot.'], k, l);
else
    % Mark the identified peak location on the contour plot
    plot(k, l, 'k+', 'MarkerSize', 8, 'LineWidth', 2);
    text(k + 1.5, l + 0.8, sprintf('(%d, %d, %d)\n->{%7.4f, %7.4f, %7.4f}', k, l, k+l, fk, fl, fk+fl), 'Color', 'w', 'FontWeight', 'bold');
    
    % Bispectral Spatial Mode (p-component)
    subplot(2,2,2)
    phi1 = squeeze(P(1, triadIdx, :)); 
    patch('Faces', tri, 'Vertices', coords, 'FaceVertexCData', real(phi1), ...
          'FaceColor', 'interp', 'EdgeColor', 'none');
    axis equal tight; caxis(max(abs(phi1))*0.5*[-1 1]); 
    xlabel('x'), ylabel('y'); 
    title(sprintf('\\phi_{%+d, %+d} (Bispectral mode p)', k, l));
    
    % Spatial Interaction Map
    subplot(2,2,4)
    psi   = abs(phi1 .* phi1);        
    patch('Faces', tri, 'Vertices', coords, 'FaceVertexCData', psi, ...
          'FaceColor', 'interp', 'EdgeColor', 'none');
    axis equal tight; caxis(max(abs(psi))*[0 1]); 
    xlabel('x'), ylabel('y'); 
    title(sprintf('\\psi_{%+d, %+d} (Interaction map)', k, l));
end
