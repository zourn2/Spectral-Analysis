% Example 4 is the adaptation for the jetLES from SPOD data 

clc, clear variables
close all
addpath('utils')

%% Load data
load('data/jetLES.mat')

nDFT         = 2048;
nOvlp        = 1024;
window       = hann(nDFT);
dV           = (x(2,1)-x(1,1))*(r(1,2)-r(1,1));
weight       = trapzWeightsPolar(r(:,1),x(1,:));
opts.regions = 1;
% opts.regions = 1:8;
opts.nfreq  = 100;
% opts.precision = 'single';
X            = zeros(nt/2,nr,nx,1);
X(:,:,:,1)   = p; clear p

%%%%%%%%%
%% BMD %%
%%%%%%%%%
%[B,P,f,idx] = bmd(X,nDFT,weight,nOvlp,dt,opts);
[B,P,f,idx] = bmd_par(X,"Threads",8,nDFT,weight,nOvlp,dt,opts);
%[B,P,f,idx] = bmd_gpu(X,nDFT,weight,nOvlp,dt,opts);
[f1,f2]     = ndgrid(f);

%% Mode magnitude bispectrum
spec_fig = figure;
subplot(3,4,[1 2 5 6 9 10]);
contourf(f1,f2,log(abs(B)),100,'linecolor','none'); axis equal tight
c   = colorbar('Location','east'); c.Label.String = '|\lambda_1|';
xlabel('f_1'), ylabel('f_2'), zlabel('|\lambda_1|'); axis equal
xlim([min(f1(idx)) max(f1(idx))]); ylim([min(f2(idx)) max(f2(idx))])
title('Mode bispectrum')

%% Interactive visualization of modes
dcm             = datacursormode(spec_fig);
dcm.Enable      = 'on';
dcm.UpdateFcn   = @displayTriplet;
disp(' ');
disp('Click any point in the mode bispecrtum to plot modes, or ESC to exit visualization mode!');
while 1
    waitforbuttonpress;
    
    % exit while loop if ESC is pressed
    key = get(gcf,'currentcharacter');
    if key==27, break, end
       
    point       = getCursorInfo(dcm);
    triadIdx    = find(idx==find(f1==point.Position(1)&f2==point.Position(2)));  % find the linear index of the frequency triad {f2,f2,f1+f2}
    
    subplot(3,4,3)
    mode  = squeeze(P(1,triadIdx,:,:,1));
    pcolor(x,r,real(mode)), axis equal tight, caxis(max(abs(mode(:)))*0.5*[-1 1])
    xlabel('x'), ylabel('r'), title('\phi^u_{k+l}')
    shading interp
    
    subplot(3,4,4)
    mode  = squeeze(P(1,triadIdx,:,:,1));
    pcolor(x,r,real(mode)), axis equal tight, caxis(max(abs(mode(:)))*0.5*[-1 1])
    xlabel('x'), ylabel('r'), title('\phi^v_{k+l}');
    shading interp
    
    subplot(3,4,7)
    mode  = squeeze(P(2,triadIdx,:,:,1));
    pcolor(x,r,real(mode)), axis equal tight, caxis(max(abs(mode(:)))*0.5*[-1 1])
    xlabel('x'), ylabel('r'), title('\phi^u_{k\circ l}');
    shading interp
    
    subplot(3,4,8)
    mode  = squeeze(P(2,triadIdx,:,:,1));
    pcolor(x,r,real(mode)), axis equal tight, caxis(max(abs(mode(:)))*0.5*[-1 1])
    xlabel('x'), ylabel('r'), title('\phi^v_{k\circ l}');
    shading interp   
    
    subplot(3,4,11)
    mode  = abs(squeeze(P(1,triadIdx,:,:,1)).*squeeze(P(2,triadIdx,:,:,1)));
    pcolor(x,r,real(mode)), axis equal tight
    xlabel('x'), ylabel('r'), title('\phi^u_{k\circ l}\circ\phi^u_{k+l}');
    shading interp
    
    subplot(3,4,12)
    mode  = abs(squeeze(P(1,triadIdx,:,:,1)).*squeeze(P(2,triadIdx,:,:,1)));
    pcolor(x,r,real(mode)), axis equal tight
    xlabel('x'), ylabel('r'), title('\phi^v_{k\circ l}\circ\phi^v_{k+l}');
    shading interp     
    
    drawnow   
    figure(spec_fig)
end

function txt = displayTriplet(~,info)
    x = info.Position(1);
    r = info.Position(2);
    txt = ['{' num2str(x,'%.2g') ',' num2str(r,'%.2g') ',' num2str(x+r,'%.2g') '}'];
end