function [L,P,f,idx,T] = bmd_gpu(X,varargin)
% BMD Bispectral Mode Decomposition
%
%          f2 or l
%             ^
%     ________|
%     |\      |\
%     |  \  7 |  \
%     | 6  \  | 8 /\
%     |      \| / 1  \
% ----+-------+-------+-> f1 or k
%      \  5 / |\      |
%        \/ 4 |  \  2 |
%          \  | 3  \  |
%            \|______\|
%             |
% Figure. Regions in the f1-f2 plane (select via OPTS.regions). Triads are
%         expressed as frequency triplets, {f1,f2,f1+f2}, or frequency
%         index triplets, (k,l,k+l).
%  
%  Default parallel pool setting : parpool('Threads', 4);
%
%  [L,P,F,IDX] = BMD(X) returns the bispectral mode decomposition of the
%  data matrix X. The first dimension of X must be time (use PERMUTE if it
%  does not). X can have any number of additional spatial dimensions or
%  variable indices. The mode bispectrum is returned in L and the modes in
%  P. The spatial dimensions of the modes are identical to those of X. The
%  bispectral modes are stored in P(1,...) and the cross-frequency fields
%  in P(2,...). The second index of P is the triad index. IDX contains the
%  linear index corresponding to the row and column subscripts of the
%  frequency doublets. Refer to the examples for conversion from linear
%  index to frequency doublets using IDX (Matlab does not support sparse
%  arrays). F is the frequency vector. If DT is not specified, the
%  frequency index is returned in F. Although BMD(X) automatically chooses
%  default spectral estimation parameters, it is recommended to manually
%  specify problem-dependent parameters on a case-to-case basis.
%
%  [L,P,F,IDX] = BMD(X,WINDOW) uses a temporal window. If WINDOW is a
%  vector, X is divided into segments of the same length as WINDOW. Each
%  segment is then weighted (pointwise multiplied) by WINDOW. If WINDOW is
%  a scalar, a Hamming window of length WINDOW is used. If WINDOW is
%  omitted or empty, a Hamming window is used.
%
%  [L,P,F,IDX] = BMD(X,WINDOW,WEIGHT) uses a spatial inner product weight,
%  usually quadrature weights. WEIGHT must have the same spatial dimensions
%  as X.
%
%  [L,P,F,IDX] = BMD(X,WINDOW,WEIGHT,NOVERLAP) increases the number of
%  segments by overlapping consecutive blocks by NOVERLAP snapshots.
%  NOVERLAP defaults to 50% of the length of WINDOW if not specified.
%
%  [L,P,F,IDX] = BMD(X,WINDOW,WEIGHT,NOVERLAP,DT) uses the time step DT
%  between consecutive snapshots to determine a physical frequency F.
%
%  [L,P,F,IDX] = BMD(X,WINDOW,WEIGHT,NOVERLAP,DT,OPTS) specifies options:
%  OPTS.regions     regions of the bispectrum to compute, see figure
%                   above [vector | {[1 2]}]
%  OPTS.precision   compute BMD in single precision [true | {false}]
%  OPTS.mean        provide a mean that is subtracted from each
%                   snapshot [array of size X | {temporal mean of X}]
%  OPTS.solver      optimizer for x*Ax [{'MengiOverton'} | 'HeWatson' |
%                   'simpleit'] 
%  OPTS.tol         tolerance for optimizer [scalar | {1e-6}]
%  OPTS.nfreq       restrict computation to |l|,|k|<=OPTS.nfreq
%                   [integer | {all}] 
%  OPTS.nitmax      number of iterations to converge numerical radius 
%                   [integer | 500] 
%
%  References:
%   [1] Schmidt, O. T., Bispectral mode decomposition of nonlinear flows,
%       Nonlinear Dynamics, 2020
%       DOI 10.1007/s11071-020-06037-z
%       https://rdcu.be/cbg3D
%
% O. T. Schmidt (oschmidt@ucsd.edu)
% Last revision: 17-Aug-2023
%       - Mengi & Overton's (2005) globally convergent numerical radius
%       algorithm implemented by B. Yeung is now standard
%       - standard tolerance changed to 1e-6

evalc('delete(gcp(''nocreate''))');
parpool('Threads'); 
maxNumCompThreads(1);

single_prec = false;
if length(varargin) >= 5
    opts = varargin{5};
    if ~isfield(opts,'regions'),  opts.regions = [1 2];  end
    if isfield(opts,'precision')  &&  strncmpi(opts.precision,'single',6)
        single_prec = true;
    end
else
    opts.regions = [1 2];
end
tic;

% Transfer initial data to the GPU
X_gpu = gpuArray(X);

% Get problem dimensions
dim     = size(X_gpu);
nt       = dim(1);
nx      = prod(dim(2:end));

% Get default spectral estimation parameters and options
[window,weight,nOvlp,dt,nDFT,nBlks] = parser(nt,nx,varargin{:});

% Determine correction for FFT window gain
winWeight   = 1/mean(window);

% Optimizers for x*Ax
if isfield(opts,'solver')
    switch opts.solver
        case {'MengiOverton','HeWatson','simpleIteration','eig'}
        otherwise
            error('Unknown solver.')
    end
else
    opts.solver = 'MengiOverton';
end

% Number of iterations to converge numerical radius
if ~isfield(opts,'nitmax')
    opts.nitmax = 500;
end

% Standard tolerance
if ~isfield(opts,'tol')
    opts.tol    = 1e-6;
end

% Use long-time mean if provided
if isfield(opts,'mean')
    x_mean      = opts.mean;
    mean_name   = 'provided long-time mean';    
else
    x_mean      = mean(X,1);
    mean_name   = 'data mean';
end
x_mean          = x_mean(:);
disp(['Mean                      : ' mean_name]);

% Obtain frequency axis
[f,nFreq,idx,f_idx,f1_idx,f2_idx,f3_idx] = faxes(nDFT,dt,opts);

nTriads = length(idx);

% Loop over number of blocks and generate Fourier realizations
disp(' ')
disp('Calculating temporal DFT on GPU')
disp('------------------------------------')

% Pre-allocate the DFT matrix on the GPU
Q_hat = zeros(nFreq,nx,nBlks, 'gpuArray');
window_gpu = gpuArray(window);
weight_gpu = gpuArray(weight);

for iBlk = 1:nBlks
    % Calculate time indices for the current block
    offset                  = min((iBlk-1)*(nDFT-nOvlp)+nDFT,nt)-nDFT;
    timeIdx                 = (1:nDFT) + offset;
    disp(['block ' num2str(iBlk) '/' num2str(nBlks) ' (' ...
        num2str(timeIdx(1)) ':' num2str(timeIdx(end)) ')'])
    
    % Assemble the current block and subtract the temporal mean
    % (use bsxfun for MATLAB versions prior to R2016b)
    Q_blk = X_gpu(timeIdx,:) - x_mean.';
    % Q_blk   = bsxfun(@minus,X(timeIdx,:),x_mean.');
    
    % Window and Fourier transform block
    Q_blk                   = Q_blk .* window_gpu;
    Q_blk_hat               = winWeight/nDFT*fft(Q_blk);
    Q_blk_hat               = fftshift(Q_blk_hat,1);
    
    Q_hat(:,:,iBlk)         = Q_blk_hat;
end
% Clear intermediate variables to free up VRAM
clear X X_gpu Q_blk Q_blk_hat x_mean window_gpu

% Calculate BMD across all frequency triads
disp(' ')
disp('Calculating BMD')
disp('------------------------------------')
L      = nan(nFreq,nFreq);
if nargout>4
    T      = nan(nFreq,nFreq);
end
P       = zeros(2,nTriads,nx);

% Permute Q_hat to ensure contiguous memory access
% [nFreq, nx, nBlks] -> [nx, nBlks, nFreq]
Q_hat = permute(Q_hat, [2, 3, 1]); 

if single_prec
    P       = single(P);
    Q_hat   = single(Q_hat);
end


% PHASE 1: Compute all B matrices in bulk on the GPU
disp('Phase 1: Computing Triad interactions on GPU...');
B_all_gpu = zeros(nBlks, nBlks, nTriads, 'gpuArray');

for i=1:nTriads
    Q_hat_f1 = Q_hat(:, :, f1_idx(i)); 
    Q_hat_f2 = Q_hat(:, :, f2_idx(i));
    Q_hat_f3 = Q_hat(:, :, f3_idx(i));
    
    Q12 = Q_hat_f1 .* Q_hat_f2;
    B_all_gpu(:,:,i) = Q_hat_f3' * (Q12 .* weight_gpu) / nBlks;
end

B_all_cpu = gather(B_all_gpu); 
clear B_all_gpu;    % Free VRAM after transferring B

% Extract variables to minimize overhead in parfor
calc_T              = (nargout > 4);      
solver_name         = opts.solver;        
tol                 = opts.tol;
nitmax              = opts.nitmax;


% PHASE 2: Solve eigenvalue problems concurrently on the CPU
% Progress Bar for Parallel Computing
D = parallel.pool.DataQueue;
processedTriads = 0;

% Calculate the update interval based on total steps
% (default : 20 steps ; interval : 100%/20 = 5%)
updateStep = max(1, ceil(nTriads / 20)); 

% Nested function (below) is triggered by signals from background workers
afterEach(D, @updateProgressBar);
    function updateProgressBar(~)
        processedTriads = processedTriads + 1;

        % Display only if the progress reaches the specified interval or 100%
        if mod(processedTriads, updateStep) == 0 || processedTriads == nTriads
            pct = (processedTriads / nTriads) * 100;

            % Length of the bar = n-steps
            numBars = floor(pct / 5); 
            numBars = min(20, max(0, numBars)); 
            barStr = [repmat('#', 1, numBars), repmat('-', 1, 20 - numBars)];

            fprintf(2,'Process: [%s] %3.0f%% (%d / %d)\n', barStr, pct, processedTriads, nTriads);
        end
    end

disp(' ')
disp('Phase 2: Solving problems on CPU...');
a_all_cpu = zeros(nBlks, nTriads);
r_all_cpu = zeros(nTriads, 1);

% Parallel computation of BMD using parfor 
parfor i = 1:nTriads
    % Initial variables to prevent warnings
    r = 0;
    a = zeros(nBlks, 1);

    B_cpu = B_all_cpu(:,:,i);

    % Optimizer for x*Ax
    switch solver_name
        case {'MengiOverton'}
            %  Mengi & Overton's algorithm
            [r, a] = MengiOverton(B_cpu, tol, nitmax);
        case {'HeWatson'}
            %  He & Watson's sophisticated iteration
            [r, a] = HeWatson(B_cpu, tol, nitmax);
        case {'simpleit'}
            %  Watson's simple iteration
            a     = rand(nBlks, 1) + 1i * rand(nBlks, 1); % random initial guess
            [r, a] = simpleIteration(B_cpu, a);
        otherwise
            error('Unknown solver.')
    end

    a_all_cpu(:, i) = a;
    r_all_cpu(i) = r;

    % Send a signal after each computation
    send(D, 1);
end

a_all_gpu = gpuArray(a_all_cpu);
% Free CPU RAM after transferring data to the GPU
clear B_all_cpu a_all_cpu;


% PHASE 3 : Reconstruct spatial modes on GPU in memory-safe batches
disp(' ')
disp('Phase 3: Reconstructing Spatial Modes on GPU with Batching...');
batch_size = 1000;
num_batches = ceil(nTriads / batch_size);

for b = 1:num_batches
    % Calculate the boundary indices for the current batch
    idx_start = (b-1)*batch_size + 1;
    idx_end   = min(b*batch_size, nTriads);
    curr_batch_size = idx_end - idx_start + 1;
    
    % Pre-allocate a localized, temporary GPU array for the current batch
    P_batch_gpu = zeros(2, curr_batch_size, nx, 'gpuArray');
    
    for bi = 1:curr_batch_size
        % Map the local batch index (bi) to the global triad index (i)
        i = idx_start + bi - 1;
        
        Q_hat_f1 = Q_hat(:, :, f1_idx(i)); 
        Q_hat_f2 = Q_hat(:, :, f2_idx(i));
        Q_hat_f3 = Q_hat(:, :, f3_idx(i));
        
        Q12 = Q_hat_f1 .* Q_hat_f2;
        a_gpu = a_all_gpu(:, i);
        
        % i+j component
        Psi1 = Q_hat_f3 * a_gpu;
        Psi1 = Psi1 / sqrt(Psi1' * (Psi1 .* weight_gpu)); % normalize by inner product
        
        % i*j component
        Psi2 = Q12 * a_gpu;
        Psi2 = Psi2 / sqrt(Psi2' * (Psi2 .* weight_gpu)); % normalize by inner product
        
        % Directly store the results into the GPU batch buffer
        P_batch_gpu(1, bi, :) = Psi1.';
        P_batch_gpu(2, bi, :) = Psi2.';
        
        L(f1_idx(i), f2_idx(i)) = r_all_cpu(i);
        
        % Energy transfer term
        if calc_T
            T(f1_idx(i), f2_idx(i)) = gather(real((Q_hat_f3 * a_gpu)' * (Q12 * a_gpu)) / nBlks);
        end
    end
    
    % Offload the completed batch from GPU (VRAM) to CPU (RAM)
    P(:, idx_start:idx_end, :) = gather(P_batch_gpu);
end

% Reshape the reconstructed modes to match the original spatial dimensions
P   = reshape(P,[2 nTriads dim(2:end) 1]);
toc;
evalc('delete(gcp(''nocreate''))');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [window,weight,nOvlp,dt,nDFT,nBlks] = parser(nt,nx,varargin)
% PARSER Parser for BMD parameters

% read input arguments from cell array
window = []; weight = []; nOvlp = []; dt = [];
nvarargin = length(varargin);

if nvarargin >= 1
    window = varargin{1};
    if nvarargin >= 2
        weight   = varargin{2};
        if nvarargin >= 3
            nOvlp   = varargin{3};
            if nvarargin >= 4
                dt      = varargin{4};
            end
        end
    end
end

window = window(:); weight = weight(:);

% check arguments and determine default spectral estimation parameters
% window size and type
if isempty(window)
    nDFT        = 2^floor(log2(nt/5)); if nDFT>256, nDFT=256; end
    window      = hammwin(nDFT);
    window_name = 'Hamming';
elseif length(window)==1
    nDFT        = window;
    window      = hammwin(window);
    window_name = 'Hamming';
elseif length(window) == 2^nextpow2(length(window))
    nDFT        = length(window);
    window_name = 'user specified';
else
    nDFT        = length(window);
    window_name = 'user specified';
end

% block overlap
if isempty(nOvlp)
    nOvlp = floor(nDFT/2);
elseif nOvlp > nDFT-1
    error('Overlap too large.')
end

% time step between consecutive snapshots
if isempty(dt)
    dt = 1/nDFT;
end

% inner product weight
if isempty(weight)
    weight      = ones(nx,1);
    weight_name = 'uniform';
elseif numel(weight) ~= nx
    error('Weights must have the same spatial dimensions as data.');
else
    weight_name = 'user specified';
end

% number of blocks
nBlks = floor((nt-nOvlp)/(nDFT-nOvlp));

% test feasibility
if nDFT < 4 || nBlks < 2
    error('Spectral estimation parameters not meaningful.');
end

% display parameter summary
disp(' ')
disp('BMD parameters')
disp('------------------------------------')
disp(['No. of snaphots per block : ' num2str(nDFT)])
disp(['Block overlap             : ' num2str(nOvlp)])
disp(['No. of blocks             : ' num2str(nBlks)])
disp(['Windowing fct. (time)     : ' window_name])
disp(['Weighting fct. (space)    : ' weight_name])
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [window] = hammwin(N)
% HAMMWIN standard Hamming window of lenght N
window = 0.54-0.46*cos(2*pi*(0:N-1)/(N-1))';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w,z] = simpleIteration(A,z_0,tol)
% SIMPLEITERATION simple power iteration from He & Watson that is not
% guaranteed to find global optimum; small tolerences proposed in original
% paper tends to prevent convergence. 1e-8 works in most cases.
z       = z_0/sqrt(z_0'*z_0);
w       = Inf;
w_err   = Inf;
it_max  = 100;
it      = 0;
% tol   = 10*length(A)*eps*norm(A,1);
while w_err > tol
    
    w_old   = w;
    w       = z'*A*z;
    w_err   = abs(w-w_old);
    z       = w*A'*z + w'*A*z;
    z       = z/sqrt(z'*z);
    
    it      = it+1;
    if it>it_max, break, end
end
w       = z'*A*z;
% disp(['Watson's simple iteration required ' num2str(it) ' iterations.']);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w,z] = HeWatson(A,tol,nitmax)
% HEWATSON 'An Algorithm' from He & Watson (1997) that is guranteed to find
% the global optimum upon convergence
N       = size(A,1);
normA   = norm(A,1);
z       = rand(N,1) + 1i*rand(N,1);
lb      = 0;
ub      = normA;
Z       = zeros(N);
I       = eye(N);
S       = [A Z; Z I];

it  = 0;
while (ub-lb)>tol || it==0
    it = it + 1;
    
    [w,z]   = simpleIteration(A,z,tol);
    lb      = max(lb,abs(w));
    alpha   = lb + tol;
    R       = [2*alpha*I -A'; I Z];
    [V,D]   = eig(R,S);
    D       = diag(D);
    
    ucirc   = abs(abs(D)-1) < (sqrt(eps)*normA);
    if sum(ucirc)==0
        break
    elseif mod(it,100)==0
        disp(['He & Watson algorithm did not converge in ' num2str(it) ' iterations! Trying new initial guess...']);
        z       = rand(N,1) + 1i*rand(N,1);
    elseif it>=nitmax
        disp(['He & Watson algorithm did not converge in ' num2str(it) ' iterations!']);
        break
    else
        idx = find(ucirc==1);
        z   = V(end-N+1:end,idx(1));
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w,z] = MengiOverton(A,tol,nitmax)
% MENGIOVERTON level-set algorithm from Mengi & Overton (2005) that is
% globally convergent
%
% B. Yeung (byeung@ucsd.edu)
% Last revision: 2023-08-16

N       = size(A,1);
normA   = norm(A,1);
Z       = zeros(N);
I       = eye(N);
S       = [A Z; Z I];

it = 0;
phi = 0;
while ~isempty(phi)
    w_temp = maxFOV(A,phi);
    [w,idx] = max(w_temp);
    phi_max = phi(idx);
    
    w = w*(1+tol);
    R       = [2*w*I -A'; I Z];
    [~,D]   = eig(R,S,'vector');
    isunimod   = abs(abs(D)-1) <= (sqrt(eps)*normA);
    Dunimod = D(isunimod);
    thetaprime = angle(Dunimod);
    theta = [];
    for i = 1:length(thetaprime)
        if abs(maxFOV(A,thetaprime(i))-w)<=sqrt(eps)*w
            theta = [theta; thetaprime(i)];
        end
    end
    theta = unique(theta);
    phi = [];
    for i=1:length(theta)
        lb = theta(i);
        if i<length(theta)
            ub = theta(i+1);
            mid = (lb+ub)/2;
        else
            ub = theta(1);
            mid = mod((lb+ub+2*pi)/2,2*pi);
        end
        if maxFOV(A,mid)>w
            phi = [phi; mid];
        end
    end
    it = it+1;
    if it>=nitmax
        disp(['Mengi & Overton algorithm did not converge in ' num2str(it) ' iterations!']);
        break
    end
end
B = A*exp(1i*phi_max);
H = (B+B')/2;
[V,D] = eig(H,'vector');
[~,idx] = max(abs(D));
z = V(:,idx);

% reconstruct complex w
w = z'*A*z;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function lmax = maxFOV(A,theta)
% MAXFOV maximum field of value of matrix A at angle theta
ntheta = length(theta);
lmax = zeros(ntheta,1);
for i = 1:ntheta
    A_rot = A*exp(1i*theta(i));
    H = 0.5*(A_rot+A_rot');
    lmax(i) = max(abs(eig(H)));
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f,nFreq,idx,f_idx,f1_idx,f2_idx,f3_idx] = faxes(nDFT,dt,opts)
% FAXES obtain frequency axes and indices
f_idx = (0:nDFT-1);
if mod(nDFT,2)==0
    f_idx(nDFT/2+1:end)     = f_idx(nDFT/2+1:end)-nDFT;
else
    f_idx((nDFT+1)/2+1:end) = f_idx((nDFT+1)/2+1:end)-nDFT;
end
f_idx   = fftshift(f_idx);
f       = f_idx/dt/nDFT;
fNyq_idx= -f_idx(1);
nFreq   = numel(f_idx);
if isfield(opts,'nfreq')
    f_idx_max   = opts.nfreq;
else
    f_idx_max   = fNyq_idx;
end

region  = nan(nFreq,nFreq);
idx     = nan(nFreq^2,1);
f1_idx  = idx;
f2_idx  = idx;
f3_idx  = idx;
count   = 0;
for i=1:nFreq
    for j=1:nFreq
        f1plus2  = f_idx(i)+f_idx(j);
        if abs(f1plus2)<fNyq_idx && abs(f_idx(i))<=f_idx_max && abs(f_idx(j))<=f_idx_max
            if sum(opts.regions==1)>0 && f_idx(i)>=0 && f_idx(j)>=0 && f_idx(i)>=f_idx(j)            % region #1
                region(i,j)     = 1;
            end
            if sum(opts.regions==2)>0 && f_idx(i)>=0 && f_idx(j)<=0 && f_idx(i)>=abs(f_idx(j))       % region #2
                region(i,j)     = 2;
            end
            if sum(opts.regions==3)>0 && f_idx(i)>=0 && f_idx(j)<=0 && f_idx(i)<=abs(f_idx(j))       % region #3
                region(i,j)     = 3;
            end
            if sum(opts.regions==4)>0 && f_idx(i)<=0 && f_idx(j)<=0 && f_idx(i)>=(f_idx(j))          % region #4
                region(i,j)     = 4;
            end
            if sum(opts.regions==5)>0 && f_idx(i)<=0 && f_idx(j)<=0 && f_idx(i)<=f_idx(j)            % region #5
                region(i,j)     = 5;
            end
            if sum(opts.regions==6)>0 && f_idx(i)<=0 && f_idx(j)>=0 && abs(f_idx(i))>=f_idx(j)       % region #6
                region(i,j)     = 6;
            end
            if sum(opts.regions==7)>0 && f_idx(i)<=0 && f_idx(j)>=0 && abs(f_idx(i))<=f_idx(j)       % region #7
                region(i,j)     = 7;
            end
            if sum(opts.regions==8)>0 && f_idx(i)>=0 && f_idx(j)>=0 && f_idx(i)<=f_idx(j)            % region #8
                region(i,j)     = 8;
            end
        end
        
        if ~isnan(region(i,j))
            count           = count + 1;
            idx(count)      = sub2ind([nFreq nFreq],i,j);
            f1_idx(count)   = i;
            f2_idx(count)   = j;
            f3_idx(count)   = find(f_idx==f1plus2);
        end
    end
end
f1_idx  = f1_idx(1:count);
f2_idx  = f2_idx(1:count);
f3_idx  = f3_idx(1:count);
idx     = idx(1:count);

end