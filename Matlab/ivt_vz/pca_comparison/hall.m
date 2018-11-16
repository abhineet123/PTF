function [UZ, DZ, muZ, nZ] = hall(data, UX, DX, muX, nX, K)
% [U, D, mu, n] = hall(data, UX, DX, muX, nX, K)
% [U, D, mu, n] = hall(data)
% [U, D, mu, n] = hall(data, [], [], [], [], K)
%
% This function implements an incremental approach to PCA, using the 
% method described in "Adding and subtracting eigenspaces with eigenvalue 
% decomposition and singular value decomposition", by Peter Hall, 
% David Marshall, Ralph Martin, Image and Vision Computing 20 (2002).

if nargin ~= 1 && nargin ~= 6
    error('function must have either 1 or 6 arguments');
end

% make sure DX is a diagonal matrix
if nargin >= 3 && size(DX,1) ~= size(DX,2)
    DX = diag(DX);
end

% compute svd of new points
nY = size(data,2);
muY = mean(data,2);
data = data - repmat(muY,[1,nY]);
[UY,DY,VY] = svd(data,0);
DY = DY.^2/nY; % added by Ruei-Sung

% quit if we're only given 'data'
if nargin == 1 || isempty(UX)
    if nargin < 6 || isempty(K)
        K = size(data,2);
    end
    UZ = UY;
    DZ = diag(DY);
    muZ = muY;
    nZ = nY;
    % keep only the top K eigenvectors
    UZ = UZ(:,1:min(K,end));
    DZ = DZ(1:min(K,end));
    return
end

% compute new mean and count
nZ = nX + nY;
muZ = (nX*muX + nY*muY) / nZ;

% do equations 9-15
g = UX'*(muX - muY);
G = UX'*UY;
H = UY - UX*G;
h = (muX - muY) - UX*g;
% v = orth([H h]);
[v junk] = qr([H h], 0); % using QR to orthogonalize is faster
Gamma = v'*UY;
gamma = v'*(muX - muY);

% form the matrix in equation 8
p = size(UX,2);
t = size(v,2);
matrix = (nX/nZ) * [DX zeros(p,t); zeros(t,t+p)] ...
    + (nY/nZ) * [G*DY*G' G*DY*Gamma'; Gamma*DY*G' Gamma*DY*Gamma'] ...
    + (nX*nY/(nZ^2)) * [g*g' g*gamma'; gamma*g' gamma*gamma'];

% [R_,DZ_,VZ_] = svd(matrix,0); 
[R,DZ] = eig(matrix);
[DZ order] = sort(diag(DZ),'descend');
R = R(:,order);

% find R
% use the criterion from Levy & Lindenbaum
% cutoff = sum(DZ.^2) * 1e-6;
% DZ = DZ(DZ.^2 >= cutoff);
% UZ = [UX v] * R(:, 1:length(DZ));

% keep all vectors
% UZ = [UX v]*R;

% keep only the top K
keep = 1:min([K,length(DZ),nZ]);
DZ = DZ(keep);
UZ = [UX v]*R(:, keep);

