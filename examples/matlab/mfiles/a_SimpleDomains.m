%% Simple domains examples

clear ; close ; path=pwd();
[s,e]=regexp(path,'.+(?=[\\/]{1,2}examples)');
path=path(s:e); addpath(genpath(path)); clear s e path
%% 1) 0D domain

domain0D = Domain(1); % a single dot
figure()
domain0D.plot()
%% 1) 1D domain

domain1D = Domain(2); % a 1D line segment has 2 vertices
figure()
domain1D.plot()
%% 2) 2D domain

nVertices2D = 5;
domain2D = Domain(nVertices2D);  % a regular polygon
figure()
domain2D.plot()
%% 3) 3D domain

name3D = "diamond8"; % (supported: tetra, cube, diamond, prism, pyramid)
domain3D = Domain(name3D);
figure()
domain3D.plot()