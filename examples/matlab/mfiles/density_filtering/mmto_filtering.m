%% Multi-material density filtering with periodicity

% See this paper for some contextualization :  10.1108/COMPEL-10-2023-0546
% (https://hal.science/hal-04634402v1)

clear; close all

%% 1) Setup the problem 

%% Define a material domain

% domain
myMaterialDomain = Domain("pyramid4");

%% Define the material coordinates

% let's consider air and 4 magnets orientations inside the domain
% - mag1 -> magnetization toward the right
% - mag2 -> magnetization toward the top
% - mag3 -> magnetization toward the left
% - mag4 -> magnetization toward the bottom

materialCoordinates = dictionary();

materialCoordinates("magRight") =  VertexFunction(@(a) myMaterialDomain.Vertices(1,:).',...
                     @(a) 0, "magRight", 0, myMaterialDomain.Dimension);

materialCoordinates("magTop")= VertexFunction(@(a) myMaterialDomain.Vertices(2,:).'.',...
                              @(a) 0, "magTop", 0, myMaterialDomain.Dimension);

materialCoordinates("magLeft") = VertexFunction(@(a) myMaterialDomain.Vertices(3,:).',...
                                @(a) 0 , "magLeft", 0, myMaterialDomain.Dimension);

materialCoordinates("magBot") = VertexFunction(@(a) myMaterialDomain.Vertices(4,:).',...
                     @(a) 0, "magBot", 0, myMaterialDomain.Dimension);

materialCoordinates("air")= VertexFunction(@(a) myMaterialDomain.Vertices(5,:).',...
                           @(a) 0 , "air", 0, myMaterialDomain.Dimension);

% material placement (in the vertex order)
vertex2material_inside = ["magRight", "magTop", "magLeft", "magBot", "air"];

% draw the domain
figure()
myMaterialDomain.plot(vertex2material_inside,1);
title("Material domain")
%% Define the material properties

%% Colors
% we now define our vertexFunction as the centered RGB color of the material,
% in order to visualize the material distribution

colorVertexFunctions = dictionary();

colorVertexFunctions("magRight") = VertexFunction(@(a) legend2color("magRight").',...
                               @(a) 0, "magRight", 0, 3);

colorVertexFunctions("magTop") = VertexFunction(@(a) legend2color("magTop").',...
                                @(a) 0 , "magTop", 0, 3);

colorVertexFunctions("magLeft") = VertexFunction(@(a) legend2color("magLeft").',...
                     @(a) 0, "magLeft", 0, 3);

colorVertexFunctions("magBot") = VertexFunction(@(a) legend2color("magBot").',...
                     @(a) 0, "magBot", 0, 3);

colorVertexFunctions("air") = VertexFunction(@(a) legend2color("air").',...
                            @(a) 0 , "air", 0, 3);

materialColor_inside.Domain = myMaterialDomain;
materialColor_inside.Children = colorVertexFunctions(vertex2material_inside);
materialColor_inside.Label = "color";

materialColor_inside = Interpolation(materialColor_inside);

%% Arrows (magnetization)

arrowsVertexFunctions = dictionary();

arrowsVertexFunctions("magRight") = VertexFunction(@(a) [1;0],...
                               @(a) 0, "magRight", 0, 2);

arrowsVertexFunctions("magTop") = VertexFunction(@(a) [0;1],...
                                @(a) 0 , "magTop", 0, 2);

arrowsVertexFunctions("magLeft") = VertexFunction(@(a) [-1;0],...
                     @(a) 0, "magLeft", 0, 2);

arrowsVertexFunctions("magBot") = VertexFunction(@(a) [0;-1],...
                     @(a) 0, "magBot", 0, 2);

arrowsVertexFunctions("air") = VertexFunction(@(a) [0;0] ,...
                            @(a) 0 , "air", 0, 2);

materialArrow_inside.Domain = myMaterialDomain;
materialArrow_inside.Children = arrowsVertexFunctions(vertex2material_inside);
materialArrow_inside.Label = "arrows";

materialArrow_inside = Interpolation(materialArrow_inside);

%% Define a mesh with anti-periodicity
run("mesh_mmto_filter.m")

% define the regions of the mesh
air_zone = msh.TRIANGLES(:,4)==1; air_vertex_inside = find(vertex2material_inside == "air");
magRight_zone = msh.TRIANGLES(:,4)==2 ; magRight_vertex_inside = find(vertex2material_inside == "magRight");

%% Define the reference material distribution

N_triangles = size(msh.TRIANGLES,1);
rho_ref = materialColor_inside.initializeVariable(N_triangles,"zero");
rho_ref.color(air_zone,:) = myMaterialDomain.Vertices(air_vertex_inside,:).*ones(sum(air_zone),1);
rho_ref.color(magRight_zone,:) = myMaterialDomain.Vertices(magRight_vertex_inside,:).*ones(sum(magRight_zone),1);
rho_ref = materialColor_inside.projection(rho_ref); % necessary to avoid internal divisions by 0.

plotColor = @(rho) patch('Faces',msh.TRIANGLES(:,1:3), ...
                       'Vertices',msh.POS,'FaceColor', "flat", "edgecolor",[0.5,0.5,0.5],...
                       "CData", permute(materialColor_inside.eval(rho,zeros(1,1,N_triangles)),[2,3,1])+0.5);



figure()
plotColor(rho_ref); hold on 
plotArrows(rho_ref, materialArrow_inside, msh); hold off 
axis equal
title("Reference material distribution")

%% 2) Define the filter
radius = 0.3;  % radius of the filter kernel
kernelFunction = @(x,y) max(0, radius - sqrt(x.^2 + y.^2));
%kernelFunction = @(x,y) max(0, (radius - sqrt(x.^2 + y.^2))>0);

% function to plot the filter kernel
triangleCenter = 50;
plotFilter = @(M) patch('Faces',msh.TRIANGLES(:,1:3), 'Vertices',msh.POS,'FaceColor', "flat", "edgecolor",[0.5,0.5,0.5],...
                       "CData", M(triangleCenter,1:end/3)+M(triangleCenter,(end/3+1):(2*end/3))+M(triangleCenter,(2*end/3+1:end)));

%% a) Cropped filter kernel
Mcropped = buildFilterMatrix(msh.POS,msh.TRIANGLES,kernelFunction);
figure(); subplot(1,2,1)
plotFilter(Mcropped);  axis equal
title("Cropped kernel")

%% b) Wrapped filter kernel
% function to define the geometric (anti)periodicity (here : rotation)
wrapFunction = @(xy,coeff) xy * [cos(coeff*pi/2), sin(coeff*pi/2); ...
                                -sin(coeff*pi/2), cos(coeff*pi/2)]; % +pi/2 rotation
Mwrapped = buildFilterMatrix(msh.POS,msh.TRIANGLES,kernelFunction,wrapFunction);
subplot(1,2,2)
plotFilter(Mwrapped); axis equal
title("Wrapped kernel")

%% 3) Naive filter application
% function to apply the filter
naiveFilter = @(M, rho) setfield(rho, materialColor_inside.Label, (M * repmat(rho.(materialColor_inside.Label),3,1)));

%% cropped kernel
rhoCropped=naiveFilter(Mcropped, rho_ref);
figure(); subplot(1,2,1)
plotColor(rhoCropped);  hold on 
plotArrows(rhoCropped, materialArrow_inside, msh); hold off 
axis equal
title("Cropped filter (naive)")

% works fine with a cropped kernel
% however, we don't see any influence of the material on the other side.
% the (anti)-periodicity is not respected !

%% wrapped kernel
subplot(1,2,2)
rhoWrapped = naiveFilter(Mwrapped, rho_ref);
plotColor(rhoWrapped); hold on 
plotArrows(rhoWrapped, materialArrow_inside, msh); hold off 
axis equal
title("Wrapped filter (naive)")

% here, we can see the influence of the material on the other side.
% however, it's the WRONG ONE! since the material is a magnet, its
% magnetization should also be rotated. We expect the associated dark
% green color.

%% 4) Correct filter application
% We should consider some transformations on the materials.
% There are two transformations: one for the materials located above and
% another for the material located below.
% One should then define two permutations that models the material change
% after a positive or negative rotation.
% We can implement this by defining two interpolations  with different
% material placements


%% a) Definition of the "above" interpolation (periodicity)
% If the transformed material are included in the material set, then the
% transformation is simply a permutation of the materials over the vertices
% of the material domain.

%vertex2material_inside =         ["magRight", "magTop" , "magLeft", "magBot"  , "air"];
vertex2material_abovePeriodic =   ["magTop"  , "magLeft", "magBot" , "magRight", "air"];

abovePeriodic.Domain = myMaterialDomain;
abovePeriodic.Children = materialCoordinates(vertex2material_abovePeriodic);
abovePeriodic.Label = "material_vertices_above";
abovePeriodic = Interpolation(abovePeriodic);

%% b) Definition of the "below" interpolation (periodicity)
%vertex2material_inside =         ["magRight", "magTop" , "magLeft", "magBot"  , "air"];
vertex2material_belowPeriodic =   ["magBot"  , "magRight", "magTop" , "magLeft", "air"];

belowPeriodic.Domain = myMaterialDomain;
belowPeriodic.Label = "material_vertices_below";
belowPeriodic.Children = materialCoordinates(vertex2material_belowPeriodic);
belowPeriodic = Interpolation(belowPeriodic);

%% Application of the wrapped filter
% add "above" and "below" components to rho, that will be transformed
% To do it inline we should 
rho_ref.(abovePeriodic.Label) = rho_ref.(materialColor_inside.Label);
rho_ref.(belowPeriodic.Label) = rho_ref.(materialColor_inside.Label);

rhoAbove = @(rho) setfield(rho, abovePeriodic.Label, rho.(materialColor_inside.Label));
rhoBelow = @(rho) setfield(rho, belowPeriodic.Label, rho.(materialColor_inside.Label));

rhoAbovePeriodic = @(rho) permute(abovePeriodic.eval(rhoAbove(rho),zeros(1,1,N_triangles)), [3,1,2]);

rhoBelowPeriodic = @(rho) permute(belowPeriodic.eval(rhoBelow(rho),zeros(1,1,N_triangles)), [3,1,2]);

% then the filter reads

filterPeriodic = @(M, rho) setfield(rho, materialColor_inside.Label, ...
    M * [rho.(materialColor_inside.Label);  rhoAbovePeriodic(rho); rhoBelowPeriodic(rho)] );


%% wrapped kernel
figure()
wrappedPeriodicRho = filterPeriodic(Mwrapped, rho_ref);
plotColor(wrappedPeriodicRho);   hold on 
plotArrows(wrappedPeriodicRho, materialArrow_inside, msh); hold off 
axis equal
title("Wrapped filter (periodic)")


%% Anti-periodicity
% We can do the same for anti-periodic boundary conditions. We just have to
% adapt the "above" and "below" transformations.

%% a) Definition of the "above" interpolation (antoperiodicity)

%vertex2material_inside =              ["magRight", "magTop" , "magLeft", "magBot"  , "air"];
vertex2material_aboveAntiPeriodic =    ["magBot"  , "magRight", "magTop" , "magLeft", "air"];

aboveAntiPeriodic.Domain = myMaterialDomain;
aboveAntiPeriodic.Children = materialCoordinates(vertex2material_aboveAntiPeriodic);
aboveAntiPeriodic.Label = "material_vertices_above";
aboveAntiPeriodic = Interpolation(aboveAntiPeriodic);

%% b) Definition of the "below" interpolation (periodicity)
%vertex2material_inside =             ["magRight", "magTop" , "magLeft", "magBot"  , "air"];
vertex2material_belowAntiPeriodic =   ["magTop"  , "magLeft", "magBot" , "magRight", "air"];

belowAntiPeriodic.Domain = myMaterialDomain;
belowAntiPeriodic.Label = "material_vertices_below";
belowAntiPeriodic.Children = materialCoordinates(vertex2material_belowAntiPeriodic);
belowAntiPeriodic = Interpolation(belowAntiPeriodic);

%% Application of the wrapped filter
% add "above" and "below" components to rho, that will be transformed
% To do it inline we should 
rho_ref.(aboveAntiPeriodic.Label) = rho_ref.(materialColor_inside.Label);
rho_ref.(belowAntiPeriodic.Label) = rho_ref.(materialColor_inside.Label);

rhoAbove = @(rho) setfield(rho, aboveAntiPeriodic.Label, rho.(materialColor_inside.Label));
rhoBelow = @(rho) setfield(rho, belowAntiPeriodic.Label, rho.(materialColor_inside.Label));

rhoAbovePeriodic = @(rho) permute(aboveAntiPeriodic.eval(rhoAbove(rho),zeros(1,1,N_triangles)), [3,1,2]);

rhoBelowPeriodic = @(rho) permute(belowAntiPeriodic.eval(rhoBelow(rho),zeros(1,1,N_triangles)), [3,1,2]);

% then the filter reads

filterAntiPeriodic = @(M, rho) setfield(rho, materialColor_inside.Label, ...
    M * [rho.(materialColor_inside.Label);  rhoAbovePeriodic(rho); rhoBelowPeriodic(rho)] );


%% wrapped kernel
figure()
wrappedAntiPeriodicRho = filterAntiPeriodic(Mwrapped, rho_ref);
plotColor(wrappedAntiPeriodicRho);  hold on 
plotArrows(wrappedAntiPeriodicRho, materialArrow_inside, msh); hold off 
axis equal


%% Functions

function plotArrows(rho, interpArrows, msh)
x = msh.POS(:,1); x0 = mean(x(msh.TRIANGLES(:,1:3)),2);
y = msh.POS(:,2); y0 = mean(y(msh.TRIANGLES(:,1:3)),2);
rho.(interpArrows.Label) = rho.color;
uv = permute(interpArrows.eval(rho,zeros(1,1,size(msh.TRIANGLES,1))),[3,1,2]);
quiver(x0,y0,uv(:,1),uv(:,2),3);
end