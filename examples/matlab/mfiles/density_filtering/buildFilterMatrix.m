function M = buildFilterMatrix(nodes, triangles, kernelFunction, periodicFunction)
% buildFilterMatrix Construct a normalized spatial filtering matrix on a triangular mesh.
%
%   M = buildFilterMatrix(nodes, triangles, kernelFunction)
%   M = buildFilterMatrix(nodes, triangles, kernelFunction, periodicFunction)
%
%   Constructs a sparse filtering (or convolution) matrix acting on
%   element-wise quantities defined over a 2D triangular mesh. The filter
%   weights are computed from a user-defined kernel function evaluated at
%   the barycenters of the mesh elements.
%
%   The function optionally accounts for periodicity by extending the
%   domain with virtual copies of the mesh shifted according to a user-
%   supplied periodic mapping.
%
%   Inputs:
%       nodes : [Nn x 2] array of nodal coordinates. Only the first two
%               columns (x,y) are used.
%
%       triangles : [Ne x 3] array of triangle connectivity. Only the first
%                   three columns (node indices) are used.
%
%       kernelFunction : Function handle defining the spatial filter kernel.
%               It must accept two matrices (Δx, Δy) and return a matrix
%               of weights:
%                   K = kernelFunction(Δx, Δy)
%
%       periodicFunction : (optional) Function handle defining periodic
%               extensions of the nodal coordinates. It must have the form:
%                   nodesShifted = periodicFunction(nodes, shiftID)
%               where shiftID = ±1 typically corresponds to upper/lower
%               periodic images.
%
%   Output:
%       M : [Ne x 3Ne] sparse, row-normalized filter matrix. Each row
%           contains the weights applied to:
%               - the original mesh elements,
%               - the periodic image above,
%               - the periodic image below.
%
%   Method:
%       - Triangle barycenters are computed for both reference and test meshes.
%       - The kernel function is evaluated pairwise between all barycenters.
%       - Periodic contributions are added if requested.
%       - Each row of the matrix is normalized so that its coefficients sum
%         to one.
%
%   Notes:
%       - The resulting matrix acts on element-wise fields.
%       - The kernel evaluation is fully vectorized but may require
%         significant memory for large meshes.
%       - If no periodicFunction is provided, only the interior contribution
%         is used.

nodes = nodes(:,1:2);
triangles = triangles(:,1:3);
M_inside = buildFilterMatrix_(nodes, triangles, kernelFunction, nodes);

if nargin >=4 && ~ isempty(periodicFunction)
    nodesAbove = periodicFunction(nodes,1);
    nodesBelow = periodicFunction(nodes,-1);
    M_above = buildFilterMatrix_(nodes,triangles, kernelFunction, nodesAbove);
    M_below = buildFilterMatrix_(nodes,triangles, kernelFunction, nodesBelow);
else
    M_above = sparse(size(triangles,1),size(triangles,1));
    M_below = M_above;
end

M = [M_inside, M_above, M_below];

% normalization
M = M./sum(M,2);
end




function M = buildFilterMatrix_(nodesRef, triangles, kernelFunction, nodesTest)

% barycenters of the triangles (evaluation points of the kernel function)
xRef = nodesRef(:,1); xRef0 = mean(xRef(triangles),2);
yRef = nodesRef(:,2); yRef0 = mean(yRef(triangles),2);

xTest = nodesTest(:,1); xTest0 = mean(xTest(triangles),2);
yTest = nodesTest(:,2); yTest0 = mean(yTest(triangles),2);

% need quite some memory, but fast...
M = sparse(kernelFunction(xTest0.'-xRef0, yTest0.'-yRef0));
end
