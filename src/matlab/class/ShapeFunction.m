classdef ShapeFunction
    %ShapeFunction Class
    %
    % To create a ShapeFunction object, type
    % obj = ShapeFunction(type,domain,label)
    %
    % where 'type' is a string indicating the type of shape function.
    % Default = "wachspress" (only available type currently), based on [1]
    % Other types (to implement) could be "mean-value" or others, see [2].
    %
    % ShapeFunctions are the weigths in the interpolation of the
    % VertexFunctions, depending on x.
    %
    % [1] Floater, M., Gillette, A., & Sukumar, N. (2014). Gradient bounds
    % for Wachspress coordinates on polytopes. SIAM Journal on Numerical
    % Analysis, 52(1), 515–532. https://doi.org/10.1137/130925712.
    %
    %
    % [2] Floater, M. S. (2014). Wachspress and mean value coordinates.
    % Springer Proceedings in Mathematics and Statistics, 83(1), 81–102.
    % https://doi.org/10.1007/978-3-319-06404-8_6
    %
    %Copyright (C) 2024 Théodore CHERRIERE (theodore.cherriere@ricam.oeaw.ac.at.fr)
    %This program is free software: you can redistribute it and/or modify
    %it under the terms of the GNU General Public License as published by
    %the Free Software Foundation, either version 3 of the License, or
    %any later version.
    %
    %This program is distributed in the hope that it will be useful,
    %but WITHOUT ANY WARRANTY; without even the implied warranty of
    %MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    %GNU General Public License for more details.
    %
    %You should have received a copy of the GNU General Public License
    %along with this program.  If not, see <https://www.gnu.org/licenses/>.

    properties
        Type
        Domain
        Reference
        Label=""
    end

    methods
        function obj = ShapeFunction(type,domain,label)
            %obj = ShapeFunction(type,domain,label)
            %
            % type is a string which indicates the nature of the basis
            % functions. Available :
            %  type = "w" or "wachspress" = Wachspress basis functions [1,2]
            %  Other types are theoretically possible but are not
            %  implemented yet (feel free to contribute !)
            %
            % domain is an instance of the Domain class.
            % 
            % label is facultative string identifier (defaut : "").
            %
            %
            % [1] Wachspress, E. L. (1975). A Finite Element Rational Basis.
            % Academic Press, Inc. https://doi.org/10.1016/s0076-5392(09)60113-2
            %
            %  [2] Floater, M., Gillette, A., & Sukumar, N. (2014). Gradient bounds
            % for Wachspress coordinates on polytopes. SIAM Journal on Numerical
            % Analysis, 52(1), 515–532. https://doi.org/10.1137/130925712.

            obj.Domain = domain;
            if nargin>=3; obj.Label = label; end

            if lower(type)=="w" || lower(type)=="wachspress"
                obj.Type = "wachspress";
                obj.Reference = "Wachspress, 1975 ; Floater et. al. 2014";
            else
                error(strcat("Basis function type ",type, " not supported"))
            end
        end

        function [w,dwdx] = eval(obj,x)
            %[w,dwdx] = eval(obj,x)
            % evaluate the shape functions w and their gradient dwdx for
            % a given position x inside the obj.Domain.

            if obj.Type=="wachspress"
                [n,dim]=size(x);
                assert(dim==0 || dim==obj.Domain.Dimension,"bad density dimension");

                if dim==0
                    w=ones(1,1,n); dwdx=zeros(1,0,n);
                elseif dim==1
                    x=shiftdim(x,-2);
                    w=[-x;x]+0.5; % assuming centering on 0;
                    dwdx=repmat([-1;1],[1,1,n]);
                elseif dim==2
                    [w,dwdx]= wachspress2dVectorized(x,obj.Domain);
                elseif dim==3
                    [w,dwdx]= wachspress3dVectorized(x,obj.Domain);
                else
                    error("incompatible density field")
                    % if you want to add other types of basis functions,
                    % follow the same model than the Wachspress' one.
                    % Please vectorize the code for maximizing the
                    % performance (avoid for loop)
                end
            else
                error(strcat("Basis function type ",obj.Type, " not supported"))
            end
        end

        function plot(obj,nVertex)
            % plot(obj,nVertex)
            %
            % Plot the shape function associated with the specified vertex
            % (defaut : nVertex = 1)

            if nargin<=1 || isempty(nVertex); nVertex=1; end

            vCenter=obj.Domain.Vertices-mean(obj.Domain.Vertices,1);
            if size(vCenter,2)==1 && size(vCenter,1)==2 % 1D = 2 vertices
                x=linspace(vCenter(1),vCenter(2),100).';
                w=obj.eval(x);
                w=w(nVertex,:,:);
                plot(x,w(:));
                xlabel('x');
                ylabel('\omega(x)');
                for i=1:size(vCenter,1)
                    text(vCenter(i)*1.15,0,strcat("v",num2str(i)))
                end

            elseif size(vCenter,2)==2 %2D
                pgon=polyshape(vCenter(:,1)*0.99999,vCenter(:,2)*0.99999);
                m = max(abs(vCenter(:)));
                [x,y]=meshgrid(linspace(-m,m,200),linspace(-m,m,200));
                in=pgon.isinterior(x(:),y(:));
                tr = delaunayTriangulation(x(in),y(in));
                w=obj.eval([x(in),y(in)]);
                c=permute(w(nVertex,:,:),[3 2 1]);
                trisurf(tr.ConnectivityList,tr.Points(:,1),tr.Points(:,2),c)
                colorbar off; maxval=max(tr.Points(:));
                axis([-maxval maxval -maxval maxval]*1.5);axis equal
                hold on; plot(pgon,'facealpha',0); grid on;
                shading interp
                for i=1:size(vCenter,1)
                    text(vCenter(i,1)*1.15,vCenter(i,2)*1.2,strcat("v",num2str(i)))
                end
                axis off
                hold off;
                light('Position',[-50 -15 10])
                material([0.8 0.5 0.3])
                lighting flat
                view(45,30)
                colormap(jet)
                caxis([0,1]);
                colorbar
            elseif   size(vCenter,2)==3 %3D

                shp=alphaShape(vCenter(:,1),vCenter(:,2),vCenter(:,3),Inf);
                [BF, P] = boundaryFacets(shp);
                [x,y]=meshgrid(0:0.01:1,0:0.01:1);
                tri1=delaunay(x,y);
                tri=[];
                nodes=[];
                for i=1:size(BF,1)
                    plan1=[0 0 0;1 0 0;0 1 0]+2;
                    plan2=[P(BF(i,:),1),P(BF(i,:),2),P(BF(i,:),3)];
                    M=getTransformedPlan(plan1,plan2);
                    nodes_plan=M*[x(:).'+2;y(:).'+2;ones(1,length(x(:)))*2;ones(1,length(x(:)))];
                    tri=[tri;tri1+length(nodes)];
                    nodes=[nodes;nodes_plan(1:3,:).'];
                end
                nodes=nodes*0.99999;

                X0=sum([nodes(tri(:,1),1),nodes(tri(:,2),1),nodes(tri(:,3),1)],2)/3;
                Y0=sum([nodes(tri(:,1),2),nodes(tri(:,2),2),nodes(tri(:,3),2)],2)/3;
                Z0=sum([nodes(tri(:,1),3),nodes(tri(:,2),3),nodes(tri(:,3),3)],2)/3;
                in=shp.inShape(X0,Y0,Z0);
                tri=tri(in,:);
                w=obj.eval(nodes); w=w(nVertex,:,:);
                w(w>1)=1; w(w<0)=0; % for points outside the polyhedron
                trisurf(tri,nodes(:,1),nodes(:,2),nodes(:,3),w(:))
                shading interp
                axis equal
                colorbar off; grid on;
                light('Position',[-50 -15 10])
                material([0.8 0.5 0.3])
                lighting flat
                view(45,30)
                axis off

                for i=1:size(vCenter,1)
                    text(vCenter(i,1)*1.15,vCenter(i,2)*1.2,vCenter(i,3)*1.2,strcat("v",num2str(i)))
                end
                colormap(jet)
                caxis([0,1]);
                colorbar
            end
        end
    end
end


%% Basis functions library

function [w,dwdrho] = wachspress2dVectorized(rho,domain2D)
%% Evaluate Wachspress basis functions and their gradients in a convex polygon
%% Inputs:
% - rho = coordinates where the shape functions should be computed
% - domain3D = Domain object of dimension 2.
%% Outputs:
% w  : basis functions = [w_1; ...; w_n]
% dwdrho : gradient of basis functions = [dwdrho_1; ...; dwdrho_n]
%
% based on the Matlab code from :
% M. Floater, A. Gillette, and N. Sukumar,
% “Gradient bounds for Wachspress coordinates on polytopes,”
% SIAM J. Numer. Anal., vol. 52, no. 1, pp. 515–532, 2014,
% doi: 10.1137/130925712.
%
% vectorized by T. Cherriere (2021)

v=domain2D.Vertices;
np=size(rho,1);
n = size(v,1);

un = reshape(getNormals(v).',2,1,[]);
h = mult(reshape(v.',1,2,[]) - reshape(rho.',1,2,1,np),un);
p=un./h;
i=1:n; im1=mod(i-2,n) + 1;
w=p(1,:,im1,:).*p(2,:,i,:)-p(1,:,i,:).*p(2,:,im1,:);
wsum=sum(w,3);
R= permute(p(:,:,im1,:)+p(:,:,i,:),[3,1,4,2]);
w = t(shiftdim(w./wsum,1));
phiR = mult(t(w),R);
dwdrho = w.*(R-phiR);

end

function un = getNormals(v)
% Function to compute the outward unit normal to each edge
n = size(v,1);un = zeros(n,2);
ind1=mod(1:n,n)+1;
ind2=1:n;
d = v(ind1,:) - v(ind2,:);
un(ind2,:) = [d(ind2,2) -d(ind2,1)]./vecnorm(d,2,2);
end

function [w, dwdrho] = wachspress3dVectorized(rho,domain3D)
% Evaluate Wachspress basis functions and their gradients
% in a convex polyhedron
%% Inputs:
% - rho = coordinates where the shape functions should be computed
% - domain3D = Domain object of dimension 3.
%% Outputs:
% w  : basis functions = [w_1; ...; w_n]
% dwdrho : gradient of basis functions = [dwdrho_1; ...; dwdrho_n]
%
% based on the Matlab code from :
% M. Floater, A. Gillette, and N. Sukumar,
% “Gradient bounds for Wachspress coordinates on polytopes,”
% SIAM J. Numer. Anal., vol. 52, no. 1, pp. 515–532, 2014,
% doi: 10.1137/130925712.
%
% vectorized by T. Cherriere (2021)

np=size(rho,1);
v=domain3D.Vertices; un=domain3D.Normals; g=domain3D.Vertices2Facets;

n = size(v,1);w = zeros(n,1,np); R = zeros(n,3,np);

rho=reshape(rho.',1,3,1,np);
un=reshape(un.',1,3,[]);
v=reshape(v.',1,3,[]);

for i = 1:n
    f = g{i};
    k = length(f);
    h = mult(v(:,:,i) - rho,t(un(:,:,f)));
    p = permute(un(:,:,f)./ h,[1,2,4,3]);
    j = 1:k-2;
    wloc=permute(p(1,1,:,j).*(p(1,2,:,j+1).*p(1,3,:,k)-p(1,2,:,k).*p(1,3,:,j+1))+...
        p(1,1,:,j+1).*(p(1,3,:,j).*p(1,2,:,k)-p(1,3,:,k).*p(1,2,:,j))+...
        p(1,1,:,k).*(p(1,2,:,j).*p(1,3,:,j+1)-p(1,2,:,j+1).*p(1,3,:,j)),[4,2,3,1]);
    Rloc=permute(p(:,:,:,j)+p(:,:,:,j+1)+p(:,:,:,k),[4,2,3,1]);
    w(i,:,:) = sum(wloc,1);
    R(i,:,:) = mult(t(wloc),Rloc)./ w(i,:,:);
end

wsum = sum(w,1);
w  = w./wsum;
phiR = mult(t(w),R);
dwdrho= w .* (R - phiR);
end