classdef Interpolation
    %Interpolation Class
    %
    % To create an Interpolation object, type
    % obj = Interpolation(structure)
    %
    % where 'structure' contains the following fields :
    % - 'Domain' : an instance of the Domain class with n vertices
    % - 'Children' : a vector of n VertexFunction or Interpolation instances
    % - 'Label' : a unique string identifier
    % - 'Penalization' (facultative) : an array of Penalization instances
    %
    % An Interpolation object describes a tree. The leaves are
    % VertexFunctions, interpolated by ShapeFunctios and Penalizations. See
    % the examples files for some illustrations.
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
        Penalization = Penalization.empty(); 
        ShapeFunction  = ShapeFunction.empty();
        Children
        DimInput
        DimOutput
        Label
    end

    methods
        function obj = Interpolation(structure)
            % obj = Interpolation(structure)
            % where 'structure' contains the following fields :
            % - 'Domain' : an instance of the Domain class with n vertices
            % - 'Children' : a vector of n VertexFunction or Interpolation instances
            % - 'Label' : a unique string identifier
            % - 'Penalization' (facultative) : an array of Penalization instances 
            %
            % see "b_SimpleScalarInterp", "c_SimpleVectorInterp", 
            % "d_HierarchicalScalarInterp" and "e_HierarchicalVectorInterp"
            % for some examples.

            if ~isfield(structure,"ShapeFunction")
                structure.ShapeFunction="wachspress";
            end

            domain = structure.Domain; nm=size(domain.Vertices,1);
            assert(length(structure.Children)==nm,...
                "Number of children different than the number of vertices");
            obj.Children = structure.Children(:);
            obj.DimInput = obj.Children(1).DimInput;
            obj.DimOutput = obj.Children(1).DimOutput;
            obj.Label = structure.Label;
            obj.ShapeFunction = ShapeFunction(structure.ShapeFunction,domain,structure.Label);

            if isfield(structure,"Penalization")
                if numel(structure.Penalization)==1
                    obj.Penalization= repmat(structure.Penalization,nm,1);
                elseif numel(structure.Penalization)==nm
                    obj.Penalization=structure.Penalization(:);
                else
                    error('Wrong number of penalization functions')
                end
            else
                obj.Penalization= repmat(Penalization("simp",1),nm,1) ;

            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Initialize variables

        function x0 = initializeVariable(obj,n,type,radius,x0)
            % x0 = initializeVariable(obj,n,type,radius)
            %
            % - n = number of variables
            % - type = "rand" or "zero" 
            % - coeff = number
            %
            % returns a structure with a labeling corresponding to the
            % Interpolation obj, containing n points at the origin or
            % randomly distributed in a certain radius

            if nargin<=3 || isempty(radius); radius = 1; end
            if nargin<=4 || isempty(x0); x0 = struct(); end
            assert(~isfield(x0,obj.Label),"Non-unique labelisation of the interpolation nodes")
            dim = obj.ShapeFunction.Domain.Dimension;
            if lower(type) == "zero" || lower(type) == "null"
                x0.(obj.Label) = zeros(n,dim);
                for i=1:numel(obj.Children)
                    if class(obj.Children(i)) == "Interpolation"
                        x0 = obj.Children(i).initializeVariable(n,type,radius,x0);
                    end
                end
            elseif  lower(type) == "rand" || lower(type) == "random"
                switch dim
                    case 0
                        x0.(obj.Label) = zeros(n,dim);
                    case 1
                        x0.(obj.Label) = radius*(rand(n,dim)-0.5);
                    case 2
                        R = radius*rand(n,1);
                        th = rand(n,1)*2*pi;
                        [x,y] = pol2cart(th,R);
                        x0.(obj.Label) = [x,y];
                    case 3
                        R = radius*rand(n,1);
                        th = rand(n,1)*pi;
                        phi = rand(n,1)*2*pi;
                        [x,y,z] =  sph2cart(th,phi,R);
                        x0.(obj.Label) = [x,y,z] ;
                end
                for i=1:numel(obj.Children)
                    if class(obj.Children(i)) == "Interpolation"
                        x0 = obj.Children(i).initializeVariable(n,type,radius,x0);
                    end
                end
            end

        end

        function xproj = projection(obj,x,xproj)
            % xproj = projection(obj,x)
            %
            % Project x onto the domain of all the subdomains contained in  
            % the interpolation
            if nargin<=2 || isempty(xproj) ; xproj = x; end
            xproj.(obj.Label) = obj.ShapeFunction.Domain.projection(x.(obj.Label));
            for i=1:numel(obj.Children)
                if class(obj.Children(i)) == "Interpolation"
                    xproj = obj.Children(i).projection(x,xproj);
                end
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Compute interpolation

        function [w,dwdx]=evalBasisFunction(obj,x,w,dwdx)
            % [w,dwdrho]=evalBasisFunction(obj,x)
            % evaluate the value of the basis functions w and their
            % derivative dwdx at points x. 
            % x should be a structure with the labeled generated by the
            % initializeVariable method
            [w.(obj.Label),dwdx.(obj.Label)]=obj.ShapeFunction.eval(x.(obj.Label));
            if class(obj.Children)=="Interpolation"
                for i=1:length(obj.Children)
                    [w,dwdx]=obj.Children(i).evalBasisFunction(x,w,dwdx);
                end
            end

        end

        function result = eval(obj,x,a,w)
            % result = eval(obj,x,field)
            %
            %evaluate the interpolation at points x, for a given
            %VertexFunction field a.
            % x should be a structure with the labeled generated by the
            % initializeVariable method.

            if nargin<=3 || isempty(w); w=obj.evalBasisFunction(x); end

            sz = size(a,3);
            nChildren = numel(obj.Children);

            % 1) computation of the children
            coeff=zeros(obj.Children(1).DimOutput,1, sz,nChildren);
            if class(obj.Children)=="VertexFunction"
                for i=1:nChildren
                    coeff(:,:,:,i)=obj.Children(i).Expression(a);
                end
            else
                for i=1:nChildren
                    coeff(:,:,:,i)=obj.Children(i).eval(x,a,w);
                end
            end

            % 2) multiplication by the shape functions
            val=zeros(1,1,sz,nChildren);
            for i=1:nChildren
                val(:,:,:,i)=obj.Penalization(i).eval(w.(obj.Label)(i,:,:));
            end

            % 2) result
            result=sum(coeff.*val,4);
        end

        function result = evalda(obj,x,a,w)
            % result = eval(obj,x,field)
            %
            %evaluate the derivative of the interpolation at points x, 
            % for a given VertexFunction field a, w.r.t a
            % x should be a structure with the labeled generated by the
            % initializeVariable method.

            if nargin<=3 || isempty(w); w=obj.evalBasisFunction(x); end

            sz = size(a,3);
            d1 = obj.Children(1).DimOutput;
            d2 = obj.Children(1).DimInput;
            nChildren = numel(obj.Children);

            % 1) computation of the derivative of the children
            coeffd=zeros(d1,d2,sz,nChildren);
            if class(obj.Children)=="VertexFunction"
                for i=1:nChildren
                    coeffd(:,:,:,i) = obj.Children(i).Derivative(a);
                end
            else
                for i=1:nChildren
                    coeffd(:,:,:,i) = obj.Children(i).evalda(x,a);
                end
            end

            % 2) multiplication by the penalized shape functions
            Pw=zeros(1,1,sz,nChildren);
            for i=1:nChildren
                Pw(:,:,:,i)=obj.Penalization(i).eval(w.(obj.Label)(i,:,:));
            end

            % 3) result
            result=sum(coeffd.*Pw,4);
        end

        function result = evaldx(obj,x,a,w,dwdx,k,result)
            % result = eval(obj,x,field)
            %
            %evaluate the derivative of the interpolation at points x, 
            % for a given VertexFunction field a, w.r.t x
            % x should be a structure with the labeled generated by the
            % initializeVariable method.

            sz = size(a,3);
            d1 = obj.Children(1).DimOutput;
            nChildren = numel(obj.Children);

            if nargin<=4 || isempty(w) || isempty(dwdx); [w,dwdx] = obj.evalBasisFunction(x); end
            if nargin<=5 || isempty(k); k=1; end
            if nargin<=6 || isempty(result); result=struct(); end

            % 1) computation of the values of the children
            coeff = zeros(d1,1,sz,nChildren);
            if class(obj.Children)=="VertexFunction"
                for i=1:nChildren
                    coeff(:,:,:,i)=obj.Children(i).Expression(a);
                end
            else
                for i=1:nChildren
                    coeff(:,:,:,i)=obj.Children(i).eval(x,a,w);
                    Pw = obj.Penalization(i).eval(w.(obj.Label)(i,:,:));
                    result = obj.Children(i).evaldx(x,a, w, dwdx, k.*Pw, result);
                end
            end

            % 2) computation of the derivative of the penalized shape functions
            dPdw = zeros(1,1,sz,nChildren);
            for i=1:nChildren
                dPdw(1,1,:,i) = obj.Penalization(i).evald(w.(obj.Label)(i,1,:));
            end

            % 3) result
            result.(obj.Label) = k.*sum((coeff.*permute(dwdx.(obj.Label),[4 2 3 1]).*dPdw),4);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Gather data

        function nodes=getAllNodes(obj,nodes)
            % nodes=getAllNodes(obj)
            %
            % returns every nodes of the interpolation (that are also
            % Interpolation objects).
            if nargin<=1 || isempty(nodes)
                nodes = Interpolation.empty();
            end
            if class(obj.Children) == "Interpolation"
                nodes = [nodes;obj.Children(:)];
                for i=1:numel(obj.Children)
                    nodes = obj.Children(i).getAllNodes(nodes);
                end
            end
        end

        function vertexFunctions=getAllVertexFunctions(obj,vertexFunctions)
            % vertexFunctions=getAllVertexFunctions(obj,vertexFunctions)
            %
            % returns all leaves of the tree (VertexFunctions)
            if nargin<=1 || isempty(vertexFunctions)
                vertexFunctions = vertexFunctions.empty();
            end
            if class(obj.Children) == "Interpolation"
                for i=1:numel(obj.Children)
                    vertexFunctions = obj.Children(i).getAllNodes(vertexFunctions);
                end
            elseif class(obj.Children) == "VertexFunction"
                vertexFunctions = [vertexFunctions;obj.Children(:)];
            end
        end
      
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Display

        function [p,labels]=plotTree(obj,p,root,labels)
            % plotTree(obj)
            % Visualization of the interpolation as a tree
            if nargin<=1 || isempty(p)
                p=0;
                labels=strcat(obj.Label,', ',num2str(obj.ShapeFunction.Domain.Dimension),'D');
                root=1;
                if class(obj.Children)=="VertexFunction"
                    p=[p,root*ones(1,length(obj.Children))];
                    labels=[labels,obj.Children(:).Label];
                else
                    for i=1:length(obj.Children)
                        [p,labels]=obj.Children(i).plotTree(p,root,labels);
                    end
                end
                figure()
                treeplot(p,'bo','r:');
                [x,y] = treelayout(p);
                text(x + 0.02,y,labels);
                axis off
                hold off
            else
                p=[p,root];
                labels=[labels,strcat(obj.Label,', ',num2str(obj.ShapeFunction.Domain.Dimension),'D')];
                root=length(p);
                if class(obj.Children)=="VertexFunction"
                    p=[p,root*ones(1,length(obj.Children))];
                    labels=[labels,obj.Children(:).Label];
                else
                    for i=1:length(obj.Children)
                        [p,labels]=obj.Children(i).plotTree(p,root,labels);
                    end
                end
            end
        end

        function plot(obj,x,level,offset,distance,dir)
            % plot(obj,x)
            % Return a 3D vizualization of the hierarchized interpolation
            % domains
            % if x is given (structure with fields generated by
            % initializeVariable method), plot the position of the points
            % in the domains.
            if nargin<=1 || isempty(x); xyz=zeros(0,3); x=[] ; else; xyz = x.(obj.Label); end
            if nargin<=2 || isempty(level); level = 1; end
            if nargin<=3 || isempty(offset); offset = [0,0,0]; end
            if nargin<=4 || isempty(distance); distance = @(p) 0; end
            if nargin<=5 || isempty(dir); dir = [0,0,0]; end

            
            xyz3d = ones(size(xyz,1),3).*offset;
            xyz3d(:,1:size(xyz,2)) =  xyz3d(:,1:size(xyz,2))+xyz;
            c = legend2color(strcat("v",num2str(level)))+0.5;
            d = obj.ShapeFunction.Domain;
            v = d.Vertices;

            v3d = ones(size(v,1),3).*offset;
            v3d(:,1:size(v,2)) =  v3d(:,1:size(v,2))+v;

            fmin = @(p) 0;
            for i=2:size(v,2)
                fmin = @(p) fmin(p) +  (distance(p(i,:))-distance(p(i-1,:)))^2;
            end
            fmin = @(p) fmin(p) +  (distance(p(1,:))-distance(p(end,:)))^2;

            fobj = @(th) fmin(rotPoint((v3d-offset).',th(1),th(2),th(3)).'+offset);
            options = optimset('Display','off','algorithm','levenberg-marquardt');
            thOpt = fsolve(fobj,rand(3,1)*1e-2,options);

            v3d = rotPoint((v3d-offset).',thOpt(1),thOpt(2),thOpt(3)).'+offset;
            xyz3d = rotPoint((xyz3d-offset).',thOpt(1),thOpt(2),thOpt(3)).'+offset;
            hold on
            switch d.Dimension
                case 0
                    scatter3(v3d(:,1),v3d(:,2),v3d(:,3),100,c,'filled');
                    text(offset(1)+dir(1)*0.15,offset(2)+dir(2)*0.15,offset(3)+dir(3)*0.15,obj.Label)
                case 1
                    line(v3d(:,1),v3d(:,2),v3d(:,3),'color',c,'linewidth',3);
                    text(offset(1)+dir(1)*0.15,offset(2)+dir(2)*0.15,offset(3)+dir(3)*0.15,obj.Label)

                case 2
                    fill3(v3d(:,1),v3d(:,2),v3d(:,3),c,'FaceAlpha',0.3,'EdgeColor',c,'EdgeAlpha',0.3);
                    text(offset(1)+dir(1)*0.15,offset(2)+dir(2)*0.15,offset(3)+dir(3)*0.15,obj.Label)

                case 3
                    shp = alphaShape(v3d,inf);
                    plot(shp,'FaceColor',c,'FaceAlpha',0.3,'EdgeColor',c,'EdgeAlpha',0.3);
                    text(offset(1)+dir(1)*0.15,offset(2)+dir(2)*0.15,offset(3)+dir(3)*0.15,obj.Label)

            end
            scatter3(xyz3d(:,1),xyz3d(:,2),xyz3d(:,3),'+');

            hold off
            grid on

            bc = mean(v3d,1);
            if class(obj.Children) == "Interpolation"
                for i=1:numel(obj.Children)
                    offset = 2*(v3d(i,:)-bc)+bc;
                    dir = (v3d(i,:)-bc)/norm(v3d(i,:)-bc);
                    dist = @(p) norm(p-bc);
                    hold on
                    plot3([v3d(i,1),offset(1)],...
                        [v3d(i,2),offset(2)],...
                        [v3d(i,3),offset(3)],'k--');
                    hold off
                    obj.Children(i).plot(x,level+1,offset,dist,dir);
                end
            else
                for i=1:numel(obj.Children)
                    a = norm(v3d(i,:)-bc);
                    if a~=0
                        dir2 = 0.1*(v3d(i,:)-bc)/a;
                    else
                        dir2 = 0.1*dir;
                    end
                    l = obj.Children(i).Label;
                    text(v3d(i,1)+dir2(1),v3d(i,2)+dir2(2),v3d(i,3)+dir2(3),l,'color',c);
                end
            end
             view( [-5 3 5]);
             axis off
        end

    end
end

