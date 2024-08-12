classdef Domain
    %Domain Class
    %
    %Domain(type,radius, epsProj,lengthNormalFan)
    %
    % For 0D, 1D and 2D regular polytopes, type is a number which
    % indicates the number of vertices :
    %  type = 1 -> a dot (dimension = 0)
    %  type = 2 -> a line (dimension = 1)
    %  type = n > 2 -> a regular polygon with n vertices (dimension 2)
    %
    % For 3D polytopes, type is a string which denote a specific
    % polytope. Available :
    %  "tetra" or "tetraedron" -> tetraedron
    %  "cube" -> cube
    %  "diamondN" -> diamond with a base of N vertices (N is a number)
    %  "prismN" -> prism with a base of N vertices (N is a number)
    %
    % radius is a float, representing the radius of the circumcircle
    % (default value = 1).
    %
    % epsProj is a float determining the accuracy of the
    % projection. 0 leads to exact solution but singular shape 
    % functions values. Default 1e-5
    %
    % lengthNormalFan is the length of the Normal fan cones. When the
    % points to project are outside the circle with 0.49*lengthNormalFan
    % radius in 2D (based on equilateral triangle domain) or outside a 
    % sphere with 0.32*lengthNormalFan radius (based on regular
    % tetraedron), they are first projected onto the associated cirle /
    % sphere to be inside the normal cones. Default : 1000
    %
    % to add 2D domains, just add the vertices (however the 
    % available regular polygons should satisfy all your needs)
    % to add 3D domains, the vertices are not sufficient, one
    % should also specify the edges, the facets and their
    % connectivity.
    %
    % In all cases, the polytopes must be convex and centered on 
    % the origin.
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
        Vertices
        Edges
        Facets
        Normals
        Vertices2Facets
        Edges2Facets
        Dimension
        NormalFan
    end
    
    methods
        function obj = Domain(type,radius, epsProj,lengthNormalFan)
            %obj = Domain(type, radius, epsProj)
            %
            % For 0D, 1D and 2D regular polytopes, type is a number which
            % indicates the number of vertices :
            %  type = 1 -> a dot (dimension = 0)
            %  type = 2 -> a line (dimension = 1)
            %  type = n > 2 -> a regular polygon with n vertices (dimension 2)
            %
            % For 3D polytopes, type is a string which denote a specific
            % polytope. Available :
            %  "tetra" or "tetraedron" -> tetraedron
            %  "cube" -> cube
            %  "diamondN" -> diamond with a base of N vertices (N is a number)
            %  "prismN" -> prism with a base of N vertices (N is a number)
            %
            % radius is a float, representing the radius of the circumcircle
            % (default value = 1).
            %
            % epsProj is a float determining the accuracy of the
            % projection. 0 leads to exact solution but singular shape 
            % functions values. Default 1e-5
            %
            % to add 2D domains, just add the vertices (however the 
            % available regular polygons should satisfy all your needs)
            % to add 3D domains, the vertices are not sufficient, one
            % should also specify the edges, the facets and their
            % connectivity.
            %
            % In all cases, the polytopes must be convex and centered on 
            % the origin.
           
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% 0D
            
            if nargin<=1 || isempty(radius)
                radius=1;
            end

            if class(type)=="struct"
                obj.Vertices =type.vertices;
                obj.Edges = type.edges;
                obj.Facets = type.facets;
                obj.Normals = type.normals;
                obj.Vertices2Facets = type.vertices2facets;
                obj.Edges2Facets = type.edges2facets;
                obj.Dimension = type.dimension;
                obj.NormalFan = type.normal_fan;
            else
                
                if class(type)=="double" && type==1 % dimension 0 : a dot
                    obj.Vertices=0;
                    obj.Dimension=0;
                    v=obj.Vertices;
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %% 1D
                    
                elseif class(type)=="double" && type==2 % dimension 1 : a line
                    obj.Vertices=[-0.5;0.5];
                    obj.Dimension=1;
                    v=obj.Vertices;
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %% 2D
                    
                elseif class(type)=="double" % dimension 2 : a regular polygon
                    if  length(type)==1
                        theta=linspace(0,2*pi,type+1)+(2*pi)/(2*type);
                        theta=shiftdim(theta(1:end-1),-2);
                        R=mult([cos(theta),-sin(theta);sin(theta),cos(theta)],[1;0]);
                        obj.Vertices=reshape(R,[2,type]).';
                        v=obj.Vertices;
                    else
                        v=type;
                        obj.Vertices=type;
                    end
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %% 3D
                    
                elseif lower(type)=="tetra" || lower(type)=="tetraedron"
                    
                    theta=linspace(0,2*pi,3+1)+(2*pi)/(2*3);
                    theta=shiftdim(theta(1:end-1),-2);
                    h = 1/3*radius;
                    R=mult([cos(theta),-sin(theta);sin(theta),cos(theta)],[radius*2*sqrt(2)/3;0]);
                    v1=reshape(R,[2,3]).';
                    obj.Vertices=[v1,-ones(3,1)*h;[0 0 radius]];
                    obj.Vertices=obj.Vertices-mean(obj.Vertices,1);
                    v=obj.Vertices;
                    
                    un(1,:)=cross(v(4,:)-v(1,:),v(2,:)-v(1,:)); un(2,:)=cross(v(4,:)-v(2,:),v(3,:)-v(2,:));
                    un(3,:)=cross(v(4,:)-v(3,:),v(1,:)-v(3,:)); un(4,:)=cross(v(2,:)-v(1,:),v(3,:)-v(1,:));
                    obj.Normals=-un./vecnorm(un,2,2);
                    
                    obj.Vertices2Facets={[1,3,4];[2,1,4];[3,2,4];[1,2,3]};
                    obj.Edges=[1,2;2,3;3,1;4,1;4,2;4,3];
                    obj.Facets={[1,-5,4];[2,-6,5];[3,-4,6];[-1,-3,-2]};
                    
                    obj.Edges2Facets=zeros(length(obj.Edges),2);
                    for i=1:length(obj.Facets)
                        edges=abs(obj.Facets{i});
                        for j=1:length(edges)
                            if obj.Edges2Facets(edges(j),1)==0
                                obj.Edges2Facets(edges(j),1)=i;
                            else
                                obj.Edges2Facets(edges(j),2)=i;
                            end
                        end
                    end
                    %%
                elseif lower(type)=="cube"
                    v=2*radius*sqrt(3)/3*[0,0,0;1 0 0;0 1 0 ;1 1 0; 0 0 1; 1 0 1; 0 1 1; 1 1 1]; 
                    obj.Vertices=v-mean(v,1);
                    obj.Vertices2Facets={[1,4,6];[2,1,6];[4,3,6];[3,2,6];[5,4,1];[5,1,2];[5,3,4];[5,2,3]};
                    
                    un(1,:)=cross(v(2,:),v(5,:)); un(2,:)=cross(v(4,:)-v(2,:),v(6,:)-v(2,:));
                    un(3,:)=cross(v(3,:)-v(4,:),v(8,:)-v(4,:)); un(4,:)=cross(v(1,:)-v(3,:),v(7,:)-v(3,:));
                    un(5,:)=cross(v(6,:)-v(5,:),v(7,:)-v(5,:)); un(6,:)=cross(v(1,:)-v(2,:),v(4,:)-v(2,:));
                    obj.Normals=un./vecnorm(un,2,2);
                    
                    obj.Edges=[1,2;2,4;4,3;3,1;5,6;6,8;8,7;7,5;1,5;2,6;4,8;3,7];
                    obj.Facets={[1,10,-5,-9];[2,11,-6,-10];[3,12,-7,-11];[4,9,-8,-12];...
                        [5,6,7,8];[1,2,3,4]};
                    obj.Edges2Facets=zeros(length(obj.Edges),2);
                    for i=1:length(obj.Facets)
                        edges=abs(obj.Facets{i});
                        for j=1:length(edges)
                            if obj.Edges2Facets(edges(j),1)==0
                                obj.Edges2Facets(edges(j),1)=i;
                            else
                                obj.Edges2Facets(edges(j),2)=i;
                            end
                        end
                    end
                    v=obj.Vertices;
                    %%
                elseif contains(lower(type),"diamond")
                    n  = str2double(extract(lower(type),regexpPattern("\d+")));
                    theta=linspace(0,2*pi,n+1)+pi/n;
                    theta=shiftdim(theta(1:end-1),-2);
                    R=mult([cos(theta),-sin(theta);sin(theta),cos(theta)],[radius;0]);
                    v=reshape(R,[2,n]).'; v=[[v,zeros(n,1)];0,0,radius;0,0,-radius]; obj.Vertices=v;
                    for i=1:n
                        i1=i;
                        i2 = mod(i+n-2,n)+1;
                        i3 = i2+n;
                        i4 = i+n;
                        obj.Vertices2Facets{i} = [i1,i2,i3,i4];
                    end
                    obj.Vertices2Facets{n+1} = 1:n;
                    obj.Vertices2Facets{n+2} = (2*n):-1:(n+1);
                    for i=1:n
                        un(i,:)=cross(v(mod(i,n)+1,:)-v(i,:),v(n+1,:)-v(i,:));
                        un(i+n,:)=cross(v(n+2,:)-v(i,:),v(mod(i,n)+1,:)-v(i,:));
                    end
                    obj.Normals=un./vecnorm(un,2,2);
                    for i=1:n
                        obj.Edges(i,:)=[i,mod(i,n)+1];
                        obj.Edges(i+n,:)=[i,1+n];
                        obj.Edges(i+2*n,:)=[2+n,i];
                    end
                    for i=1:n
                        i1 = i;
                        i2 = n + 1+ mod(i,n);
                        i3 = - n - i;
                        i2b = -i2 - n;
                        i3b = -i3+n;
                        obj.Facets{i} = [i1,i2,i3];
                        obj.Facets{i+n} = [i1,i2b,i3b];
                    end
                    obj.Edges2Facets=zeros(length(obj.Edges),2);
                    for i=1:length(obj.Facets)
                        edges=abs(obj.Facets{i});
                        for j=1:length(edges)
                            if obj.Edges2Facets(edges(j),1)==0
                                obj.Edges2Facets(edges(j),1)=i;
                            else
                                obj.Edges2Facets(edges(j),2)=i;
                            end
                        end
                    end
                    v=obj.Vertices;
                    %%
                elseif contains(lower(type),"prism")
                    n  = str2double(extract(lower(type),regexpPattern("\d+")));
                    theta=linspace(0,2*pi,n+1)+pi/n;
                    theta=shiftdim(theta(1:end-1),-2);
                    R=mult([cos(theta),-sin(theta);sin(theta),cos(theta)],[sqrt(3)*radius/2;0]);
                    v=reshape(R,[2,n]).'; v=[[v,-radius/2*ones(n,1)];[v,radius/2*ones(n,1)]]; obj.Vertices=v;
                    for i=1:n
                        i1=i;
                        i2 = mod(i+n-2,n)+1;
                        obj.Vertices2Facets{i} = [i1,i2,n+1];
                        obj.Vertices2Facets{i+n} = [i1,n+2,i2];
                    end
                    for i=1:n
                        i2 = mod(i,n)+1;
                        obj.Edges(i,:) = [i,i2];
                        obj.Edges(i+n,:) = [i+n,i2+n];
                        obj.Edges(i+2*n,:)= [i,i+n];
                    end
                    for i=1:n
                        i1=i;
                        i2 = mod(i,n)+1+2*n;
                        i3 = -mod(i-1,n)-1-n;
                        i4 = -mod(i-1,n)-1-2*n;
                        obj.Facets{i} = [i1,i2,i3,i4];
                    end
                    obj.Facets{n+1}=1:n;
                    obj.Facets{n+2}=n+(1:n);
                    for i=1:n
                        un(i,:)=cross(v(mod(i,n)+1,:)-v(i,:),v(n+i,:)-v(i,:));
                    end
                    un(n+1,:)=[0,0,-1];
                    un(n+2,:)=[0,0,1];
                    obj.Normals=un./vecnorm(un,2,2);
                    obj.Edges2Facets=zeros(length(obj.Edges),2);
                    for i=1:length(obj.Facets)
                        edges=abs(obj.Facets{i});
                        for j=1:length(edges)
                            if obj.Edges2Facets(edges(j),1)==0
                                obj.Edges2Facets(edges(j),1)=i;
                            else
                                obj.Edges2Facets(edges(j),2)=i;
                            end
                        end
                    end
                end
                
                if nargin<=2 || isempty(epsProj), epsProj = 1e-5; end
                if nargin<=3 || isempty(lengthNormalFan), lengthNormalFan = 1000; end
                if size(obj.Vertices,2)==2
                    [cones_edge,cones_vertices,edges,vertices]=normalFan2D(v,epsProj,lengthNormalFan);
                    obj.NormalFan.cones_edge=cones_edge;
                    obj.NormalFan.cones_vertices=cones_vertices;
                    obj.NormalFan.edges=edges;
                    obj.NormalFan.vertices=vertices;
                    obj.NormalFan.length = lengthNormalFan;
                    obj.Dimension=2;
                    
                    obj.Edges=edges;
                    normales=[0,1;-1 0]*(obj.Vertices(obj.Edges(:,2),:)-obj.Vertices(obj.Edges(:,1),:)).';
                    obj.Normals=((normales).')./vecnorm(normales.',2,2);
                    
                elseif size(obj.Vertices,2)==3
                    [cones_facets,cones_edges,cones_vertices,facets,edges,vertices]=normalFan3D(obj,epsProj,lengthNormalFan);
                    obj.NormalFan.cones_facets=cones_facets;
                    obj.NormalFan.cones_edges=cones_edges;
                    obj.NormalFan.cones_vertices=cones_vertices;
                    obj.NormalFan.facets=facets;
                    obj.NormalFan.edges=edges;
                    obj.NormalFan.vertices=vertices;
                    obj.NormalFan.length = lengthNormalFan;
                    obj.Dimension=3;
                end
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Projection
        
        function val = projection(obj,val)
            % val = projection(obj,val)
            % projection of val onto the domain
            if obj.Dimension==1
                val(val<-0.5)=-0.5;
                val(val>0.5)=0.5;
            elseif obj.Dimension==2
                con_edg = obj.NormalFan.cones_edge;
                con_ver = obj.NormalFan.cones_vertices;
                edg = obj.NormalFan.edges;
                vert = obj.NormalFan.vertices;
                L = obj.NormalFan.length;
                % Projection with the normal fan
                val=projection2D(val,con_ver,con_edg,vert,edg,L);
                
            elseif obj.Dimension==3
                con_fac = obj.NormalFan.cones_facets;
                con_edg = obj.NormalFan.cones_edges;
                con_ver = obj.NormalFan.cones_vertices;
                fac = obj.NormalFan.facets;
                edg = obj.NormalFan.edges;
                vert = obj.NormalFan.vertices;
                L = obj.NormalFan.length;
                % Projection with the normal fan
                val=projection3D(val,obj.Normals,con_fac,con_edg,con_ver,fac,edg,vert,L);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Display
        
        function plot(obj,legend,pow,resolution)
        % plot(obj,legend,pow,resolution)
        % representation of the domain

           wachspress=ShapeFunction("wachspress",obj);
           
            if nargin<=1 || isempty(legend)
                legend=string([]);
                for i=1:length(obj.Vertices)
                    legend(i)=convertCharsToStrings(strcat("v",num2str(i)));
                end
            end
            if nargin<=2 || isempty(pow)
                pow = 5;
            end

            color = legend2color(legend);
            nColors=length(obj.Vertices);
            vCenter=obj.Vertices-mean(obj.Vertices,1);
           
            if obj.Dimension==0
                scatter(0,0,500,100,'o','filled');
                try
                    text(vCenter*1.15,0,legend)
                end
                
            elseif size(vCenter,2)==1 && size(vCenter,1)==2 % 1D = 2 materials
                if nargin<=3 || isempty(resolution)
                    resolution = 100;
                end
                x=linspace(vCenter(1),vCenter(2),resolution).';
                y=zeros(100,1) ;
                w=wachspress.eval(x);
                c=permute(sum(color(1:2,:).*(w.^pow),1)+0.5,[3 2 1]);
                surf([x(:) x(:)], [y(:) y(:)], [x(:) x(:)], ...  % Reshape and replicate data
                    'FaceColor', 'none', ...    % Don't bother filling faces with color
                    'EdgeColor', 'interp', ...  % Use interpolated color for edges
                    'LineWidth', 5);            % Make a thicker line
                view(2);   % Default 2-D view
                colormap(c);
                axis(1.4*[-max(abs(vCenter(:))),max(abs(vCenter(:))),-max(abs(vCenter(:))),max(abs(vCenter(:)))])
                for i=1:length(legend)
                    try
                        text(vCenter(i)*1.15,0,legend(i))
                    end
                end
                axis off
                
            elseif size(vCenter,2)==2 %2D
                if nargin<=3 || isempty(resolution)
                    resolution = 100;
                end
                pgon=polyshape(vCenter(:,1),vCenter(:,2));
                [x,y]=meshgrid(linspace(-max(abs(vCenter(:)))*0.99999,...
                    max(abs(vCenter(:)))*0.99999,resolution),...
                    linspace(-max(abs(vCenter(:)))*0.99999,...
                    max(abs(vCenter(:)))*0.99999,resolution));
                in=pgon.isinterior(x(:),y(:));
                tr = delaunayTriangulation(x(in),y(in));
                nodes=tr.Points; elements=tr.ConnectivityList;
                X0=sum([nodes(elements(:,1),1),nodes(elements(:,2),1),nodes(elements(:,3),1)],2)/3;
                Y0=sum([nodes(elements(:,1),2),nodes(elements(:,2),2),nodes(elements(:,3),2)],2)/3;
                w=wachspress.eval([X0,Y0]);
                
                c=permute(sum(color(1:nColors,:).*(w.^pow),1)+0.5,[3 2 1]);
                plotcolor2D(nodes,elements,(1:length(elements)));
                colormap(c); colorbar off; maxval=max(abs(nodes(:)));
                axis([-maxval maxval -maxval maxval]*1.5);axis equal
                hold on; plot(pgon,'facealpha',0); grid on;
                
                for i=1:length(legend)
                    try
                        text(vCenter(i,1)*1.15,vCenter(i,2)*1.2,legend(i))
                    end
                end
                axis off
                hold off;
                
            elseif size(vCenter,2)==3 %3D
                if nargin<=3 || isempty(resolution)
                    resolution = 50;
                end
                shp=alphaShape(vCenter(:,1),vCenter(:,2),vCenter(:,3),Inf);
                [BF, P] = boundaryFacets(shp);
                [x,y]=meshgrid(linspace(0,1,resolution),linspace(0,1,resolution));
                tri1=delaunay(x,y);
                tri=[];
                nodes=[];
                for i=1:size(BF,1)
                    plan1=[0 0 0;1 0 0;0 1 0]+2;
                    plan2=[P(BF(i,:),1),P(BF(i,:),2),P(BF(i,:),3)];
                    M=getTransformedPlan(plan1,plan2);
                    nodesPlane=M*[x(:).'+2;y(:).'+2;ones(1,length(x(:)))*2;ones(1,length(x(:)))];
                    tri=[tri;tri1+length(nodes)];
                    nodes=[nodes;nodesPlane(1:3,:).'];
                end
                nodes=nodes*0.99999;
                
                X0=sum([nodes(tri(:,1),1),nodes(tri(:,2),1),nodes(tri(:,3),1)],2)/3;
                Y0=sum([nodes(tri(:,1),2),nodes(tri(:,2),2),nodes(tri(:,3),2)],2)/3;
                Z0=sum([nodes(tri(:,1),3),nodes(tri(:,2),3),nodes(tri(:,3),3)],2)/3;
                
                in=shp.inShape(X0,Y0,Z0);
                tri=tri(in,:);
                X0=X0(in); Y0=Y0(in); Z0=Z0(in);
                w=wachspress.eval([X0,Y0,Z0]);
                X=[nodes(tri(:,1),1),nodes(tri(:,2),1),nodes(tri(:,3),1)];
                Y=[nodes(tri(:,1),2),nodes(tri(:,2),2),nodes(tri(:,3),2)];
                Z=[nodes(tri(:,1),3),nodes(tri(:,2),3),nodes(tri(:,3),3)];
                c=permute(sum(color(1:nColors,:).*(w.^pow),1)+0.5,[3 2 1]);
                %figure()
                patch(X.',Y.',Z.',1:length(c),'CDataMapping','direct','Edgealpha',0);
                colormap(c); axis equal
                colorbar off; grid on;
                light('Position',[-50 -15 10])
                material([0.8 0.5 0.3])
                lighting flat
                view(45,30)
                
                for i=1:length(legend)
                    text(vCenter(i,1)*1.2,vCenter(i,2)*1.2,vCenter(i,3)*1.2,legend(i))
                end
            end
            hold off;
            axis off
        end     

        function plot_normalFan(obj)
            % plot_normalFan(obj)
            % give a reprensentation of the normal cones of the domain. The
            % normal cones are useful for the orthogonal projection of 
            % points onto the domain
            alpha=0.2;
            L3D=1;
            L2D=3;
            figure();
            hold on
            listplot=obj.Vertices(reshape(obj.Edges.',[],1),:);

            if obj.Dimension==2
                for i=1:length(obj.Vertices) % point
                    p1=obj.Vertices(i,:);
                    p2=p1+L2D*obj.Normals(i,:);
                    p3=p1+L2D*obj.Normals(mod(i+length(obj.Vertices)-2,length(obj.Vertices))+1,:);
                    p=[p1;p2;p3];
                    triangle=polyshape(p(:,1),p(:,2));
                    plot(triangle,'facecolor','r')
                end
                for i=1:length(obj.NormalFan.cones_edge) % edge
                    p1=[obj.Vertices(obj.Edges(i,1),1),obj.Vertices(obj.Edges(i,1),2)];
                    p4=[obj.Vertices(obj.Edges(i,2),1),obj.Vertices(obj.Edges(i,2),2)];
                    p2=p1+L2D*obj.Normals(i,:);
                    p3=p4+L2D*obj.Normals(i,:);
                    p=[p1;p2;p3;p4];
                    rectangle=polyshape(p(:,1),p(:,2));
                    plot(rectangle,'facecolor','g')
                end
                scatter(obj.Vertices(:,1),obj.Vertices(:,2),'k','filled');
                plot(reshape(listplot(:,1),2,[]),reshape(listplot(:,2),2,[]),'k');
            elseif obj.Dimension==3
                for i=1:length(obj.NormalFan.cones_vertices) % point
                    fac_loc=obj.Vertices2Facets{i};
                    p=obj.Vertices(i,:);
                    for j=1:length(fac_loc)
                        p=[p;p(1,:)+L3D*obj.Normals(fac_loc(j),:)];
                    end
                    cone=alphaShape(p(:,1),p(:,2),p(:,3),Inf);
                    plot(cone,'facecolor','r','edgecolor','r','FaceAlpha',alpha,'EdgeAlpha',0)
                end
                for i=1:length(obj.NormalFan.cones_edges) % edge
                    fac_loc=obj.Edges2Facets(i,:);
                    p1=obj.Vertices(obj.Edges(i,1),:);
                    p2=obj.Vertices(obj.Edges(i,2),:);
                    p3=p1+obj.Normals(fac_loc(1),:)*L3D;
                    p4=p2+obj.Normals(fac_loc(1),:)*L3D;
                    p5=p1+obj.Normals(fac_loc(2),:)*L3D;
                    p6=p2+obj.Normals(fac_loc(2),:)*L3D;
                    p=[p1;p2;p3;p4;p5;p6];
                    toit=alphaShape(p(:,1),p(:,2),p(:,3),Inf);
                    plot(toit,'facecolor','g','edgecolor','g','FaceAlpha',alpha,'EdgeAlpha',0)
                end  
                for i=1:length(obj.NormalFan.cones_facets) % facet
                    edges_facette=abs(obj.Facets{i}(:));
                    noeuds_facette=unique(obj.Edges(edges_facette(:),:));
                    vloc_base=obj.Vertices(noeuds_facette,:);
                    vloc_extrude=vloc_base+obj.Normals(i,:)*L3D;
                    vshp=[vloc_base;vloc_extrude];
                    prism=alphaShape(vshp(:,1),vshp(:,2),vshp(:,3),Inf);
                    plot(prism,'facecolor','b','edgecolor','b','FaceAlpha',alpha,'EdgeAlpha',0)
                end
                scatter3(obj.Vertices(:,1),obj.Vertices(:,2),obj.Vertices(:,3),'k','filled');
                plot3(reshape(listplot(:,1),2,[]),reshape(listplot(:,2),2,[]),reshape(listplot(:,3),2,[]),'k');
            else
                error("Incorrect dimension")
            end
            axis equal
            axis off
            hold off
        end
              
    end
end

%% Projection function

function [rectangles,triangles,edges,vertices]=normalFan2D(v,epsProj,L)
% [rectangles,triangles,edges,vertices]=normalFan2D(v,epsProj)
%
% returns a list of triangles and rectangles that constitute the normal
% fan of the polygon (which must be convex), as well as the associated 
% edges and points.

if nargin<=1 || isempty(epsProj),  epsProj=1e-5; end
if nargin<=2 || isempty(L),  L=1000; end

nv=length(v);
v=v-mean(v,1);
v=v*(1-epsProj);
edges=[(1:nv).',[(2:nv),1].'];
vertices=v;
normals=[0,1;-1 0]*(v(edges(:,2),:)-v(edges(:,1),:)).';
normals=((normals).')./vecnorm(normals.',2,2);

rectangles=cell(nv,1);

for i=1:nv % rectangle
    p1=[v(edges(i,1),1),v(edges(i,1),2)];
    p4=[v(edges(i,2),1),v(edges(i,2),2)];
    p2=p1+L*normals(i,:);
    p3=p4+L*normals(i,:);
    p=[p1;p2;p3;p4];
    rectangles{i}=polyshape(p(:,1),p(:,2));
    %plot(rectangles{i},'facecolor','b') %uncomment to see the construction process
    %hold on
    %pause(0.1)
end

triangles=cell(nv,1);

for i=1:nv-1 % triangles
    p1=v(i+1,:);
    p2=p1+L*normals(i,:);
    p3=p1+L*normals(i+1,:);
    p=[p1;p2;p3];
    triangles{i+1}=polyshape(p(:,1),p(:,2));
    %plot(triangles{i+1},'facecolor','r')
    %hold on
    %pause(0.1)
end
p1=v(1,:);
p2=p1+L*normals(nv,:);
p3=p1+L*normals(1,:);
p=[p1;p2;p3];
triangles{1}=polyshape(p(:,1),p(:,2));
%plot(triangles{1},'facecolor','r')
end

function [prisms,roofs,cones,facets,edges,vertices]=normalFan3D(domain,epsProj,L)
% [prisms,roofs,cones,facets,edges,vertices]=normalFan3D(domain,epsProj)
%
% returns a list of prisms, roofs, cones that constitute the normal
% fan of the 3D polytope (which must be convex), as well as the associated 
% facets, edges and points.

if nargin<=1 || isempty(epsProj),  epsProj=1e-5; end
if nargin<=2 || isempty(L),  L=1000; end

v=domain.Vertices;
v=v-mean(v,1);
v=v*(1-epsProj);

edges=domain.Edges;
facets=domain.Facets;
vertices=v;
normals=domain.Normals;

nv=length(v); ne=length(edges); nf=length(facets);

prisms=cell(nf,1);

for i=1:nf % prismes
    edges_facette=abs(facets{i}(:));
    noeuds_facette=unique(edges(edges_facette(:),:));
    vloc_base=v(noeuds_facette,:);
    vloc_extrude=vloc_base+normals(i,:)*L;
    vshp=[vloc_base;vloc_extrude];
    prisms{i}=alphaShape(vshp(:,1),vshp(:,2),vshp(:,3),Inf);
%       plot(prismes{i},'facealpha',alpha,'facecolor','r','edgecolor','g','edgealpha',alpha)
%        hold on
%        pause(0.6)
end

roofs=cell(nv,1);

for i=1:ne % toits
    fac_loc=domain.Edges2Facets(i,:);
    p1=v(edges(i,1),:);
    p2=v(edges(i,2),:);
    p3=p1+normals(fac_loc(1),:)*L;
    p4=p2+normals(fac_loc(1),:)*L;
    p5=p1+normals(fac_loc(2),:)*L;
    p6=p2+normals(fac_loc(2),:)*L;
    p=[p1;p2;p3;p4;p5;p6];
    roofs{i}=alphaShape(p(:,1),p(:,2),p(:,3),Inf);
%       plot(toits{i},'facealpha',alpha,'facecolor','g','edgecolor','g','edgealpha',alpha)
%       hold on
%       pause(0.6)
end

cones=cell(nv,1);

for i=1:nv % cones
    fac_loc=domain.Vertices2Facets{i};
    p=v(i,:);
    for j=1:length(fac_loc)
        p=[p;p(1,:)+L*normals(fac_loc(j),:)];
    end
    cones{i}=alphaShape(p(:,1),p(:,2),p(:,3),Inf);
%       plot(cones{i},'facealpha',alpha,'facecolor','b','edgecolor','b','edgealpha',alpha)
%       hold on
%      pause(0.6)
end

end

function rho=projection2D(rho,triangles,rectangles,vertices,edges,L)
% rho=projection2D(rho,triangles,rectangles,vertices,edges)
%
% returns projected values onto a convex 2D polygon.

% 0) radial projection to avoid lying outside the normal fan
    R = vecnorm(rho,2,2);
    coeff = 0.49;
    indL = R>(L*coeff);
    rho(indL,:) = rho(indL,:) .* (L*coeff./R(indL));

    % 1) projection on vertices (triangles in the normal fan)
N=length(triangles);
for i=1:N
    s=triangles{i}.Vertices;
    in=inpolygon(rho(:,1),rho(:,2),s(:,1),s(:,2));
    if sum(in)>0
        rho(in,1)=vertices(i,1);
        rho(in,2)=vertices(i,2);
    end
end

% 2) projection on edges (rectangles in the normal fan)

for i=1:N
    s=rectangles{i}.Vertices;
    in=inpolygon(rho(:,1),rho(:,2),s(:,1),s(:,2));
    if sum(in)>0
        projpoints=proj([vertices(edges(i,:).',1),vertices(edges(i,:).',2)], rho(in,:));
        rho(in,:)=shiftdim(projpoints,2);
    end
end


end

function [ProjPoint] = proj(vector, q)

p0 = vector(1,:);
p1 = vector(2,:);
q=shiftdim(q,-2);

a = [-q(1,1,:,1).*(p1(1)-p0(1)) - q(1,1,:,2).*(p1(2)-p0(2)); ...
    repmat(-p0(2).*(p1(1)-p0(1)) + p0(1).*(p1(2)-p0(2)),[ 1,1,size(q,3)])]; 
b = [p1(1) - p0(1), p1(2) - p0(2);...
    p0(2) - p1(2), p1(1) - p0(1)];

detb=b(1,1).*b(2,2)-b(1,2).*b(2,1);
invb=[b(1,1), b(2,1); b(1,2), b(2,2)]./detb;

ProjPoint = -mult(invb,a);
end

function rho=projection3D(rho,un,prismes,toits,cones,facets,edges,vertices,L)

% 0) radial projection to avoid lying outside the normal fan
    R = vecnorm(rho,2,2);
    coeff = 0.32;
    indL = R>(L*coeff);
    rho(indL,:) = rho(indL,:) .* (L*coeff./R(indL));

    % 1) projection on vertices (cones)
N=length(cones);
vertices=vertices-mean(vertices,1);
for i=1:N
    in=inShape(cones{i},rho);
    if sum(in)>0
        rho(in,1)=vertices(i,1);
        rho(in,2)=vertices(i,2);
        rho(in,3)=vertices(i,3);
    end
end

% 2) projection on edges (roofs)

N=length(toits);
for i=1:N
    in=inShape(toits{i},rho);
    if sum(in)>0
        p=vertices(edges(i,1),:);
        vec_dir=vertices(edges(i,2),:)-vertices(edges(i,1),:);
        projpoints=projEdges(vec_dir,p,rho(in,:));
        rho(in,:)=shiftdim(projpoints,2);
    end
end

% 3) projection on facets (prisms)

N=length(prismes);
for i=1:N
    in=inShape(prismes{i},rho);
    if sum(in)>0
        p=vertices(edges(abs(facets{i}(1)),1),:);
        n=un(i,:);
        projpoints=proj_plan(n,p,rho(in,:));
        rho(in,:)=shiftdim(projpoints,2);
    end
end

end

function [ProjPoint] = projEdges(vec_dir,p,q)

q=permute(q,[3,2,1]);
vec_dir=vec_dir(:)/norm(vec_dir,2);
ProjPoint=(p+mult(q-p,vec_dir).*vec_dir.');
end

function [ProjPoint] = proj_plan(n,p, q)
% project onto the plane
q=permute(q,[3,2,1]);
ProjPoint = (q - mult(q - p, n(:)).*n);
end