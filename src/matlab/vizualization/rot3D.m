function R = rot3D(ax,th)
% necessary for nice 3D plots of hierarchical interpolation
u = ax/norm(ax); ux = u(1); uy = u(2); uz=u(3);
c = cos(th); s = sin(th);
R = [ux^2*(1-c)+c, ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s;
    ux*uy*(1-c)+uz*s, uy^2*(1-c)+c,  uy*uz*(1-c)-ux*s;
    ux*uz*(1-c)-uy*s, uy*uz*(1-c)+ux*s, uz^2*(1-c)+c];
end