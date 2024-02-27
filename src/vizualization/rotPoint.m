function p = rotPoint(p,th1,th2,th3)
% necessary for nice 3D plots of hierarchical interpolation
p = rot3D([0;0;1],th3)*rot3D([0;1;0],th2)*rot3D([1;0;0],th1)*p;
end