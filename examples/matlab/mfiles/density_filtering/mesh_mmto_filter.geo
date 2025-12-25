// Gmsh project created on Tue Dec 23 09:02:45 2025

kh = 0.05;

Rint = 1;
Rext = 2;
th = Pi/6;

krint1 = 0.2;
krext1 = 0.6;
kth1 = 0.3;

krint2 = krint1;
krext2 = krext1;
kth2 = 0.3;

h = kh * (Rext-Rint);

p1 = newp; Point(p1) = {0,0,0};
p2 = newp; Point(p2) = {Rint,0,0,h};
p3 = newp; Point(p3) = {Rext,0,0,h};
p4 = newp; Point(p4) = {Rext*Cos(th),Rext*Sin(th),0,h};
p5 = newp; Point(p5) = {Rint*Cos(th),Rint*Sin(th),0,h};

th1 = kth1 * th;
Rint1 = Rint + krint1 * (Rext - Rint);
Rext1 = Rint1 + krext1 * (Rext - Rint1);

p6 = newp; Point(p6) = {Rint1,0,0,h};
p7 = newp; Point(p7) = {Rext1,0,0,h};
p8 = newp; Point(p8) = {Rext1*Cos(th1),Rext1*Sin(th1),0,h};
p9 = newp; Point(p9) = {Rint1*Cos(th1),Rint1*Sin(th1),0,h};


th2 = kth2 * (th-th1);
Rint2 = Rint + krint2 * (Rext - Rint);
Rext2 = Rint2 + krext2 * (Rext - Rint2);

p10 = newp; Point(p10) = {Rint2*Cos(th),Rint2*Sin(th),0,h};
p11 = newp; Point(p11) = {Rint2*Cos(th-th2),Rint2*Sin(th-th2),0,h};
p12 = newp; Point(p12) = {Rext2*Cos(th-th2),Rext2*Sin(th-th2),0,h};
p13 = newp; Point(p13) = {Rext2*Cos(th),Rext2*Sin(th),0,h};


// Lines

Line(1) = {2, 6};
Line(2) = {6, 7};
Line(3) = {7, 3};
Line(4) = {9, 8};
Line(5) = {5, 10};
Line(6) = {10, 13};
Line(7) = {13, 4};
Line(8) = {11, 12};
Circle(9) = {2, 1, 5};
Circle(10) = {6, 1, 9};
Circle(11) = {7, 1, 8};
Circle(12) = {3, 1, 4};
Circle(13) = {12, 1, 13};
Circle(14) = {11, 1, 10};

// Surfaces
Curve Loop(1) = {4, -11, -2, 10};
Plane Surface(1) = {1};
Curve Loop(2) = {14, 6, -13, -8};
Plane Surface(2) = {2};
Curve Loop(3) = {5, -14, 8, 13, 7, -12, -3, 11, -4, -10, -1, 9};
Plane Surface(3) = {3};

// Physical

Physical Surface("interior", 15) = {3};
Physical Surface("magBottom", 16) = {1};
Physical Surface("magTop", 17) = {2};

Physical Curve("master", 18) = {1, 2, 3};
Physical Curve("slave", 19) = {5, 6, 7};

// Periodicity

Periodic Curve {5,6,7}={1,2,3} Rotate{{0,0,1},{0,0,0},th};


