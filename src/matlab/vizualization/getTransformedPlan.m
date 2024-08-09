function out1= getTransformedPlan(plan1,plan2)
% out1= getTransformedPlan(plan1,plan2)
%
% for plotting the faces of a 3D polytope

x1=plan1(1,1); x2=plan1(2,1); x3=plan1(3,1);
y1=plan1(1,2); y2=plan1(2,2); y3=plan1(3,2);
z1=plan1(1,3); z2=plan1(2,3); z3=plan1(3,3);

a1=plan2(1,1); a2=plan2(2,1); a3=plan2(3,1);
b1=plan2(1,2); b2=plan2(2,2); b3=plan2(3,2);
c1=plan2(1,3); c2=plan2(2,3); c3=plan2(3,3);

t2 = x1.*y2.*z3;
t3 = x1.*y3.*z2;
t4 = x2.*y1.*z3;
t5 = x2.*y3.*z1;
t6 = x3.*y1.*z2;
t7 = x3.*y2.*z1;
t8 = -t3;
t9 = -t4;
t10 = -t7;
t11 = t2+t5+t6+t8+t9+t10;
t12 = 1.0./t11;
mt1 = [t12.*(a1.*y2.*z3-a1.*y3.*z2-a2.*y1.*z3+a2.*y3.*z1+a3.*y1.*z2-a3.*y2.*z1),t12.*(b1.*y2.*z3-b1.*y3.*z2-b2.*y1.*z3+b2.*y3.*z1+b3.*y1.*z2-b3.*y2.*z1),t12.*(c1.*y2.*z3-c1.*y3.*z2-c2.*y1.*z3+c2.*y3.*z1+c3.*y1.*z2-c3.*y2.*z1),t12.*(y1.*z2-y2.*z1-y1.*z3+y3.*z1+y2.*z3-y3.*z2),-t12.*(a1.*x2.*z3-a1.*x3.*z2-a2.*x1.*z3+a2.*x3.*z1+a3.*x1.*z2-a3.*x2.*z1),-t12.*(b1.*x2.*z3-b1.*x3.*z2-b2.*x1.*z3+b2.*x3.*z1+b3.*x1.*z2-b3.*x2.*z1)];
mt2 = [-t12.*(c1.*x2.*z3-c1.*x3.*z2-c2.*x1.*z3+c2.*x3.*z1+c3.*x1.*z2-c3.*x2.*z1),-t12.*(x1.*z2-x2.*z1-x1.*z3+x3.*z1+x2.*z3-x3.*z2),t12.*(a1.*x2.*y3-a1.*x3.*y2-a2.*x1.*y3+a2.*x3.*y1+a3.*x1.*y2-a3.*x2.*y1),t12.*(b1.*x2.*y3-b1.*x3.*y2-b2.*x1.*y3+b2.*x3.*y1+b3.*x1.*y2-b3.*x2.*y1),t12.*(c1.*x2.*y3-c1.*x3.*y2-c2.*x1.*y3+c2.*x3.*y1+c3.*x1.*y2-c3.*x2.*y1),t12.*(x1.*y2-x2.*y1-x1.*y3+x3.*y1+x2.*y3-x3.*y2),0.0,0.0,0.0,0.0];
out1 = reshape([mt1,mt2],4,4);
end
