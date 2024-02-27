function plotcolor2D(p,elts,data,alpha)
elt1=elts(:,1); elt2=elts(:,2); elt3=elts(:,3);
x=[p(elt1,1), p(elt2,1), p(elt3,1)];
y=[p(elt1,2), p(elt2,2), p(elt3,2)];
z = zeros(size(x));
if nargin<=3 || isempty(alpha)
    patch(x.',y.',z.',data(:).','Edgecolor','none');
else
       hold on
   for i=1:length(x(:))/3
       patch(x(i,:).',y(i,:).',z(i,:).',data(i).','Edgecolor','none','FaceAlpha',alpha(i)); 
   end
   hold off
end
end
