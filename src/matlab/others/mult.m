function result=mult(varargin)
% resultat=mult(A,B,...,N)
%
% Compute page-wise products of matrices
% A(:,:,i,j..)*B(:,:,i,j..)*...*N(:,:,i,j..) efficiently.
% Based on pagemtimes (Matlab versions newer than R2020B).
% For older version, uncomment line 13 and use mmx :
% https://de.mathworks.com/matlabcentral/fileexchange/37515-mmx

result=varargin{1};

v = version; [s,e] = regexp(v,'20\d{2}\w'); v = v(s:e);
if str2double(v(1:4))>2020 || (str2double(v(1:4))==2020 && v(5) == "a")
    for i=2:length(varargin)
        result=pagemtimes(result,varargin{i}); % if version matlab >=2020b
    end
else
    result=mmx('mult',result,varargin{i}); % else, should install mmx
end
end

