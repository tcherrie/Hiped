function tM=t(M)
%tM=t(M)
%
% Shortcut for pagetranspose.

v = version; [s,e] = regexp(v,'20\d{2}\w'); v = v(s:e);
if str2double(v(1:4))>2020 || (str2double(v(1:4))==2020 && v(5) == "a")
    tM=pagetranspose(M); % if version Matlab >= R2020B
else
tM=permute(M,[2 1 3 4 5 6 7 8 9]); % else
end