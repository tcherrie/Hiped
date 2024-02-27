function color=legend2color(legend)
% for having nice colors associated to vertices of domains
cycleSize = 10;
Ncycle = 4;
defaultColor = [0.3,0.3,0.3];
whiteLabels = ["air", "void"];
blackLabels = ["iron","steel","fe","fesi","feco","feni","smc"];
colorPrefix = ["mag","v","mat","f","n"];

nColor = cycleSize*Ncycle;
cmap = hsv(nColor); % other colormaps are possible : jet, ...
sampleList = (repmat(0:cycleSize-1,Ncycle,1).*Ncycle + (1:Ncycle).').';
sampleList = sampleList(:);
cmap = cmap(sampleList,:);

% RGB triplet of predefined labels

definedColor.ap = [1,0,0]; definedColor.am = [1,0,1];
definedColor.bp = [0,1,0]; definedColor.bm = [1,1,0];
definedColor.cp = [0,0,1]; definedColor.cm = [0,1,1];

%%
legend = convertCharsToStrings(legend);
color = ones(numel(legend),3).*defaultColor;
for i=1:numel(legend)
    l = lower(legend(i));
    if ismember(l,whiteLabels)
        color(i,:) = [1,1,1];
    elseif ismember(l,blackLabels)
        color(i,:) = [0,0,0];
    elseif isfield(definedColor,l)
        color(i,:) = definedColor.(l);
    elseif contains(l,colorPrefix)
        c = convertStringsToChars(l);
        [s,e] = regexp(c,"\d+");
        if ~isempty(s)
            n = str2double(c(s:e));
            color(i,:) = cmap(mod(n-1,nColor)+1,:);
        else
            warning(strcat(l," label color unknown, default color."))
        end
    else
        warning(strcat(l," label color unknown, default color."))
    end
end

color = color - 0.5; % for mixture with the other color.
end