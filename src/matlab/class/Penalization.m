classdef Penalization
     %Penalization Class
    %
    % To create a Penalization object, type
    % obj = Penalization(type,p)
    %
    % - 'type' is a string identifier related to a type of penalization
    % such as "simp", "ramp", etc.
    % - p is the penalization coefficient
    %
    % Penalization are applied to shape functions. They can be useful in
    % the context of topology optimization.
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
        Expression % function handle
        Derivative % function handle
        Type       % str
        Reference  % str
        Coeff      % double
    end

    methods
        function obj = Penalization(varargin)
            % obj = Penalization(type, p)
            % - type is a string identifier refering to a type of
            % penalization (available : "simp", "ramp", "zhu", "lukas", 
            % and some of their concave conterpart)
            % - p is the penalization coefficient

            type = varargin{1};
            if nargin<=1 || isempty(varargin{2})
                obj.Coeff=1;
            else
                obj.Coeff=varargin{2};
            end
            if lower(type)=="simp"
                obj.Expression=@(x,p) real(x.^p);
                obj.Derivative=@(x,p) real(p*x.^(p-1));
                obj.Type=lower(type);
                obj.Reference = "Mlejnek (1992), Bendsøe and Sigmund (1999), Sigmund (2003)";

            elseif lower(type)=="simp_concave" || lower(type)=="simp_inv"
                obj.Expression= @(x,p) 1-(real((1-x).^p));
                obj.Derivative=@(x,p) real(p.*(1 - x).^(p - 1));
                obj.Type=lower(type);
                obj.Reference = "1-simp(1-x,p)";

            elseif lower(type)=="ramp"
                obj.Expression= @(x,p) real(x./(p - p.*x + 1));
                obj.Derivative=@(x,p)real((p + 1)./(p - p.*x + 1).^2);
                obj.Type=lower(type);
                obj.Reference = "Stolpe and Svanberg (2001), Hansen (2005)";

            elseif lower(type)=="ramp_concave" || lower(type)=="ramp_inv"
                obj.Expression= @(x,p) 1-real((1-x)./(p - p.*(1-x) + 1));
                obj.Derivative=@(x,p) real(1./(p.*x + 1)) - real((p.*(x - 1))./(p.*x + 1).^2);
                obj.Type=lower(type);
                obj.Reference = "1-ramp(1-x,p)";

            elseif lower(type)=="lukas"
                obj.Expression= @(x,p) real((1+atan(p*(2*x-1))./atan(p))/2);
                obj.Derivative = @(x,p) real(p./(atan(p).*(p.^2.*(2*x - 1).^2 + 1)));
                obj.Type=lower(type);
                obj.Reference = "Lukáš (2006) - An Integration of Optimal Topology and Shape Design for Magnetostatics";
            else
                error("unknown penalization type")
            end
        end

        function out = eval(obj,x)
            % out = eval(obj,x)
            % evaluate the penalization (x should be in [0,1])
            out = obj.Expression(x,obj.Coeff);
        end

        function out = evald(obj,x)
            % evaluate the penalization derivative (x should be in [0,1])
            out = obj.Derivative(x,obj.Coeff);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Display
        function plot(obj)
            % plot(obj)
            % plot the penalization function in [0,1]
            x=0:0.01:1;
            plot(x,obj.eval(x));
            grid on
            xlabel("\omega")
            ylabel("P(\omega)")
        end
    end
end

