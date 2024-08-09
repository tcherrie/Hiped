classdef VertexFunction
    %VertexFunction Class
    %
    % To create a VertexFunction object, type
    % obj = VertexFunction(expression,derivative,label,dimInput,dimOuput)
    %
    % where 'expression' and 'derivative' are function handles
    %       'label' is a string identifier (default "")
    %       'dimInput' and 'dimOutput' are numerics (default 1)
    %
    % Following operators are overloaded : + - * / ^ .* ./ .^
    %
    % A vertexFunction is a function depending on a variable a (scalar or
    % vector), and that will combined with others in an interpolation that
    % depends on another variable x.
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
        DimInput  = 1 % integer
        DimOutput = 1 % integer
        FlagParExp = false
        FlagParMult = false
        Label = ""
    end

    methods
        function obj = VertexFunction(expression,derivative,label,dimInput,dimOuput)

            %obj = VertexFunction(expression,derivative,dimInput,dimOuput,label)
            % - 'expression' and 'derivative' are function handles
            % please use vectorized operators within them
            % - label is a facultative string identifier
            % - dimInput is a number indicating the dimension of the input
            % (scalar : n=1, if n>1 the input is a vector)
            % - dimOutput is a number indicating the dimension of the output
            % (scalar : n=1, if n>1 the output is a vector)
            % defaut : label = "", dimInput = 1 ; dimOuput = 1
            %
            % examples are given in the "f_VertexFunctionOperations" file

            assert(isa(expression,"function_handle"),"Expression of the function should be a function handle");
            assert(isa(derivative,"function_handle"),"Derivative of the function should be a function handle");
            obj.Expression = expression;
            obj.Derivative = derivative;
            if nargin>=3; obj.Label = label; end
            if nargin>=4
                assert(isa(dimInput,"numeric"),"dimInput should be an integer >= 1");
                obj.DimInput = dimInput;
            end
            if nargin>=5
                assert(isa(dimInput,"numeric"),"dimOuput should be an integer >= 1");
                obj.DimOutput = dimOuput;
            end
        end

        function result = eval(obj,x)
            result = obj.Expression(x);
        end

        function result = evald(obj,x)
            result = obj.Derivative(x);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Operator overloading

        function result = plus(obj1,obj2)
            result=obj1;
            result.FlagParExp = true;
            result.FlagParMult = true;
            if class(obj2)=="double"
                result.Expression=@(x) obj1.Expression(x)+obj2;
                result.Label=strcat(obj1.Label," + ",num2str(obj2));
            elseif class(obj2)=="VertexFunction"
                assert(obj1.DimInput==obj2.DimInput,"Different input dimensions")
                assert(obj1.DimOutput==obj2.DimOutput,"Different output dimensions")
                result.Expression=@(x) obj1.Expression(x) + obj2.Expression(x);
                result.Derivative=@(x) obj1.Derivative(x) + obj2.Derivative(x);
                result.Label = strcat(obj1.Label," + ",obj2.Label);
            end
        end

        function result = minus(obj1,obj2)
            result=obj1;
            result.FlagParExp = true;
            result.FlagParMult = true;
            if class(obj2)=="double"
                result.Expression=@(x) obj1.Expression(x)-obj2;
                result.Label=strcat(obj1.Label," - ",num2str(obj2));
            elseif class(obj2)=="VertexFunction"
                assert(obj1.DimInput==obj2.DimInput,"Different input dimensions")
                assert(obj1.DimOutput==obj2.DimOutput,"Different output dimensions")
                result.Expression=@(x) obj1.Expression(x) - obj2.Expression(x);
                result.Derivative=@(x) obj1.Derivative(x) - obj2.Derivative(x);
                result.Label = strcat(obj1.Label," - ",obj2.Label,")");
            end
        end

        function result = uminus(obj)
            result=obj;
            result.FlagParExp = true;
            result.FlagParMult = false;
            result.Expression=@(x) - obj.Expression(x) ;
            result.Derivative=@(x) - obj.Derivative(x) ;
            result.Label = strcat("-",obj.Label);
        end

        function result = times(obj1,obj2)
            result=obj1;
            dimout1 = obj1.DimOutput;
            if class(obj2)=="double"
                dimout2 = [size(obj2),1];
                result.Expression=@(x) obj1.Expression(x).*obj2;
                result.Derivative = @(x) obj1.Derivative(x)*obj2;
                if result.FlagParMult;  result.Label= strcat("(",obj1.Label,") .* ", num2str(obj2));
                else;  result.Label =  strcat(obj1.Label," .* ", num2str(obj2)); end

            elseif class(obj2)=="VertexFunction"
                dimout2 = [obj2.DimOutput,1];
                result.Expression=@(x) obj1.Expression(x) .* obj2.Expression(x);
                result.Derivative = @(x) obj1.Derivative(x).*obj2.Expression(x) +...
                    obj2.Derivative(x).*obj1.Expression(x);
                if result.FlagParMult && obj2.FlagParMult; result.Label = "(" + obj1.Label + ") .* (" + obj2.Label+ ")";
                elseif result.FlagParMult && ~ obj2.FlagParMult; result.Label = "(" + obj1.Label + ") .* " + obj2.Label;
                elseif ~ result.FlagParMult && obj2.FlagParMult; result.Label =  obj1.Label + " .* (" + obj2.Label + ")";
                else;  result.Label =  obj1.Label + " .* " + obj2.Label; end
            end
            result.DimOutput = size(zeros(dimout1).*zeros(dimout2));
            if result.DimOutput(2) == 1; result.DimOutput = result.DimOutput(1); end
            result.FlagParExp = true;
            result.FlagParMult = false;
        end


        function result = mtimes(obj1,obj2) %  matrix product

            result=obj1;
            if class(obj2)=="double"
                dimOut1 = obj1.DimOutput;
                dimOut2 = size(obj2);
                if length(obj2) ==1
                else
                    result = obj1.*obj2;
                    try
                        assert(dimOut1(2) == dimOut2(1) && length(dimOut1)<=2 && length(dimOut2)<=2, ...
                            "Output dimensions of VertexFunctions don't match to perform matrix-multiplication");
                    catch
                        error("Output dimensions of VertexFunctions don't match to perform matrix-multiplication");
                    end

                    if length(dimOut2) == 1; result.DimOutput = dimOut1(1);
                    elseif length(dimOut2) == 2; result.DimOutput = [dimOut1(1), dimOut2(2)];
                    else ; error("Output dimension higher than 2 are not supported"); end

                    result.Expression=@(x) mult(obj1.Expression(x),obj2);
                    result.Derivative = @(x) mult(t(obj2), obj1.Derivative(x));
                    if result.FlagParMult && obj2.FlagParMult; result.Label = "(" + obj1.Label + ") * (" + num2str(obj2) + ")";
                    elseif result.FlagParMult && ~ obj2.FlagParMult; result.Label = "(" + obj1.Label + ") * " + num2str(obj2);
                    elseif ~ result.FlagParMult && obj2.FlagParMult; result.Label =  obj1.Label + " * (" + num2str(obj2) + ")";
                    else ;  result.Label =  obj1.Label + " * " + str(obj2); end
                end
            elseif class(obj2)=="VertexFunction"
                dimOut1 = obj1.DimOutput;
                dimOut2 = obj2.DimOutput;
                try
                    assert(dimOut1(2) == dimOut2(1) && length(dimOut1)<=2 && length(dimOut2)<=2, ...
                        "Output dimensions of VertexFunctions don't match to perform matrix-multiplication");
                catch
                    error("Output dimensions of VertexFunctions don't match to perform matrix-multiplication");
                end
                if length(dimOut2) == 1; result.DimOutput = dimOut1(1);
                elseif length(dimOut2) == 2; result.DimOutput = [dimOut1(1), dimOut2(2)]; 
                else ; error("Output dimension higher than 2 are not supported"); end
                result.Expression=@(x) mult(t(obj1.Expression(x)),obj2.Expression(x));
                result.Derivative = @(x) mult(t(obj1.Derivative(x)),obj2.Expression(x)) +...
                    mult(t(obj2.Derivative(x)),obj1.Expression(x));
                result.DimOutput = 1;
                if result.FlagParMult && obj2.FlagParMult ; result.Label = "(" + obj1.Label + ") * (" + obj2.Label + ")";
                elseif result.FlagParMult && ~ obj2.FlagParMult ; result.Label = "(" + obj1.Label + ") * " + obj2.Label;
                elseif ~ result.FlagParMult && obj2.FlagParMult ; result.Label =  obj1.Label + " * (" + obj2.Label + ")";
                else ;  result.Label =  obj1.Label + " * " + obj2.Label; end
            end
            result.FlagParExp = true;
            result.FlagParMult = false;
        end

        function result = t(obj)  %transpose, just for "constant" matrices
            result = obj;
            result.Expression = @(x) t(obj.Expression(x));
            result.Derivative =  @(x) t(obj.Derivative(x));
            if result.FlagParMult ; result.Label = "(" + obj.Label + ")^t";
            else ;  result.Label =  obj.Label + "^t"; end
            result.FlagParExp = true;
            result.FlagParMult = false;
            end


        function result = innerProduct(obj1,obj2)
            dimOut1 = obj1.DimOutput;
            result = obj1;
            if class(obj2) == "double"
                dimOut2 = size(obj2);
                result.Expression = @ (x) mult(t(obj1.Expression(x)), obj2);
                result.Derivative =  @ (x) mult(t(obj1.Derivative(x)), obj2);
                if result.FlagParMult && obj2.FlagParMult ; result.Label = "<(" + str(obj2) +  ") , (" + obj1.Label + ")>";
                elseif result.FlagParMult && ~ obj2.FlagParMult ; result.Label = "<(" + str(obj2) + ") , " + obj1.Label + ">";
                elseif ~ result.FlagParMult && obj2.FlagParMult ; result.Label = "<" + str(obj2) + " , (" + obj1.Label + ")>";
                else ;  result.Label =  "<" + str(obj2) + " , " + obj1.Label + ">"; end

            elseif class(obj2) == "VertexFunction"
                dimOut2 = obj2.DimOutput;
                result.Expression = @ (x) mult(t(obj1.Expression(x)), obj2.Expression(x));
                result.Derivative =  @ (x) mult(t(obj2.Expression(x)), obj1.Derivative(x)) + mult(t(obj1.Expression(x)), obj2.Derivative(x));
                if result.FlagParMult && obj2.FlagParMult ; result.Label = "<(" + obj1.Label +  ") , (" +  obj2.Label + ")>";
                elseif result.FlagParMult && ~ obj2.FlagParMult ; result.Label = "<(" + obj1.Label + ") , " + obj2.Label + ">";
                elseif ~ result.FlagParMult && obj2.FlagParMult ; result.Label = "<" + obj1.Label + " , (" + obj2.Label + ")>";
                else ;  result.Label =  "<" + obj1.Label + " , " + obj2.Label + ">"; end
            end
            
        result.DimOutput = 1;
        result.FlagParExp = true;
        result.FlagParMult = false;
        end

        function result = rdivide(obj1,obj2)
            result=obj1;
            if class(obj2)=="double"
                result.Expression=@(x) obj1.Expression(x)./obj2;
                result.Derivative = @(x) obj1.Derivative(x)./obj2;
                if result.FlagParMult; result.Label =  "(" + obj1.Label + ") ./ " + num2str(obj2);
                else;  result.Label = obj1.Label+ " ./ " + num2str(obj2); end
            elseif class(obj2)=="VertexFunction"
                result.Expression=@(x) obj1.Expression(x) ./ obj2.Expression(x);
                result.Derivative = @(x)  (obj1.Derivative(x).*obj2.Expression(x)-...
                    obj2.Derivative(x).*obj1.Expression(x))./((obj2.Expression(x)).^2);
                if result.FlagParMult && obj2.FlagParMult ; result.Label = "(" + obj1.Label + ") ./ (" + obj2.Label + ")";
                elseif result.FlagParMult && ~ obj2.FlagParMult ; result.Label = "(" + obj1.Label + ") ./ " + obj2.Label;
                elseif ~ result.FlagParMult && obj2.FlagParMult ; result.Label =  obj1.Label + " ./ (" + obj2.Label + ")";
                else ;  result.Label =  obj1.Label + " ./ " + obj2.Label; end
            end
            result.FlagParExp = true;
            result.FlagParMult = false;
        end

        function result = mrdivide(obj1,obj2)
            result=obj1;
            if class(obj2)=="double"
                result.Expression=@(x) obj1.Expression(x)/obj2;
                result.Derivative = @(x) obj1.Derivative(x)/obj2;
                if result.FlagParMult; result.Label =  "(" + obj1.Label + ") / " + num2str(obj2);
                else;  result.Label = obj1.Label+ " / " + num2str(obj2); end
            elseif class(obj2)=="VertexFunction"
                result.Expression=@(x) obj1.Expression(x) /  obj2.Expression(x);
                result.Derivative = @(x)  (obj1.Derivative(x)*obj2.Expression(x)-...
                    obj2.Derivative(x)*obj1.Expression(x))/((obj2.Expression(x)).^2);
                if result.FlagParMult && obj2.FlagParMult ; result.Label = "(" + obj1.Label + ") / (" + obj2.Label + ")";
                elseif result.FlagParMult && ~ obj2.FlagParMult ; result.Label = "(" + obj1.Label + ") / " + obj2.Label;
                elseif ~ result.FlagParMult && obj2.FlagParMult ; result.Label =  obj1.Label + " / (" + obj2.Label + ")";
                else ;  result.Label =  obj1.Label + " / " + obj2.Label; end
            end
            result.FlagParExp = true;
            result.FlagParMult = false;
        end

        function result = power(obj1,obj2)
            result=obj1;
            if class(obj2)=="double"
                result.Expression=@(x) obj1.Expression(x).^obj2;
                result.Derivative = @(x) obj1.Derivative(x).* obj2.*(obj1.Expression(x)).^(obj2-1);
                if result.FlagParExp; result.Label =  "(" + obj1.Label + ") .^ " + num2str(obj2);
                else;  result.Label = obj1.Label+ " .^ " + num2str(obj2); end
            elseif class(obj2)=="VertexFunction"
                result.Expression=@(x) obj1.Expression(x) .^ (obj2.Expression(x));
                result.Derivative = @(x) obj1.Derivative(x).*obj2.Expression(x).*...
                    obj1.Expression(x).^(obj2.Expression(x)-1)+...
                    log(obj1.Expression(x)).*obj2.Derivative(x).*...
                    obj1.Expression(x).^(obj2.Expression(x));
                if result.FlagParExp && obj2.FlagParExp ; result.Label = "(" + obj1.Label + ") .^ (" + obj2.Label + ")";
                elseif result.FlagParExp && ~ obj2.FlagParExp ; result.Label = "(" + obj1.Label + ") .^ " + obj2.Label;
                elseif ~ result.FlagParExp && obj2.FlagParExp ; result.Label =  obj1.Label + " .^ (" + obj2.Label + ")";
                else ;  result.Label =  obj1.Label + " .^ " + obj2.Label; end
            end
            result.FlagParExp = false;
            result.FlagParMult = false;
        end

        function result = mpower(obj1,obj2)
            result=obj1;
            if class(obj2)=="double"
                result.Expression=@(x) obj1.Expression(x)^obj2;
                result.Derivative = @(x) obj1.Derivative(x)* obj2*obj1.Expression(x)^(obj2-1);
                if result.FlagParExp; result.Label =  "(" + obj1.Label + ") ^ " + num2str(obj2);
                else;  result.Label = obj1.Label+ " ^ " + num2str(obj2); end
            elseif class(obj2)=="VertexFunction"
                result.Expression=@(x) obj1.Expression(x) ^ ...
                    (obj2.Expression(x));
                result.Derivative = @(x) obj1.Derivative(x)*obj2.Expression(x)*...
                    obj1.Expression(x)^(obj2.Expression(x)-1)+...
                    log(obj1.Expression(x))*obj2.Derivative(x)*...
                    obj1.Expression(x)^(obj2.Expression(x));
                if result.FlagParExp && obj2.FlagParExp ; result.Label = "(" + obj1.Label + ") ^ (" + obj2.Label + ")";
                elseif result.FlagParExp && ~ obj2.FlagParExp ; result.Label = "(" + obj1.Label + ") ^ " + obj2.Label;
                elseif ~ result.FlagParExp && obj2.FlagParExp ; result.Label =  obj1.Label + " ^ (" + obj2.Label + ")";
                else ;  result.Label =  obj1.Label + " ^ " + obj2.Label; end
            end
            result.FlagParExp = false;
            result.FlagParMult = false;
        end
        

        function plot(obj, xmin, xmax, nPoints)
            if nargin < 2 || isempty(xmin), xmin = -1; end
            if nargin < 3 || isempty(xmax), xmax = 1; end
            if nargin < 4 || isempty(nPoints), nPoints = 30; end

            dimOut = [obj.DimOutput, 1];

            if obj.DimInput == 1
                x = linspace(xmax, xmin, nPoints);
                x = reshape(x, [1, 1, length(x)]);
                out = obj.eval(x);
                for i = 1:dimOut(1)
                    for j = 1:dimOut(2)
                        plot(squeeze(x), squeeze(out(i, j, :)), 'DisplayName', sprintf('f_{%d,%d}', i, j));
                        hold on;
                    end
                end
                xlabel('u');
                ylabel('f(u)');
                legend('Location', 'best');

            elseif obj.DimInput == 2
                X = linspace(xmax, xmin, nPoints);
                Y = linspace(xmax, xmin, nPoints);
                [X, Y] = meshgrid(X, Y);
                X = reshape(X, [1, 1, numel(X)]);
                Y = reshape(Y, [1, 1, numel(Y)]);
                xy = [X; Y];
                out = obj.eval(xy);
                figure;
                nn = dimOut(1)*dimOut(2);
                c = hsv(256);
                for i = 1:dimOut(1)
                    for j = 1:dimOut(2)
                        surf(reshape(X, nPoints, nPoints), reshape(Y, nPoints, nPoints), ...
                            reshape(out(i, j, :), nPoints, nPoints), 'Edgealpha', 0.5, 'Facealpha',0.7, ...
                            "FaceColor",c(round(255*((i-1)*(dimOut(2))+j-1)/(nn))+1,:), ...
                            "DisplayName", sprintf('f_{%d,%d}', i, j));
                         hold on;
                    end
                end
                xlabel('u_0'); ylabel('u_1'); zlabel('f(u)');
            else
                % TODO: Add handling for other dimensions if needed
            end
            legend('Location', 'best');
            grid on;
            hold off;
        end

    end
end