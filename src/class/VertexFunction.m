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

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Operator overloading

        function result = plus(obj1,obj2)
            result=obj1;
            if class(obj2)=="double"
                result.Expression=@(x) obj1.Expression(x)+obj2;
                result.Label=strcat(obj1.Label,"+",num2str(obj2));
            elseif class(obj2)=="VertexFunction"
                assert(obj1.DimInput==obj2.DimInput,"Different input dimensions")
                assert(obj1.DimOutput==obj2.DimOutput,"Different output dimensions")
                result.Expression=@(x) obj1.Expression(x) + obj2.Expression(x);
                result.Derivative=@(x) obj1.Derivative(x) + obj2.Derivative(x);
                result.Label = strcat(obj1.Label,"+",obj2.Label);
            end
        end

        function result = minus(obj1,obj2)
            result=obj1;
            if class(obj2)=="double"
                result.Expression=@(x) obj1.Expression(x)-obj2;
                result.Label=strcat(obj1.Label,"-",num2str(obj2));
            elseif class(obj2)=="VertexFunction"
                assert(obj1.DimInput==obj2.DimInput,"Different input dimensions")
                assert(obj1.DimOutput==obj2.DimOutput,"Different output dimensions")
                result.Expression=@(x) obj1.Expression(x) - obj2.Expression(x);
                result.Derivative=@(x) obj1.Derivative(x) - obj2.Derivative(x);
                result.Label = strcat("(",obj1.Label,")-(",obj2.Label,")");
            end
        end

        function result = uminus(obj)
            result=obj;
            result.Expression=@(x) - obj.Expression(x) ;
            result.Derivative=@(x) - obj.Derivative(x) ;
            result.Label = strcat("-",obj.Label);
        end

        function result = times(obj1,obj2)
            result=obj1;
            if class(obj2)=="double"
                result.Expression=@(x) obj1.Expression(x).*obj2;
                result.Derivative = @(x) obj1.Derivative(x)*obj2;
                result.Label=strcat(num2str(obj2),"*(",obj1.Label,")");
            elseif class(obj2)=="VertexFunction"
                result.Expression=@(x) obj1.Expression(x) .* obj2.Expression(x);
                result.Derivative = @(x) obj1.Derivative(x).*obj2.Expression(x) +...
                    obj2.Derivative(x).*obj1.Expression(x);
                result.Label=strcat("(",obj1.Label,").*(",obj2.Label,")");
            end
        end


        function result = mtimes(obj1,obj2)
            result=obj1; % inner product
            if class(obj2)=="double"
                result.Expression=@(x) obj1.Expression(x)*obj2;
                result.Derivative = @(x) obj1.Derivative(x)*obj2;
                result.Label=strcat(num2str(obj2),"*(",obj1.Label,")");
            elseif class(obj2)=="VertexFunction"
                result.Expression=@(x) mult(t(obj1.Expression(x)),obj2.Expression(x));
                result.Derivative = @(x) mult(t(obj1.Derivative(x)),obj2.Expression(x)) +...
                    mult(t(obj2.Derivative(x)),obj1.Expression(x));
                result.DimOutput = 1;
                result.Label=strcat("<",obj1.Label,",",obj2.Label,">");
            end
        end

        function result = innerProduct(obj1,obj2)
            result = mtimes(obj1,obj2);
        end

        function result = rdivide(obj1,obj2)
            result=obj1;
            if class(obj2)=="double"
                result.Expression=@(x) obj1.Expression(x)./obj2;
                result.Derivative = @(x) obj1.Derivative(x)./obj2;
                result.Label=strcat("(",obj1.Label,")/",num2str(obj2));
            elseif class(obj2)=="VertexFunction"
                result.Expression=@(x) obj1.Expression(x) ./ obj2.Expression(x);

                result.Derivative = @(x)  (obj1.Derivative(x).*obj2.Expression(x)-...
                    obj2.Derivative(x).*obj1.Expression(x))./((obj2.Expression(x)).^2);
                result.Label=strcat("(",obj1.Label,")./(",obj2.Label,")");
            end
        end

        function result = mrdivide(obj1,obj2)
            result=obj1;
            if class(obj2)=="double"
                result.Expression=@(x) obj1.Expression(x)/obj2;
                result.Derivative = @(x) obj1.Derivative(x)/obj2;
                result.Label=strcat("(",obj1.Label,")/",num2str(obj2));
            elseif class(obj2)=="VertexFunction"
                result.Expression=@(x) obj1.Expression(x) /  obj2.Expression(x);
                result.Derivative = @(x)  (obj1.Derivative(x)*obj2.Expression(x)-...
                    obj2.Derivative(x)*obj1.Expression(x))/((obj2.Expression(x)).^2);
                result.Label=strcat("(",obj1.Label,")/(",obj2.Label,")");
            end

        end

        function result = power(obj1,obj2)
            result=obj1;
            if class(obj2)=="double"
                result.Expression=@(x) obj1.Expression(x).^obj2;
                result.Derivative = @(x) obj1.Derivative(x).* obj2.*(obj1.Expression(x)).^(obj2-1);
                result.Label=strcat("(",obj1.Label,").^",num2str(obj2));
            elseif class(obj2)=="VertexFunction"
                result.Expression=@(x) obj1.Expression(x) .^ (obj2.Expression(x));
                result.Derivative = @(x) obj1.Derivative(x).*obj2.Expression(x).*...
                    obj1.Expression(x).^(obj2.Expression(x)-1)+...
                    log(obj1.Expression(x)).*obj2.Derivative(x).*...
                    obj1.Expression(x).^(obj2.Expression(x));
                result.Label=strcat("(",obj1.Label,").^(",obj2.Label,")");
            end
        end

        function result = mpower(obj1,obj2)
            result=obj1;
            if class(obj2)=="double"
                result.Expression=@(x) obj1.Expression(x)^obj2;
                result.Derivative = @(x) obj1.Derivative(x)* obj2*obj1.Expression(x)^(obj2-1);
                result.Label=strcat("(",obj1.Label,")^",num2str(obj2));
            elseif class(obj2)=="VertexFunction"
                result.Expression=@(x) obj1.Expression(x) ^ ...
                    (obj2.Expression(x));
                result.Derivative = @(x) obj1.Derivative(x)*obj2.Expression(x)*...
                    obj1.Expression(x)^(obj2.Expression(x)-1)+...
                    log(obj1.Expression(x))*obj2.Derivative(x)*...
                    obj1.Expression(x)^(obj2.Expression(x));
                result.Label=strcat("(",obj1.Label,")^(",obj2.Label,")");
            end
        end
    end
end