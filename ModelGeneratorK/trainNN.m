function [theta] = trainNN(X, y, lambda, layer_sizes, num_iter)

%These scripts were developed taking as the starting point several exercises of
%the online course Machine Learning of Andrew Ng (2017).
%MIT License

%Copyright (c) [2019] [Javier Alejandro Cuartas Micieces]

%Permission is hereby granted, free of charge, to any person obtaining a copy
%of this software and associated documentation files (the "Software"), to deal
%in the Software without restriction, including without limitation the rights
%to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
%copies of the Software, and to permit persons to whom the Software is
%furnished to do so, subject to the following conditions:

%The above copyright notice and this permission notice shall be included in all
%copies or substantial portions of the Software.

%THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
%IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
%FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
%AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
%LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
%OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
%SOFTWARE.

% Initialize Theta

num_layers=size(layer_sizes,2);

initial_Theta=cell(1,num_layers-1);
for i =1:num_layers-1
    initial_Theta{i} = randInitializeWeights(layer_sizes(i), layer_sizes(i+1));
end

initial_nn_params=initial_Theta{1}(:);
for i =2:num_layers-1
    initial_nn_params = [initial_nn_params ; initial_Theta{i}(:)];
end

costFunction = @(t) nnCostFunction(t, layer_sizes, X, y, lambda);

options = optimset('MaxIter', num_iter, 'GradObj', 'on');

theta = fmincg(costFunction, initial_nn_params, options);

end
