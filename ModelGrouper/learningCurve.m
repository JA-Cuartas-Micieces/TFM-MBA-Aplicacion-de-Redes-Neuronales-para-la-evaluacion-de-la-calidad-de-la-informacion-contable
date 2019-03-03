function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda, layer_sizes,num_iter,num_pts)

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


m = size(X, 1);

if m==1
    list_learning=1;
else
    list_learning=1:floor(m/num_pts):m;
    ly_learn=length(list_learning);
    if list_learning(ly_learn)~=m
        list_learning=[list_learning m];
    end
end

error_train = zeros(length(list_learning), 1);
error_val   = zeros(length(list_learning), 1);

nel=1;
for i = list_learning
    theta = trainNN(X(1:i, :), y(1:i,:), lambda, layer_sizes, num_iter);
    error_train(nel) = nnCostFunction(theta, layer_sizes, X(1:i, :), y(1:i,:), 0);
    error_val(nel) = nnCostFunction(theta, layer_sizes, Xval, yval, 0);
    nel=nel+1;
end

end
