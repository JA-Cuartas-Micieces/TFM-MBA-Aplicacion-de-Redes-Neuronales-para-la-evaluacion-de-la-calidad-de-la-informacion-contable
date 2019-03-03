function [error_train, error_val,min_i, min_lambda,min_theta] = ...
    validationCurve(X, y, Xval, yval, layer_sizes, lambda_vec,num_iter)

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

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);
    theta = trainNN(X, y, lambda, layer_sizes, num_iter);
    error_train(i) = nnCostFunction(theta, layer_sizes, X, y, 0);
    error_val(i) = nnCostFunction(theta, layer_sizes, Xval, yval, 0);
    if i==1
        min_i=i;
        min_lambda=lambda_vec(i);
        min_theta=theta;
        min_error=error_val(i);
    elseif error_val(i)<min_error
        min_i=i;
        min_lambda=lambda_vec(i);
        min_theta=theta;
        min_error=error_val(i);
    end
end

end
