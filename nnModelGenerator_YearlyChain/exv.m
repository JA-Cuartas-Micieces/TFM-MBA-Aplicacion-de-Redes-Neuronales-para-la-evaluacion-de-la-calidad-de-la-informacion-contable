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

%% Initialization
clear ; close all; clc

lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
prop_train=0.3;
prop_val=0.4;
np_els=5;
np_ls=5;
np_m=10;
min_hidden_layers=1;
max_hidden_layers=10; 
min_hidden_elements=100;         
max_hidden_elements=600;
num_iter=50;
    
for i=2012:1:2014,
    
    name=strcat(num2str(i),'_aac_1-10l_100-600el');
    X_name=strcat('x',num2str(i),'.xls');
    y_name=strcat('y',num2str(i),'.xls');

    nnSelection(name, X_name, y_name, lambda_vec, prop_train, prop_val, ...
                    np_els, np_ls, np_m, ...
                    min_hidden_layers,max_hidden_layers,...
                    min_hidden_elements,max_hidden_elements,...
                    num_iter);
                
end
