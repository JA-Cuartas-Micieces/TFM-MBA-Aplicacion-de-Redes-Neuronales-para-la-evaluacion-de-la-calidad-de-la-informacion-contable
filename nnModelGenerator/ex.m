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

name='1_mdcsf_1l_100el';
X_name='X_md1sf.xls';
y_name='y_md1sf.xls';
lambda_vec = [0 0.001]';
prop_train=0,90;
prop_val=0,05;
np_els=2;
np_ls=2;
np_m=2;
min_hidden_layers=1;
max_hidden_layers=2; 
min_hidden_elements=100;         
max_hidden_elements=200;
num_iter=50;

nnSelection(name, X_name, y_name, lambda_vec, prop_train, prop_val, ...
                np_els, np_ls, np_m, ...
                min_hidden_layers,max_hidden_layers,...
                min_hidden_elements,max_hidden_elements,...
                num_iter);
