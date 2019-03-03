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

name='a';
X_name='X_md1sf.xls';
y_name='y_md1sf.xls';
S_head='EncabezadosFilas.xls';
F_head='EncabezadosColumnas.xls';
lambda_vec = [0.01]';
layer_list=[1];
element_list=[100];
prop_train=1;
prop_val=0;
num_iter=1000;

for i=layer_list
    for j=element_list
        lnnSelection(name, S_head, F_head, X_name, y_name, lambda_vec, prop_train, prop_val, ...
                        length(element_list), length(layer_list), ...
                        i,i,...
                        j,j,...
                        num_iter);
    end
end
