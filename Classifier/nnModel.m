function [h] = nnModel(nn_params, ...
                                   layer_sizes, ...
                                   X)
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
num_layers=size(layer_sizes,2);

%Theta unrolling

Theta=cell(1,num_layers-1);
curr_pos=1;
for i =2:num_layers
    Theta{i-1}=reshape(nn_params(curr_pos:curr_pos+layer_sizes(i)*(layer_sizes(i-1)+1)-1), ...
        layer_sizes(i), (layer_sizes(i-1)+1));
    curr_pos=curr_pos+layer_sizes(i)*(layer_sizes(i-1)+1);
end

%Feed-Forward

a=cell(1,num_layers);
z=cell(1,num_layers);     %First element will be always empty

a1=[ones(size(X,1),1),X];
a{1}=a1;

for i =2:num_layers-1
z{i}=Theta{i-1}*a{i-1}';
a{i}=[ones(size(z{i}',1),1),sigmoid(z{i})'];
end

z{num_layers}=Theta{num_layers-1}*a{num_layers-1}';
a{num_layers}=sigmoid(z{num_layers})';
h=a{num_layers};

end
