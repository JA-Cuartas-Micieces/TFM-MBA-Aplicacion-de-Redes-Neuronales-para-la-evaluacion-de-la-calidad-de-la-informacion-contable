
function nnSelection(name, X_name, y_name, lambda_vec,prop_train,prop_val,np_els, np_ls, np_m, min_hidden_layers,max_hidden_layers,min_hidden_elements,max_hidden_elements,num_iter)

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

%% =========== Part 1: Loading Data =============

num_pts_complex_els=np_els-1;   %Number of hidden layer elements per layer 
num_pts_complex_layers=np_ls-1; %or number of hidden layers intervals are 
num_pts_learn=np_m-1;           %divided by the number of points we want to
                                %plot minus one (but there will possibly be
                                %one addicional point for divisions which  
                                %are not exact).
format long g;
X=xlsread(X_name);
y=xlsread(y_name);
sel = randperm(size(X, 1));
X= X(sel,:);
y=y(sel,:);

m = size(X, 1);

X_train=X(1:ceil(prop_train*m),:);
y_train=y(1:ceil(prop_train*m),:);
X_val=X((ceil(prop_train*m)+1):(ceil(prop_train*m)+floor(prop_val*m)),:);
y_val=y((ceil(prop_train*m)+1):(ceil(prop_train*m)+floor(prop_val*m)),:);
X_test=X((ceil(prop_train*m)+floor(prop_val*m)+1):end,:);
y_test=y((ceil(prop_train*m)+floor(prop_val*m)+1):end,:);

clear X y;
mkdir(num2str(name));

%% =========== Part 2: Validation for Selecting Lambda =============

ncols=max_hidden_elements-min_hidden_elements+1;
nrows=max_hidden_layers-min_hidden_layers+1;

if nrows==1
    list_layers=min_hidden_layers;
else
    list_layers=min_hidden_layers:(floor(nrows/num_pts_complex_layers)):max_hidden_layers;
    ly_len=length(list_layers);
    if list_layers(ly_len)~=max_hidden_layers
        list_layers=[list_layers max_hidden_layers];
    end
end

if ncols==1
    list_els=min_hidden_elements;
else
    list_els=min_hidden_elements:(floor(ncols/num_pts_complex_els)):max_hidden_elements;
    ly_el=length(list_els);
    if list_els(ly_el)~=max_hidden_elements
        list_els=[list_els max_hidden_elements];
    end
end

costsVal=zeros(length(list_layers),length(list_els));
costsTr=zeros(length(list_layers),length(list_els));

qls=1;
for i=list_layers
    qel=1;
    for j=list_els
        layer_sizes=[size(X_train,2) zeros(1,i) size(y_train,2)];
        for r=1:i
            layer_sizes(r+1)=j;
        end
        filename=strcat(num2str(name),'/',num2str(name),'_TriedNNs');
        fprintf('\n%d Hidden Layers, %d Hidden Elements.\n\n', i,j)
        fid1 = fopen(filename,'at');
        fprintf(fid1, '\n%d Hidden Layers, %d Hidden Elements. \n\n', i,j);
        fclose(fid1);
        [error_train, error_val,min_i, min_lambda,min_theta] = ...
            validationCurve(X_train, y_train, X_val, y_val, layer_sizes, lambda_vec,num_iter);
        testSetCheck(i, j, name, min_theta, ...
            layer_sizes, X_train,y_train, X_test,y_test,X_val,y_val,error_val,error_train,min_i,min_lambda, lambda_vec, num_iter);
        filename=strcat(num2str(name),'/',num2str(name),'_Models','/',num2str(i), 'hl',num2str(j), 'hel_',num2str(name),'_ModelAccuracy.mat');
        load(filename);
        if i==min_hidden_layers
            if j==min_hidden_elements
                min_ev=error_val;
                min_il=min_i;
                min_qls=qls;
                min_qel=qel;
            end
        end
        if error_val(min_i)<min_ev(min_il)
            min_ev=error_val;
            min_il=min_i;
            min_qls=qls;
            min_qel=qel;
        end
        costsVal(qls,qel)=error_val(min_i);
        costsTr(qls,qel)=error_train(min_i);
        clear error_train error_val min_i min_lambda min_theta layer_sizes J_test J_train J_val tp tn fp fn p R F1 tpt tnt fpt fnt pt Rt F1t;
        qel=qel+1;
    end
    qls=qls+1;
end


clear layer_sizes min_ev min_il;

filename=strcat(num2str(name),'/',num2str(name),'_Models', '/',num2str(list_layers(min_qls)), 'hl',num2str(list_els(min_qel)), 'hel_',num2str(name),'_ModelAccuracy.mat');
load(filename);
filename=strcat(num2str(name),'/',num2str(list_layers(min_qls)), 'hl',num2str(list_els(min_qel)), 'hel_',num2str(name),'_CostValOptimumModelResults.mat');
save(sprintf('%s',filename), 'error_val', 'error_train', 'min_i', 'min_lambda', 'nn_params', 'layer_sizes', 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'lambda_vec', 'num_iter', 'J_test', 'J_train', 'J_val', 'tp', 'tn', 'fp', 'fn', 'p', 'R', 'F1','tpt','tnt','fpt','fnt','pt','Rt','F1t', 'min_hidden_layers', 'max_hidden_layers', 'min_hidden_elements', 'max_hidden_elements', 'costsVal', 'costsTr');

Validation=figure;
title(sprintf('Validation Curve Minimum Cross-Validation Error Model'))
plot(lambda_vec, error_train, lambda_vec, error_val)
legend('Train', 'Cross Validation')
xlabel('Lambda')
ylabel('Error')
filename=strcat(num2str(name),'_CostValValidationCurve.jpg');
saveas(Validation,filename);

close all;

if ncols >1
    Complexity1=figure;
    title(sprintf('Error vs Complexity: Number of elements in hidden layers'))
    plot(list_els, costsTr(min_qls,:), list_els, costsVal(min_qls,:))
    legend('Train', 'Cross Validation')
    xlabel('Number of elements in each hidden layer')
    ylabel('Error')
    filename=strcat(num2str(name),'_CostValComplexityCurveHiddenElementsNumber.jpg');
    saveas(Complexity1,filename);
end

close all;

if nrows>1
    Complexity2=figure;
    title(sprintf('Error vs Complexity: Number of hidden layers'))
    plot(list_layers, costsTr(:,min_qel)', list_layers, costsVal(:,min_qel)')
    legend('Train', 'Cross Validation')
    xlabel('Number of hidden layers')
    ylabel('Error')
    filename=strcat(num2str(name),'_CostValComplexityCurveHiddenLayersNumber.jpg');
    saveas(Complexity2,filename);
end

close all;


%% =========== Part 3: Learning Curve for Neural Networks =============

clear error_train error_val theta;

[error_train, error_val] = ...
    learningCurve(X_train, y_train, X_val, y_val, min_lambda, layer_sizes,num_iter,num_pts_learn);

m_train=size(X_train,1);

if m_train==1
    list_learning=1;
else
    list_learning=1:floor(m_train/num_pts_learn):m_train;
    ly_learn=length(list_learning);
    if list_learning(ly_learn)~=m_train
        list_learning=[list_learning m_train];
    end
end

LearningCurve=figure;
plot(list_learning, error_train, list_learning, error_val)

if max(error_val)>20*max([(max(error_train)-min(error_val)) (min(error_val)-max(error_train))])
    max_error=3*max(error_train);
else
    max_error=ceil(max([max(error_train), max(error_val)]));
end

ax=[0, m_train, 0, max_error];
axis(ax)
title(sprintf('Learning Curve (lambda = %f)', min_lambda))
xlabel('Number of training examples')
ylabel('Error')
legend('Train', 'Cross Validation')
filename=strcat(num2str(name),'/',num2str(name),'_LearningCurve.jpg');
saveas(LearningCurve,filename);

close all;

end
