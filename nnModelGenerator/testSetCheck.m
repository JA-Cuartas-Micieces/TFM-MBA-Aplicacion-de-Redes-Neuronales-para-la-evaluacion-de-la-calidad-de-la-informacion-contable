function testSetCheck(i, j, name, ...
                                   nn_params, ...
                                   layer_sizes, ...
                                   X_train,y_train,X_test,y_test,X_val,y_val,...
                                   error_val,error_train,min_i, min_lambda, lambda_vec, num_iter)
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

limit=0.5;
haux = nnModel(nn_params, layer_sizes, X_test);
h=zeros(size(haux));
for e=1:size(h,1)
    for f=1:size(h,2)
        if haux(e,f)>limit
            h(e,f)=1;
        else
            h(e,f)=0;
        end
    end
end

num_el=numel(y_test);

tp=(sum(sum((y_test & h)/num_el)));
fp=((sum(sum(h))-sum(sum((y_test & h))))/num_el);
tn=(sum(sum((h==0)&(y_test==0)))/num_el);
fn=((num_el-(tn+fp+tp)*num_el)/num_el);

p=tp/(tp+fp);
R=tp/(tp+fn);
F1=2*((p*R)/(p+R));

hauxt = nnModel(nn_params, layer_sizes, X_train);
ht=zeros(size(hauxt));
for e=1:size(ht,1)
    for f=1:size(ht,2)
        if hauxt(e,f)>limit
            ht(e,f)=1;
        else
            ht(e,f)=0;
        end
    end
end

num_elt=numel(y_train);

tpt=(sum(sum(((y_train & ht)))/num_elt));
fpt=((sum(sum(ht))-sum(sum((y_train & ht))))/num_elt);
tnt=(sum(sum((ht==0)&(y_train==0)))/num_elt);
fnt=((num_elt-(tnt+fpt+tpt)*num_elt)/num_elt);

pt=tpt/(tpt+fpt);
Rt=tpt/(tpt+fnt);
F1t=2*((pt*Rt)/(pt+Rt));

J_test=nnCostFunction(nn_params, layer_sizes, X_test, y_test,min_lambda);
J_val=error_val(min_i);
J_train=error_train(min_i);

mkdir(strcat(num2str(name),'/',num2str(name),'_Models'));
filename=strcat(num2str(name),'/',num2str(name),'_Models','/',num2str(i), 'hl',num2str(j), 'hel_',num2str(name),'_testResults.txt');
fid = fopen(filename,'wt');
fprintf(fid, '=========== Resultados en el conjunto de prueba===========\n\n');
fprintf(fid, 'Red neuronal de %d capas ocultas y %d elementos en cada capa\n oculta. \n\n',length(layer_sizes)-2,layer_sizes(2));
fprintf(fid, 'Lambda %d. \n\n',min_lambda);

fprintf(fid, 'Resultados sobre el conjunto de prueba. \n\n');
fprintf(fid, '                                 Realidad    \n');
fprintf(fid, '                          1                   0  \n\n');
fprintf(fid, '   Simulación: 1     %12f          %12f  \n',tp,fp);
fprintf(fid, '   Simulación: 0     %12f          %12f  \n\n',fn,tn);
fprintf(fid, 'Accurancy: %f.\n',(tp+tn)/(tp+tn+fp+fn));
fprintf(fid, 'Precision: %f.\n',p);
fprintf(fid, 'Recall:    %f.\n',R);
fprintf(fid, 'F1:        %f.\n\n',F1);

fprintf(fid, 'Resultados sobre el conjunto de entrenamiento. \n\n');
fprintf(fid, '                                 Realidad    \n');
fprintf(fid, '                          1                   0  \n\n');
fprintf(fid, '   Simulación: 1     %12f          %12f  \n',tpt,fpt);
fprintf(fid, '   Simulación: 0     %12f          %12f  \n\n',fnt,tnt);
fprintf(fid, 'Accurancy: %f.\n',(tpt+tnt)/(tpt+tnt+fpt+fnt));
fprintf(fid, 'Precision: %f.\n',pt);
fprintf(fid, 'Recall:    %f.\n',Rt);
fprintf(fid, 'F1:        %f.\n\n',F1t);

fprintf(fid, 'Coste en el conjunto de entrenamiento: %d.\n',J_train);
fprintf(fid, 'Coste en el conjunto de validación:    %d.\n',J_val);
fprintf(fid, 'Coste en el conjunto de prueba:        %d.\n',J_test);

fclose(fid);

filename=strcat(num2str(name),'/',num2str(name),'_Models','/',num2str(i), 'hl',num2str(j), 'hel_',num2str(name),'_ModelAccuracy.mat');
save(sprintf('%s',filename), 'error_val', 'error_train', 'min_i', 'min_lambda', 'nn_params', 'layer_sizes', 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'lambda_vec', 'num_iter', 'J_test', 'J_train', 'J_val', 'tp', 'tn', 'fp', 'fn', 'p', 'R', 'F1','tpt','tnt','fpt','fnt','pt','Rt','F1t');

end
