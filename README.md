



PROJECT




The project was developed for the final master thesis “Application of Neural 
Networks in the Quality Evaluation of Accounting Information” for MBA in the 
Cantabria University (UC) so It is a personal project (you can find it in: 
https://bit.ly/2BUYNnx). 

It is not open for contributions at the moment but please, feel free to share 
any comments, questions or suggestion.

This is a Matlab implementation of a supervised classification tool based on 
neural networks. It was made to be applied to accounting information, and It 
was split into different versions or modules, each one with its own goal.




MODULES




nnModelGenerator (initial software which needs the following files to work)

ex: It calls main functions using the given arguments as inputs.

nnSelection.m: It loads data and executes validationCurve.m and testSetCheck.m 
for all of the posible combinations according to the input arguments and then, 
It prints the proper graphs (lambda, learning curve).

validationCurve.m: It trains the network and returns Theta vector, training and 
validation costs for the best model. It tries the lambda values provided as 
arguments and  returns the information for the minimum cost model.

testSetCheck.m: It classifies the training and test sets and outputs a text file 
with the performance information for the model and its main parameters.

nnCostFunction.m: It output the cost and the gradient for the given arguments. 

learningCurve.m: It makes a list of training and validation costs for models 
trained with a growing number of training cases to make it posible to print the 
learning curve.

trainNN.m: It calls randInitializeWeights.m and it executes optimization function 
using nnCostFunction as the input to get Theta.

nnModel.m: It classifies new cases taking into account Theta, X and the number of 
layers of the neural network.

randInitializeWeights.m: It randomly initializes the weights in Theta to break 
symmetry. 

sigmoid.m: It executes sigmoid function.

sigmoidGradient.m: It executes the sigmoid function gradient which will be used 
in the Backpropagation step.

fmincg.m: Optimization function. Copyright of Carl Edward Rasmussen (2002).



nnModelGeneratorK (It is a different version from the previous module. In this 
case, It trains the model for a unique number of layers and number of neurons per 
layer, and It outputs the labeled training, validation and test sets).



nnCostFunctionChecker (It is based on the module from Machine Learning course in 
Coursera of Andrew Ng to check the proper implementation of the cost function, 
applied to this new version).



nnModelGenerator_List (It is the same as nnModelGenerator but for specific hidden 
layer numbers and sizes).



nnModelGenerator_YearlyChain (The same as nnModelGenerator but for data of several 
years split in different files).



ModelGrouper (It groups models generated with nnModelGenerator_List taking the 
best model and drawing the same graphs as nnModelGenerator).



Graphprinter (It prints graphs in 2 and 3D with the information of the file 
displayed with cases in rows and variables in columns).



Classifier (ex.m takes a .mat and a .xls files as inputs, with the model and X, to 
classify the row cases in the .xls file).




CONTACT AND CONTRIBUTION




It is a personal project which is not open for contributions at the moment but 
please, feel free to share any comments, questions or suggestions through 
javiercuartasmicieces@hotmail.com.




ACKNOWLEDGEMENTS




These scripts were developed taking as the starting point several exercises of the 
online course Machine Learning of Andrew Ng in Coursera.

The optimization algorithm is fmincg.m with copyright Carl Edward Rasmussen (2002).

