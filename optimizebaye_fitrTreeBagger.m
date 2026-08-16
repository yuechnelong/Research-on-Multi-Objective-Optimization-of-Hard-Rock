function [Mdl, conv_trace] = optimizebaye_fitrTreeBagger(train_x, train_y, valid_x, valid_y, max_iter)
% OPTIMIZEBAYE_FITRTREEBAGGER  Bayesian hyperparameter optimisation for a
% Random Forest (TreeBagger) regression model (following the original code).
%
%   [Mdl, conv_trace] = optimizebaye_fitrTreeBagger(train_x, train_y, ...
%       valid_x, valid_y, max_iter)
%
%   Optimises three hyperparameters:
%       tree_num  in [10, 300]
%       minLS     in [1, 20]
%       numPTS    in [1, n_features]
%   by minimising the validation MAE with bayesopt. Returns the model
%   trained with the best hyperparameters and the convergence trace
%   (minimum objective observed so far, one value per evaluation).

    maxMinLS = 20;
    tree_range = [10, 300];
    tree_num = optimizableVariable('tree_num', tree_range, 'Type', 'integer');
    minLS = optimizableVariable('minLS', [1, maxMinLS], 'Type', 'integer');
    numPTS = optimizableVariable('numPTS', [1, size(train_x, 2)], 'Type', 'integer');
    hyperparametersRF = [minLS; numPTS; tree_num];

    results = bayesopt(@(params)oobErrRF(params, train_x, train_y, valid_x, valid_y), ...
        hyperparametersRF, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', 'Verbose', 0, ...
        'MaxObjectiveEvaluations', max_iter);

    bestHyperparameters = results.XAtMinObjective;

    Mdl = TreeBagger(bestHyperparameters.tree_num, train_x, train_y, 'Method', 'regression', ...
        'MinLeafSize', bestHyperparameters.minLS, ...
        'NumPredictorsToSample', bestHyperparameters.numPTS);

    conv_trace = results.ObjectiveMinimumTrace;

    fprintf('Bayesian optimisation of TreeBagger: Tree_Num=%d, MinLeafSize=%d, NumPredictorsToSample=%d\n', ...
        bestHyperparameters.tree_num, bestHyperparameters.minLS, bestHyperparameters.numPTS);
end

function oobErr = oobErrRF(params, train_x, train_y, valid_x, valid_y)
% Validation MAE of a TreeBagger trained with the given hyperparameters.
    Mdl = TreeBagger(params.tree_num, train_x, train_y, 'Method', 'regression', ...
        'MinLeafSize', params.minLS, ...
        'NumPredictorsToSample', params.numPTS);
    P_valid = predict(Mdl, valid_x);
    oobErr = sum(abs(P_valid - valid_y)) / length(valid_y);
end
