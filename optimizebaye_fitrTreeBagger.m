function [Mdl]  = optimizebaye_fitrTreeBagger(train_x_feature_label_norm,train_y_feature_label_norm,vaild_x_feature_label_norm,vaild_y_feature_label_norm,max_iter)    
    maxMinLS = 20;   %超参数选取
    tree_range=[10,300];
    tree_num = optimizableVariable('tree_num',tree_range,'Type','integer');
    minLS = optimizableVariable('minLS',[1,maxMinLS],'Type','integer');
    numPTS = optimizableVariable('numPTS',[1,size(train_x_feature_label_norm,2)],'Type','integer');
    hyperparametersRF = [minLS; numPTS;tree_num];
    results = bayesopt(@(params)oobErrRF(params,train_x_feature_label_norm,train_y_feature_label_norm,vaild_x_feature_label_norm,vaild_y_feature_label_norm),hyperparametersRF,...
    'AcquisitionFunctionName','expected-improvement-plus','Verbose',0,...
    'MaxObjectiveEvaluations',max_iter);
    bestHyperparameters = results.XAtMinObjective;
  
    Mdl = TreeBagger(bestHyperparameters.tree_num,train_x_feature_label_norm,train_y_feature_label_norm,'Method','regression',...
    'MinLeafSize',bestHyperparameters.minLS,...
    'NumPredictorstoSample',bestHyperparameters.numPTS);
     disp(['贝叶斯', '优化 TreeBagger:   ',"Tree_Num:",num2str(bestHyperparameters.tree_num),'   MinLeafSize: ',num2str(bestHyperparameters.minLS),'   NumPredictorstoSample: ',num2str(bestHyperparameters.numPTS)]) 
end
function [oobErr] = oobErrRF(params,train_x_feature_label_norm,train_y_feature_label_norm,vaild_x_feature_label_norm,vaild_y_feature_label_norm)

    Mdl = TreeBagger(params.tree_num,train_x_feature_label_norm,train_y_feature_label_norm,'Method','regression',...
    'MinLeafSize',params.minLS,...
    'NumPredictorstoSample',params.numPTS);
     P_vaild_y_feature_label_norm=predict(Mdl,vaild_x_feature_label_norm);
    oobErr=sum(abs(P_vaild_y_feature_label_norm-vaild_y_feature_label_norm))/length(vaild_y_feature_label_norm);
end