clc;clear;close all;	
load('MT_08_Jun_2026_13_25_04.mat')	
random_seed=G_out_data.random_seed;	
rng(random_seed)	
	
data_str=G_out_data.data_path_str ;  %读取数据的路径 	
dataO1=readtable(data_str,'VariableNamingRule','preserve'); %读取数据 	
data1=dataO1(:,2:end);test_data=table2cell(dataO1(1,2:end));	
for i=1:length(test_data)	
      if ischar(test_data{1,i})==1	
          index_la(i)=1;     %char类型	
      elseif isnumeric(test_data{1,i})==1	
          index_la(i)=2;     %double类型	
      else	
        index_la(i)=0;     %其他类型	
    end 	
end	
index_char=find(index_la==1);index_double=find(index_la==2);	
	
%%数值类型数据处理	
if length(index_double)>=1	
     data_numshuju=table2array(data1(:,index_double));	
     data_numshuju2=data_numshuju;	
     index_need_last=index_double;	
else	
     index_need_last=index_double;	
    data_numshuju2=[];	
end	
	
% 文本类型数据处理	
data_shuju=[];	
if length(index_char)>=1	
   for j=1:length(index_char)	
     data_get=table2array(data1(:,index_char(j)));	
	
     data_label=unique(data_get);	
     for NN=1:length(data_label)	
         idx = find(ismember(data_get,data_label{NN,1}));	
         data_shuju(idx,j)=NN;	
     end	
   end	
end	
data_all_last=[data_shuju,data_numshuju2];	
label_all_last=[index_char,index_need_last];	
data=data_all_last;	
     data_biao_all=data1.Properties.VariableNames;	
for j=1:length(label_all_last)	
     data_biao{1,j}=data_biao_all{1,label_all_last(j)};	
end	
	

    %% 
	
dataO=[];	
data_numshuju=data;	
for NN=1:size(data_numshuju,2)	
      data_test=data_numshuju(:,NN);	
      index=isnan(data_test);	
      data_test1=data_test;	
      data_test1(index)=[];	
      index_label=1:length(data_test);	
      index_label1=index_label;	
      index_label1(index)=[];	
     data_all=interp1(index_label1,data_test1,index_label,'spline');	
	
     dataO(:,NN)=data_all;	
end	
	
A_data1=dataO;	
data_biao1=data_biao;	
select_feature_num=G_out_data.select_feature_num;   %特征选择的个数	
predict_num=G_out_data.predict_num_set;   %预测的点个数	
index_name=data_biao1;	
print_index_name=[]; 	
RF_Model = TreeBagger(50,A_data1(:,1:end-predict_num),A_data1(:,end-predict_num+1),'Method','regression','OOBPredictorImportance','on'); 	
imp = RF_Model.OOBPermutedPredictorDeltaError;	
y_index = index_name; x_index=index_name;  	
	
	
	
[sort_value,sort_feature]=sort(imp,'descend'); 	
index_name=data_biao1; 	
feature_need_last=sort_feature(1:select_feature_num);	
for NN=1:length(feature_need_last) 	
   print_index_name{1,NN}=index_name{1,feature_need_last(NN)};	
end 	
disp('选择特征');disp(print_index_name)  	
data_select=[A_data1(:,feature_need_last),A_data1(:,end-predict_num+1:end)];  %经过特征选择后的数据 	
	
	
	
figure;	
bar_plot_f=bar(imp);    %  重要性衡量	
bar_plot_f.FaceColor = 'flat';	
color_get=G_out_data.color_get;  %颜色数据	
for i=1:length(imp)	
    bar_plot_f.CData(i,:)=[color_get(1+i*(floor(length(color_get)/length(imp))-1),:)];	
end	
index_name_plot=data_biao1(1:end-1);	
title('Curvature Test'); ylabel('Predictor importance estimates');  xlabel('Predictors');	
xticks([1:length(imp)])	
xticklabels(index_name_plot)	
	
for NN1=1:length(sort_value)	
     feature_name{NN1,1}=index_name{1,sort_feature(NN1)}; 	
end	
 	
feature_value=sort_value'; 	
print_index_table=table(feature_name,feature_value);	
disp('特征重要性排序') 	
disp(print_index_table) 	
	
data_select1=data_select;	
data_select1=data_select1;	
	
    %% 
	
	
%%波形分解	
data_select1_cell=[];	
data_select1_cell{1,1}=data_select1;	
	
	
 % 模型训练参数	
select_predict_num=G_out_data.select_predict_num;  %待预测数	
num_feature=G_out_data.num_feature;    %特征选择量	
num_series=G_out_data.num_series;     %序列选择	
num_input_serise=num_series;     	
min_batchsize=G_out_data.min_batchsize; 	
roll_num=G_out_data.roll_num;    %滚动次数	
roll_num_in=G_out_data.roll_num_in;	
num_pop=5;  %优化种群数	
num_iter=20;  %优化迭代数	
num_BO_iter=20;  %贝叶斯优化迭代次数	
max_epoch_LC=G_out_data.max_epoch_LC; %最大轮数	
method_mti=G_out_data.method_mti; %最大轮数	
list_cell=	G_out_data.list_cell;	
select_predict_num1=G_out_data.select_predict_num1;  %待预测数	
attention_label=G_out_data.attention_label;	
attention_head=G_out_data.attention_head;	
	
%% 模型训练	
x_mu_all=[];x_sig_all=[];y_mu_all=[];y_sig_all=[];  	
for NUM_all=1:length(data_select1_cell)	
    data_process=data_select1_cell{1,NUM_all};	
    x_feature_label=data_process(:,1:end-select_predict_num);	
    y_feature_label=data_process(:,end-select_predict_num+1:end);	
    y_feature_label1=y_feature_label;	
	
index_label1=randperm((size(x_feature_label,1)));	
 index_label=index_label1;	
  spilt_ri=G_out_data.spilt_ri;	
  train_num=round(spilt_ri(1)/(sum(spilt_ri))*size(x_feature_label,1));                    %训练集个数	
  vaild_num=round((spilt_ri(1)+spilt_ri(2))/(sum(spilt_ri))*size(x_feature_label,1)); %验证集个数	
  %训练集，验证集，测试集	
  train_x_feature_label=x_feature_label(index_label(1:train_num),:);	
  train_y_feature_label=y_feature_label(index_label(1:train_num),:);	
  vaild_x_feature_label=x_feature_label(index_label(train_num+1:vaild_num),:);	
  vaild_y_feature_label=y_feature_label(index_label(train_num+1:vaild_num),:);	
  test_x_feature_label=x_feature_label(index_label(vaild_num+1:end),:);	
  test_y_feature_label=y_feature_label(index_label(vaild_num+1:end),:);	
  %Zscore 标准化	
	
  %训练集	
  x_mu = mean(train_x_feature_label);  x_sig = std(train_x_feature_label); 	
  train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;    % 训练数据标准化	
  y_mu = mean(train_y_feature_label);  y_sig = std(train_y_feature_label); 	
  train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化	
  x_mu_all(NUM_all,:)=x_mu;x_sig_all(NUM_all,:)=x_sig;y_mu_all(NUM_all,:)=y_mu;y_sig_all(NUM_all,:)=y_sig;                   	
  %验证集	
  vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig;    % 训练数据标准化	
  vaild_y_feature_label_norm = (vaild_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化	
  %测试集	
  test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;    % 训练数据标准化	
  test_y_feature_label_norm = (test_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化	
	
	
	
	
   y_train_predict_norm=zeros(size(train_y_feature_label,1),size(train_y_feature_label,2));y_vaild_predict_norm=zeros(size(vaild_y_feature_label,1),size(vaild_y_feature_label,2));	
   y_test_predict_norm=zeros(size(test_y_feature_label,1),size(test_y_feature_label,2));	
	
   for N1=1:length(list_cell)	
	
     num_tree=100;   %树的棵树	
	
	
    [Mdl] = optimizebaye_fitrTreeBagger(train_x_feature_label_norm,train_y_feature_label_norm(:,list_cell{1,N1}(1)),vaild_x_feature_label_norm,vaild_y_feature_label_norm(:,list_cell{1,N1}(1)),num_BO_iter);  	
	
	
	
	
 y_train_predict_norm(:,list_cell{1,N1}(1))=predict(Mdl,train_x_feature_label_norm);	
 y_vaild_predict_norm(:,list_cell{1,N1}(1))=predict(Mdl,vaild_x_feature_label_norm);	
 y_test_predict_norm(:,list_cell{1,N1}(1))=predict(Mdl,test_x_feature_label_norm);	
	
 Model{1,N1}=Mdl;	
 model_all{NUM_all,N1}=Mdl;	
 end 	
	
train_x_feature_label_norm_roll=train_x_feature_label_norm;	
vaild_x_feature_label_norm_roll=vaild_x_feature_label_norm;	
test_x_feature_label_norm_roll=test_x_feature_label_norm;	
	
if length(list_cell)<select_predict_num	
 for N2=1:length(list_cell)	
    roll_list=list_cell{1,N2};	
     for N3=roll_list(2):roll_list(end)	
        train_x_feature_label_norm_roll(:,end-num_input_serise+1:end)=[train_x_feature_label_norm_roll(:,end-num_input_serise+2:end),y_train_predict_norm(:,N3-1)];	
        vaild_x_feature_label_norm_roll(:,end-num_input_serise+1:end)=[vaild_x_feature_label_norm_roll(:,end-num_input_serise+2:end),y_vaild_predict_norm(:,N3-1)];	
        test_x_feature_label_norm_roll(:,end-num_input_serise+1:end)=[test_x_feature_label_norm_roll(:,end-num_input_serise+2:end),y_test_predict_norm(:,N3-1)];	
        y_train_predict_norm(:,N3)=predict(Model{1,N2},train_x_feature_label_norm_roll);	
        y_vaild_predict_norm(:,N3)=predict(Model{1,N2},vaild_x_feature_label_norm_roll);	
        y_test_predict_norm(:,N3)=predict(Model{1,N2},test_x_feature_label_norm_roll);	
      end	
  end	
end	
	
	
	
	
y_train_predict_cell{1,NUM_all}=y_train_predict_norm.*y_sig+y_mu;  %反标准化操作	
y_vaild_predict_cell{1,NUM_all}=y_vaild_predict_norm.*y_sig+y_mu;	
 y_test_predict_cell{1,NUM_all}=y_test_predict_norm.*y_sig+y_mu;	
	
end	
	
    %% 
	
	
y_train_predict=0;y_vaild_predict=0;y_test_predict=0;	
for i=1:length(data_select1_cell)	
      y_train_predict=y_train_predict+ y_train_predict_cell{1,i};	
      y_vaild_predict=y_vaild_predict+ y_vaild_predict_cell{1,i};	
      y_test_predict=y_test_predict+ y_test_predict_cell{1,i};	
end	
	
train_y_feature_label=y_feature_label1(index_label(1:train_num),:); 	
vaild_y_feature_label=y_feature_label1(index_label(train_num+1:vaild_num),:);	
test_y_feature_label=y_feature_label1(index_label(vaild_num+1:end),:);	
	
Tvalue=G_out_data.Tvalue;  %使用的方法	
	
train_y=train_y_feature_label; 	
train_MAE=sum(sum(abs(y_train_predict-train_y)))/size(train_y,1)/size(train_y,2) ; disp([Tvalue,'训练集平均绝对误差MAE：',num2str(train_MAE)])	
train_MAPE=sum(sum(abs((y_train_predict-train_y)./train_y)))/size(train_y,1)/size(train_y,2); disp([Tvalue,'训练集平均相对误差MAPE：',num2str(train_MAPE)])	
train_MSE=(sum(sum(((y_train_predict-train_y)).^2))/size(train_y,1)/size(train_y,2)); disp([Tvalue,'训练集均方误差MSE：',num2str(train_MSE)])    	
train_RMSE=sqrt(sum(sum(((y_train_predict-train_y)).^2))/size(train_y,1)/size(train_y,2)); disp([Tvalue,'训练集均方根误差RMSE：',num2str(train_RMSE)]) 	
train_R2 = 1 - mean(norm(train_y - y_train_predict)^2 / norm(train_y - mean(train_y))^2);   disp([Tvalue,'训练集R方系数R2：',num2str(train_R2)]) 	
disp('************************************************************************************')	
vaild_y=vaild_y_feature_label;	
vaild_MAE=sum(sum(abs(y_vaild_predict-vaild_y)))/size(vaild_y,1)/size(vaild_y,2) ; disp([Tvalue,'验证集平均绝对误差MAE：',num2str(vaild_MAE)])	
vaild_MAPE=sum(sum(abs((y_vaild_predict-vaild_y)./vaild_y)))/size(vaild_y,1)/size(vaild_y,2); disp([Tvalue,'验证集平均相对误差MAPE：',num2str(vaild_MAPE)])	
vaild_MSE=(sum(sum(((y_vaild_predict-vaild_y)).^2))/size(vaild_y,1)/size(vaild_y,2)); disp([Tvalue,'验证集均方误差MSE：',num2str(vaild_MSE)])     	
vaild_RMSE=sqrt(sum(sum(((y_vaild_predict-vaild_y)).^2))/size(vaild_y,1)/size(vaild_y,2)); disp([Tvalue,'验证集均方根误差RMSE：',num2str(vaild_RMSE)]) 	
vaild_R2 = 1 - mean(norm(vaild_y - y_vaild_predict)^2 / norm(vaild_y - mean(vaild_y))^2);   disp([Tvalue,'验证集R方系数R2：',num2str(vaild_R2)]) 	
disp('************************************************************************************')	
test_y=test_y_feature_label;	
test_MAE=sum(sum(abs(y_test_predict-test_y)))/size(test_y,1)/size(test_y,2) ; disp([Tvalue,'测试集平均绝对误差MAE：',num2str(test_MAE)])	
test_MAPE=sum(sum(abs((y_test_predict-test_y)./test_y)))/size(test_y,1)/size(test_y,2); disp([Tvalue,'测试集平均相对误差MAPE：',num2str(test_MAPE)])	
test_MSE=(sum(sum(((y_test_predict-test_y)).^2))/size(test_y,1)/size(test_y,2)); disp([Tvalue,'测试集均方误差MSE：',num2str(test_MSE)]) 	
test_RMSE=sqrt(sum(sum(((y_test_predict-test_y)).^2))/size(test_y,1)/size(test_y,2)); disp([Tvalue,'测试集均方根误差RMSE：',num2str(test_RMSE)]) 	
test_R2 = 1 - mean(norm(test_y - y_test_predict)^2 / norm(test_y - mean(test_y))^2);   disp([Tvalue,'测试集R方系数R2：',num2str(test_R2)]) 	
	
disp('********验证集+测试集预测结果**********没有用到验证集做优化或者集成的算法可以将验证集作为测试集')	
test_y1=[vaild_y_feature_label;test_y_feature_label];	
y_test_predict1=[y_vaild_predict;y_test_predict];	
test_MAE1=sum(sum(abs(y_test_predict1-test_y1)))/size(test_y1,1)/size(test_y1,2) ; disp([Tvalue,'验证集+测试集平均绝对误差MAE：',num2str(test_MAE1)])	
test_MAPE1=sum(sum(abs((y_test_predict1-test_y1)./test_y1)))/size(test_y1,1)/size(test_y1,2); disp([Tvalue,'验证集+测试集平均相对误差MAPE：',num2str(test_MAPE1)])	
test_MSE1=(sum(sum(((y_test_predict1-test_y1)).^2))/size(test_y1,1)/size(test_y1,2)); disp([Tvalue,'验证集+测试集均方误差MSE：',num2str(test_MSE1)]) 	
test_RMSE1=sqrt(sum(sum(((y_test_predict1-test_y1)).^2))/size(test_y1,1)/size(test_y1,2)); disp([Tvalue,'验证集+测试集均方根误差RMSE：',num2str(test_RMSE1)])  	
test_R21 = 1 - mean(norm(test_y1 - y_test_predict1)^2 / norm(test_y1 - mean(test_y1))^2);   disp([Tvalue,'验证集+测试集R方系数R2：',num2str(test_R21)])  	
   




%% 
	
show_wei_num=1; %展示输出的维度	
train_y1=train_y(:,show_wei_num);	
vaild_y1=vaild_y(:,show_wei_num);	
test_y1=test_y(:,show_wei_num);	
	
y_train_predict1=y_train_predict(:,show_wei_num);	
y_vaild_predict1=y_vaild_predict(:,show_wei_num);	
y_test_predict1=y_test_predict(:,show_wei_num);	
	
color_list=G_out_data.color_list; 	
Line_Width=G_out_data.Line_Width;	
rand_list1=G_out_data.rand_list1;	
makesize=G_out_data.makesize;	
color_index=G_out_data.color_index;	
show_num1=G_out_data.show_num1;	
show_num2=G_out_data.show_num2;	
show_num3=G_out_data.show_num3;	
yang_str2=G_out_data.yang_str2;	
yang_str3=G_out_data.yang_str3;	
yangsi_idnex=G_out_data.yangsi_idnex;	
yang_fu3_ku=G_out_data.yang_fu3_ku;	
if show_num1>length(train_y1)	
    show_num1=length(train_y1);	
end	
index_show=1:show_num1;	
figure_density(yangsi_idnex(5),y_train_predict1,train_y1,'训练集')	
figure('Position',[200,200,600,350]);	
XX=1:length(train_y1);	
plot(gca,XX(index_show),train_y1(index_show),yang_fu3_ku{1,yangsi_idnex(1)},'Color',color_list(yangsi_idnex(3),:),'LineWidth',Line_Width(1))	
hold (gca,'on')	
plot(gca, XX(index_show),y_train_predict1(index_show),yang_fu3_ku{1,yangsi_idnex(2)},'Color',color_list(yangsi_idnex(4),:),'LineWidth',Line_Width(1),'MarkerSize',makesize)	
hold (gca,'on')	
title('训练集测试效果')	
	
FontName=G_out_data.FontName;	
FontSize=G_out_data.FontSize;	
kuang_width=G_out_data.kuang_width;	
 set(gca,'FontName',FontName,'FontSize',FontSize,'LineWidth',kuang_width)	
	
xlabel1=G_out_data.xlabel1;	
ylabel1=G_out_data.ylabel1;	
legend1=G_out_data.legend1;	
 xlabel(xlabel1)	
ylabel(ylabel1)	
legend(legend1)	
	
box1=G_out_data.box1;	
box(gca,box1) 	
le_kuang=G_out_data.le_kuang;	
 legend(gca,le_kuang) %图例框消失	
grid1=G_out_data.grid1;	
grid(gca,grid1)	
	
if show_num2>length(test_y1)	
   show_num2=length(test_y1);	
end	
	
index_show=1:show_num2;	
figure_density(yangsi_idnex(5),y_test_predict1,test_y1,'测试集')	
figure('Position',[200,200,600,350]);	
XX=1:length(test_y1);	
plot(gca,XX(index_show),test_y1(index_show),yang_fu3_ku{1,yangsi_idnex(1)},'Color',color_list(yangsi_idnex(3),:),'LineWidth',Line_Width(1))	
hold (gca,'on')	
plot(gca, XX(index_show),y_test_predict1(index_show),yang_fu3_ku{1,yangsi_idnex(2)},'Color',color_list(yangsi_idnex(4),:),'LineWidth',Line_Width(1),'MarkerSize',makesize)	
hold (gca,'on')	
title('测试集测试效果')	
	
 set(gca,'FontName',FontName,'FontSize',FontSize,'LineWidth',kuang_width)	
	
xlabel(xlabel1)	
ylabel(ylabel1)	
legend(legend1)	
	
 box(gca,box1) 	
 legend(gca,le_kuang) %图例框消失	
grid(gca,grid1)	
	
 if show_num3>length(vaild_y1)	
     show_num3=length(vaild_y1);	
end	
index_show=1:show_num3;	
figure_density(yangsi_idnex(5),y_vaild_predict1,vaild_y1,'验证集')	
figure('Position',[200,200,600,350]);	
XX=1:length(vaild_y1);	
plot(gca,XX(index_show),vaild_y1(index_show),yang_fu3_ku{1,yangsi_idnex(1)},'Color',color_list(yangsi_idnex(3),:),'LineWidth',Line_Width(1))	
hold (gca,'on')	
plot(gca, XX(index_show),y_vaild_predict1(index_show),yang_fu3_ku{1,yangsi_idnex(2)},'Color',color_list(yangsi_idnex(4),:),'LineWidth',Line_Width(1),'MarkerSize',makesize)	
hold (gca,'on')	
title('验证集测试效果')	
	
set(gca,'FontName',FontName,'FontSize',FontSize,'LineWidth',kuang_width)	
	
xlabel(xlabel1)	
ylabel(ylabel1)	
legend(legend1)	
	
box(gca,box1) 	
legend(gca,le_kuang) %图例框消失	
grid(gca,grid1)	
	
	
	
figure		
fu_str4=G_out_data.fu_str4;	
rand_list4=G_out_data.rand_list4;	
	
data_get=[train_y1,y_train_predict1];   	
hh(1)=scatter(data_get(:,1),data_get(:,2),makesize*10,'filled','CData',color_list(rand_list4(1),:),'LineWidth',Line_Width(1),'Marker',fu_str4{1,1}); hold on;   	
data_get1=[test_y1,y_test_predict1];   	
hh(2)=scatter(data_get1(:,1),data_get1(:,2),makesize*10,'filled','CData',color_list(rand_list4(2),:),'LineWidth',Line_Width(1),'Marker',fu_str4{1,2}); hold on;   	
   	
lm1 = fitlm(train_y1, y_train_predict1);    % 获取 R^2 系数 	
[y_pred1, y_ci1] = predict(lm1, train_y1);   	
R2_train = 1 - sum((train_y1 - y_train_predict1).^2) / sum((train_y1 - mean(train_y1)).^2);   	
[sorted_x, sortIdx] = sort(train_y1);   	
   	
h1(1)=fill(gca,[train_y1(sortIdx)',fliplr(train_y1(sortIdx)')],[y_ci1(sortIdx,1)',fliplr(y_ci1(sortIdx,2)')],color_list(rand_list4(1),:),'EdgeColor',[1,1,1],'FaceAlpha', 0.3);hold on   	
h2(1)=plot(train_y1,y_pred1,'Color',color_list(rand_list4(1),:)); hold on   	
   	
lm2 = fitlm(test_y1, y_test_predict1);   	
R2_test = 1 - sum((test_y1 - y_test_predict1).^2) / sum((test_y1 - mean(test_y1)).^2);   	
[y_pred2, y_ci2] = predict(lm2, test_y1);   	
[sorted_x1, sortIdx1] = sort(test_y1);   	
h1(2)=fill(gca,[test_y1(sortIdx1)',fliplr(test_y1(sortIdx1)')],[y_ci2(sortIdx1,1)',fliplr(y_ci2(sortIdx1,2)')],color_list(rand_list4(2),:),'EdgeColor',[1,1,1],'FaceAlpha', 0.3);hold on   	
h2(2)=plot(test_y1,y_pred2,'Color',color_list(rand_list4(2),:));hold on   	
txt_set = [ newline 'Train: R^2 = ',num2str(R2_train)...   	
     newline 'Test: R^2 = ',num2str(R2_test)];   	
   	
text(min(train_y1)+0.05*(max(train_y1)-min(train_y1)),max(y_train_predict1)-0.1*(max(y_train_predict1)-min(y_train_predict1)),txt_set,  'color','k','FontName',' Times New Roman','FontSize', 11)   	
xlabel('true');ylabel('predict')   	
set(gca,'FontName',' Times New Roman','FontSize', 12)   	
for i=1:2   	
     Phdl1(i)=PlotDensHist1( hh(i));hold on        	
     Phdl1(i).plotDensHist1();hold on      	
end   	
legend([hh,h1,h2],{'train true','test true','train 95% conf. bounds','test 95% conf. bounds','train regerss fit','test regerss fit'},'EdgeColor',[0.8,0.8,0.8],'FontSize',11,'FontName','Times New Roman','Location','southeast');   	
figure		
fu_str4=G_out_data.fu_str4;	
rand_list4=G_out_data.rand_list4;	
	
data_get=[train_y1,y_train_predict1];   	
hh(1)=scatter(data_get(:,1),data_get(:,2),makesize*10,'filled','CData',color_list(rand_list4(1),:),'LineWidth',Line_Width(1),'Marker',fu_str4{1,1}); hold on;   	
data_get1=[test_y1,y_test_predict1];   	
hh(2)=scatter(data_get1(:,1),data_get1(:,2),makesize*10,'filled','CData',color_list(rand_list4(2),:),'LineWidth',Line_Width(1),'Marker',fu_str4{1,2}); hold on;   	
   	
lm1 = fitlm(train_y1, y_train_predict1);    % 获取 R^2 系数 	
[y_pred1, y_ci1] = predict(lm1, train_y1);   	
 R2_train = 1 - sum((train_y1 - y_train_predict1).^2) / sum((train_y1 - mean(train_y1)).^2);   	
[sorted_x, sortIdx] = sort(train_y1);   	
   	
h1(1)=fill(gca,[train_y1(sortIdx)',fliplr(train_y1(sortIdx)')],[y_ci1(sortIdx,1)',fliplr(y_ci1(sortIdx,2)')],color_list(rand_list4(1),:),'EdgeColor',[1,1,1],'FaceAlpha', 0.3);hold on   	
h2(1)=plot(train_y1,y_pred1,'Color',color_list(rand_list4(1),:)); hold on   	
   	
lm2 = fitlm(test_y1, y_test_predict1);   	
R2_test = 1 - sum((test_y1 - y_test_predict1).^2) / sum((test_y1 - mean(test_y1)).^2);  	
[y_pred2, y_ci2] = predict(lm2, test_y1);   	
[sorted_x1, sortIdx1] = sort(test_y1);   	
h1(2)=fill(gca,[test_y1(sortIdx1)',fliplr(test_y1(sortIdx1)')],[y_ci2(sortIdx1,1)',fliplr(y_ci2(sortIdx1,2)')],color_list(rand_list4(2),:),'EdgeColor',[1,1,1],'FaceAlpha', 0.3);hold on   	
h2(2)=plot(test_y1,y_pred2,'Color',color_list(rand_list4(2),:));hold on   	
txt_set = [ newline 'Train: R^2 = ',num2str(R2_train)...   	
     newline 'Test: R^2 = ',num2str(R2_test)];   	
   	
text(min(train_y1)+0.05*(max(train_y1)-min(train_y1)),max(y_train_predict1)-0.1*(max(y_train_predict1)-min(y_train_predict1)),txt_set,  'color','k','FontName',' Times New Roman','FontSize', 11)   	
xlabel('true');ylabel('predict')   	
set(gca,'FontName',' Times New Roman','FontSize', 12)   	
    %% 
	

	
%% 时序概率区间预测模块  	
	
beta=G_out_data.beta;   	
eta=G_out_data.eta;   	
gailv_upper=[]; %保存上下限的结构体    	
hidden_size=G_out_data.hidden_size; %神经网络类神经元    	
label_vaild=ones(1,length(vaild_y_feature_label));    	
label_test=ones(1,length(test_y_feature_label));    	
	
Upper1=[];Lower1=[];    	
index_cluster_vaild=1:length(label_vaild);    	
index_cluster_test=1:length(label_test);    	
	
	
disp('************KDE区间预测')   	
   	
  	
for N_dim=1:size(y_test_predict,2)   	
     error_vaild=-(y_vaild_predict(:,N_dim)-vaild_y_feature_label(:,N_dim));	
      	
     kernel_label='normal';   %核分布的核平滑器类型 'normal' (default)   'box'  'triangle'   'epanechnikov'  	
     % 带宽 fitdist 使用的默认值为估计正态密度的最佳值	
     	
     pd_1 = fitdist(error_vaild,'Kernel', 'Kernel',kernel_label); %核密度分布	
     [f,xi] = ksdensity(error_vaild,'Bandwidth',pd_1.Bandwidth,'Kernel',kernel_label);	
     cdf = cumsum(f) / sum(f);	
     disp(['最佳带宽确定  ', num2str(pd_1.Bandwidth)])	
	
     if N_dim==1	
        figure;	
        set(gcf,'color','w')  	
        h1 = histogram(error_vaild,'Normalization','probability');	
        hold on	
        plot(xi,f,'LineWidth',1.2)	
        hold on;grid on      	
        set(gcf,'color','w')	
	
        xlabel('数值'); ylabel('概率')    	
        legend('直方图','概率密度函数')	
	
       figure  	
       plot(xi,cdf,'LineWidth',1.2)	
       hold on;grid on	
       set(gcf,'color','w')	
       xlabel('数值');ylabel('概率')  	
       legend('训练集核密度估计')	
	
   end	
	
    for m=1:length(beta)	
       confidence_level = beta(m);	
       upper_bound_get = xi(find(cdf >= 1-(1-confidence_level)/2, 1, 'first'));	
       % % 寻找置信水平对应的概率区间上下界	
       lower_bound_get = xi(find(cdf >= (1-confidence_level)/2, 1, 'first'));	
	
       C2_upper(:,m) = y_test_predict(:,N_dim) + upper_bound_get;	
       C2_lower(:,m) = y_test_predict(:,N_dim) + lower_bound_get;	
	
       Lower1=C2_lower;	
       Upper1=C2_upper;	
       Mdl1{N_dim,1}(:,m)=upper_bound_get;	
       Mdl1{N_dim,2}(:,m)=lower_bound_get;	
	
   end	
	
     gailv_upper(N_dim).Upper1=Upper1; 	
     gailv_upper(N_dim).Lower1=Lower1; 	
    	
end    	
	
	
	
Lower1_all=[];Upper1_all=[];test_y_feature_label_all=[];   	
for N_dim=1:size(test_y_feature_label,2)    	
       Lower1_all=[Lower1_all;gailv_upper(N_dim).Lower1];   	
       Upper1_all=[Upper1_all;gailv_upper(N_dim).Upper1];  	
       test_y_feature_label_all=[test_y_feature_label_all;test_y_feature_label(:,N_dim)];  	
end   	
    	
[value_result1]=interval_valuate1(Lower1_all,Upper1_all,test_y_feature_label_all,eta,beta);  	
    	
value_lsit1=[value_result1.PICP;value_result1.PINAW;value_result1.CWC;value_result1.MPICD;value_result1.AIS];    	
    	
index1={'PICP','PINAW','CWC','MPICD','AIS'};    	
beta_str=[];    	
for j=1:length(beta)    	
     beta_str{1,j}=num2str(beta(j));    	
end    	
 	
value_lsit_table=array2table(value_lsit1); 	
value_lsit_table.Properties.VariableNames=(beta_str); 	
value_lsit_table.Properties.RowNames=index1; 	
disp(value_lsit_table) 	
	
	
show_wei_num=G_out_data.show_wei_num;	
test_y_feature_label1=test_y_feature_label;	
	
	
test_y_feature_label=test_y_feature_label1(:,show_wei_num);	
	
Lower1=gailv_upper(:,show_wei_num).Lower1;	
Upper1=gailv_upper(:,show_wei_num).Upper1;	
	
y_test_predict1=y_test_predict(:,show_wei_num);	
	
	
	
color_list1=G_out_data.color_list1;	
color_list2=G_out_data.color_list2;	
plot_index=G_out_data.plot_index;	
yang_fu3_ku=G_out_data.yang_fu3_ku;	
Line_Width=G_out_data.Line_Width;	
yang_str3=G_out_data.yang_str3;	
FontName=G_out_data.FontName;	
FontSize=G_out_data.FontSize;	
	
kuang_width=G_out_data.kuang_width;	
xlabel1=G_out_data.xlabel1;	
ylabel1=G_out_data.ylabel1;	
grid1=G_out_data.grid1;	
legend1=G_out_data.legend1;	
box1=G_out_data.box1;	
le_kuang=G_out_data.le_kuang;	
	
PlotProbability1(y_test_predict1,test_y_feature_label,Lower1,Upper1,plot_index(1),plot_index(2),'数据',color_list1(1,:),color_list1(2,:),color_list1(3,:),'区间预测',beta);  %背景的颜色/区间的颜色/真实值颜色/预测值颜色	






%% 


%% ========== 最后追加：三目标指标 + 原始列名导出 ==========

disp('========== 开始导出三目标预测结果 ==========')

output_file = '三目标预测结果导出.xlsx';

% 目标个数：mat 文件中为 3
num_target = G_out_data.select_predict_num;

% 特征名称：必须与 train_x_feature_label / vaild_x_feature_label / test_x_feature_label 顺序一致
feature_name = data_biao1(feature_need_last);

% 目标名称：原始数据最后 3 列
target_name = data_biao1(end-num_target+1:end);

% 恢复三目标真实值，避免前面 show_wei_num 把目标改成单列
train_true_all = y_feature_label1(index_label(1:train_num), :);
vaild_true_all = y_feature_label1(index_label(train_num+1:vaild_num), :);
test_true_all  = y_feature_label1(index_label(vaild_num+1:end), :);

% 三目标预测值
train_pred_all = y_train_predict;
vaild_pred_all = y_vaild_predict;
test_pred_all  = y_test_predict;

% 特征数据，顺序与 feature_name 完全一致
train_x_all = train_x_feature_label;
vaild_x_all = vaild_x_feature_label;
test_x_all  = test_x_feature_label;

set_name = {'训练集','验证集','测试集'};
X_all = {train_x_all, vaild_x_all, test_x_all};
Y_true_all = {train_true_all, vaild_true_all, test_true_all};
Y_pred_all = {train_pred_all, vaild_pred_all, test_pred_all};

%% 1. 三个目标分别计算指标

metric_header = {'数据集','预测目标','MAE','MAPE','MSE','RMSE','R2'};
metric_cell = metric_header;

for s = 1:3
    Y_true = Y_true_all{s};
    Y_pred = Y_pred_all{s};

    for k = 1:num_target
        y_true = Y_true(:,k);
        y_pred = Y_pred(:,k);

        MAE = mean(abs(y_pred - y_true));
        MAPE = mean(abs((y_pred - y_true) ./ (abs(y_true) + eps)));
        MSE = mean((y_pred - y_true).^2);
        RMSE = sqrt(MSE);
        R2 = 1 - sum((y_true - y_pred).^2) / sum((y_true - mean(y_true)).^2);

        metric_cell(end+1,:) = {set_name{s}, target_name{k}, MAE, MAPE, MSE, RMSE, R2};
    end
end

disp('========== 三目标分别评价指标 ==========')
disp(metric_cell)

writecell(metric_cell, output_file, 'Sheet', '三目标评价指标');

%% 2. 分训练集、验证集、测试集导出：原始特征名 + 真实值 + 预测值

true_name = strcat(target_name, '_真实值');
pred_name = strcat(target_name, '_预测值');

export_header = [feature_name, true_name, pred_name];

for s = 1:3
    X = X_all{s};
    Y_true = Y_true_all{s};
    Y_pred = Y_pred_all{s};

    export_data = [X, Y_true, Y_pred];

    export_cell = [export_header; num2cell(export_data)];

    writecell(export_cell, output_file, 'Sheet', set_name{s});
end

disp(['三目标预测结果已成功导出到文件：', output_file])








%% ========== 重新生成：随机200测试样本 三目标SHAP分析 ==========

%% ========== 重新生成：随机200测试样本 三目标SHAP分析 ==========

disp('========== 开始重新计算三目标 SHAP ==========')

shap_output_file = '三目标SHAP_随机200测试样本.xlsx';

num_target = G_out_data.select_predict_num;


% print_index_name 是前面特征选择后生成的最终特征名称
% 它与 data_select(:,1:select_feature_num) 的列顺序一致
% 也应与 train_x_feature_label / test_x_feature_label 的列顺序一致

if exist('print_index_name','var')
    feature_name = print_index_name;
else
    error('未找到 print_index_name，请确认特征选择代码已经运行。')
end

feature_name = feature_name(:)';

% 测试集特征
X_test_raw = test_x_feature_label;

% SHAP使用的特征数量
num_feature = size(X_test_raw,2);

% 严格检查，防止标签和特征错位
assert(length(feature_name) == num_feature, ...
    'SHAP特征标签数量与测试集特征数量不一致：请检查 print_index_name 与 test_x_feature_label 是否对应。');

disp('========== SHAP使用的特征选择后标签 ==========')
disp(feature_name)



target_name = data_biao1(end-num_target+1:end);
target_name = target_name(:)';

disp('========== SHAP使用的目标标签 ==========')
disp(target_name)



% 注意：模型训练时使用标准化特征，因此SHAP预测也必须输入标准化后的X
X_test_norm = (X_test_raw - x_mu) ./ x_sig;

num_test = size(X_test_norm,1);


rng(G_out_data.random_seed)

num_shap_sample = min(200,num_test);
sample_index = randperm(num_test,num_shap_sample);

X_shap_raw = X_test_raw(sample_index,:);
X_shap_norm = X_test_norm(sample_index,:);



% 背景值使用训练集标准化均值
% 如果 train_x_feature_label_norm 存在，优先使用它
if exist('train_x_feature_label_norm','var')
    x_base = mean(train_x_feature_label_norm,1);
else
    train_x_norm_tmp = (train_x_feature_label - x_mu) ./ x_sig;
    x_base = mean(train_x_norm_tmp,1);
end


% shap_value_all: 样本 × 特征 × 目标
shap_value_all = zeros(num_shap_sample,num_feature,num_target);;

%% ========== 1. 计算近似SHAP ==========
% 思路：对每个样本，先计算完整输入预测值；
% 然后每次把一个特征替换为背景均值；
% 两者预测差值作为该特征贡献。

for k = 1:num_target

    disp(['正在计算目标：', target_name{k}])

    Mdl_now = model_all{1,k};

    for i = 1:num_shap_sample

        x_now = X_shap_norm(i,:);

        y_full_norm = predict(Mdl_now,x_now);

        for j = 1:num_feature

            x_replace = x_now;
            x_replace(j) = x_base(j);

            y_replace_norm = predict(Mdl_now,x_replace);

            shap_value_all(i,j,k) = ...
                (y_full_norm - y_replace_norm) * y_sig(k);

        end
    end
end

shap_value_all(~isfinite(shap_value_all)) = 0;

%% ========== 2. 计算平均绝对SHAP ==========

shap_mean_abs = squeeze(mean(abs(shap_value_all),1));  % 特征 × 目标

% 防止维度被 squeeze 转置
if size(shap_mean_abs,1) ~= num_feature && size(shap_mean_abs,2) == num_feature
    shap_mean_abs = shap_mean_abs';
end

shap_mean_abs(~isfinite(shap_mean_abs)) = 0;

total_shap = sum(shap_mean_abs,2);

% 堆叠图按总贡献从大到小排序
[~,sort_id] = sort(total_shap,'descend');

shap_plot = shap_mean_abs(sort_id,:);
feature_name_plot = feature_name(sort_id);
total_plot = total_shap(sort_id);

shap_percent = shap_plot ./ (sum(shap_plot,1) + eps) * 100;
shap_percent(~isfinite(shap_percent)) = 0;

%% ========== 3. 导出平均SHAP表格 ==========

shap_cell = cell(num_feature+1,num_target+2);
shap_cell(1,:) = [{'特征名称'}, target_name, {'总贡献'}];

for i = 1:num_feature
    shap_cell{i+1,1} = feature_name_plot{i};

    for k = 1:num_target
        shap_cell{i+1,k+1} = shap_plot(i,k);
    end

    shap_cell{i+1,num_target+2} = total_plot(i);
end

writecell(shap_cell,shap_output_file,'Sheet','平均绝对SHAP');

%% ========== 4. 导出200个样本原始特征值 ==========

sample_cell = cell(num_shap_sample+1,num_feature+1);
sample_cell(1,:) = [{'测试集样本序号'}, feature_name];

sample_cell(2:end,1) = num2cell(sample_index(:));
sample_cell(2:end,2:end) = num2cell(X_shap_raw);

writecell(sample_cell,shap_output_file,'Sheet','随机200样本特征值');

%% ========== 5. 分目标导出每个特征SHAP值 ==========

for k = 1:num_target

    shap_k = shap_value_all(:,:,k);

    shap_k_cell = cell(num_shap_sample+1,num_feature+1);
    shap_k_cell(1,:) = [{'测试集样本序号'}, feature_name];

    shap_k_cell(2:end,1) = num2cell(sample_index(:));
    shap_k_cell(2:end,2:end) = num2cell(shap_k);

    sheet_name = ['SHAP_', target_name{k}];

    if strlength(sheet_name) > 31
        sheet_name = char(extractBefore(sheet_name,32));
    end

    writecell(shap_k_cell,shap_output_file,'Sheet',sheet_name);
end

disp(['SHAP数据已导出至：', shap_output_file])

%% ========== 6. 三目标SHAP平均贡献堆叠图 ==========

fig1 = figure('Position',[100,100,1200,720]);
set(fig1,'Color','w');

ax1 = axes(fig1);
barh(ax1,shap_plot,'stacked','BarWidth',0.72);

set(ax1,'YDir','reverse')
yticks(ax1,1:num_feature)
yticklabels(ax1,feature_name_plot)

xlabel(ax1,'平均绝对 SHAP 值','FontName','Microsoft YaHei','FontSize',13)
ylabel(ax1,'特征变量','FontName','Microsoft YaHei','FontSize',13)

title(ax1,'随机200个测试样本下三目标SHAP特征贡献堆叠图', ...
    'FontName','Microsoft YaHei','FontSize',15,'FontWeight','bold')

legend(ax1,target_name, ...
    'Location','eastoutside', ...
    'FontName','Microsoft YaHei', ...
    'FontSize',11, ...
    'Box','off')

set(ax1,'FontName','Microsoft YaHei','FontSize',11, ...
    'LineWidth',1.1,'TickDir','out','Box','off')

grid(ax1,'on')

try
    ax1.Toolbar.Visible = 'off';
catch
end

drawnow
exportgraphics(fig1,'三目标SHAP_堆叠图.png','Resolution',600);

%% ========== 7. 三目标SHAP贡献率堆叠图 ==========

fig2 = figure('Position',[100,100,1200,720]);
set(fig2,'Color','w');

ax2 = axes(fig2);
barh(ax2,shap_percent,'stacked','BarWidth',0.72);

set(ax2,'YDir','reverse')
yticks(ax2,1:num_feature)
yticklabels(ax2,feature_name_plot)

xlabel(ax2,'SHAP贡献率 / %','FontName','Microsoft YaHei','FontSize',13)
ylabel(ax2,'特征变量','FontName','Microsoft YaHei','FontSize',13)

title(ax2,'随机200个测试样本下三目标SHAP贡献率堆叠图', ...
    'FontName','Microsoft YaHei','FontSize',15,'FontWeight','bold')

legend(ax2,target_name, ...
    'Location','eastoutside', ...
    'FontName','Microsoft YaHei', ...
    'FontSize',11, ...
    'Box','off')

set(ax2,'FontName','Microsoft YaHei','FontSize',11, ...
    'LineWidth',1.1,'TickDir','out','Box','off')

grid(ax2,'on')

try
    ax2.Toolbar.Visible = 'off';
catch
end

drawnow
exportgraphics(fig2,'三目标SHAP_贡献率堆叠图.png','Resolution',600);

%% ========== 8. 每个目标单独绘制SHAP条形图 + 蜂群图 ==========

for k = 1:num_target

    shap_k = shap_value_all(:,:,k);

    mean_abs_k = mean(abs(shap_k),1);
    [~,id_k] = sort(mean_abs_k,'descend');

    shap_k_plot = shap_k(:,id_k);
    X_raw_plot = X_shap_raw(:,id_k);
    feature_k_plot = feature_name(id_k);

 
    fig_bar = figure('Position',[100,100,950,650]);
    set(fig_bar,'Color','w');

    ax_bar = axes(fig_bar);
    barh(ax_bar,mean_abs_k(id_k),'BarWidth',0.72);

    set(ax_bar,'YDir','reverse')
    yticks(ax_bar,1:num_feature)
    yticklabels(ax_bar,feature_k_plot)

    xlabel(ax_bar,'平均绝对 SHAP 值','FontName','Microsoft YaHei','FontSize',13)
    ylabel(ax_bar,'特征变量','FontName','Microsoft YaHei','FontSize',13)

    title(ax_bar,[target_name{k}, '：SHAP特征重要性'], ...
        'FontName','Microsoft YaHei','FontSize',15,'FontWeight','bold')

    set(ax_bar,'FontName','Microsoft YaHei','FontSize',11, ...
        'LineWidth',1.1,'TickDir','out','Box','off')

    grid(ax_bar,'on')

    try
        ax_bar.Toolbar.Visible = 'off';
    catch
    end

    drawnow
    exportgraphics(fig_bar,[target_name{k}, '_SHAP条形图.png'],'Resolution',600);

   
    fig_bee = figure('Position',[100,100,1050,700]);
    set(fig_bee,'Color','w');

    ax_bee = axes(fig_bee);
    hold(ax_bee,'on')

    for j = 1:num_feature

        shap_j = shap_k_plot(:,j);
        xcolor = X_raw_plot(:,j);

        % 归一化颜色，表示特征值大小
        xcolor_norm = (xcolor - min(xcolor)) ./ (max(xcolor)-min(xcolor)+eps);

        jitter = (rand(num_shap_sample,1)-0.5) * 0.45;
        y_pos = j + jitter;

        scatter(ax_bee,shap_j,y_pos,28,xcolor_norm, ...
            'filled', ...
            'MarkerFaceAlpha',0.72, ...
            'MarkerEdgeAlpha',0.15);

    end

    colormap(ax_bee,jet)
    cb = colorbar(ax_bee);
    cb.Label.String = '特征值大小';
    cb.Label.FontName = 'Microsoft YaHei';
    cb.Label.FontSize = 11;

    set(ax_bee,'YDir','reverse')
    yticks(ax_bee,1:num_feature)
    yticklabels(ax_bee,feature_k_plot)

    xline(ax_bee,0,'k--','LineWidth',1)

    xlabel(ax_bee,'SHAP 值','FontName','Microsoft YaHei','FontSize',13)
    ylabel(ax_bee,'特征变量','FontName','Microsoft YaHei','FontSize',13)

    title(ax_bee,[target_name{k}, '：SHAP蜂群图'], ...
        'FontName','Microsoft YaHei','FontSize',15,'FontWeight','bold')

    set(ax_bee,'FontName','Microsoft YaHei','FontSize',11, ...
        'LineWidth',1.1,'TickDir','out','Box','off')

    grid(ax_bee,'on')

    try
        ax_bee.Toolbar.Visible = 'off';
    catch
    end

    drawnow
    exportgraphics(fig_bee,[target_name{k}, '_SHAP蜂群图.png'],'Resolution',600);

end

disp('========== SHAP堆叠图、贡献率图、条形图、蜂群图全部生成完成 ==========')











%% 
%% ========== 多算法多目标优化框架：NSGA-II + MOEA/D + MOPSO ==========

disp('========== 开始多算法多目标优化：NSGA-II + MOEA/D + MOPSO ==========')

multi_file = '多算法盾构多目标优化_Pareto解集.xlsx';

num_target = G_out_data.select_predict_num;

% 特征名称：与模型输入顺序一致
feature_name = data_biao1(feature_need_last);
feature_name = feature_name(:)';

% 目标名称：原始数据最后3列
target_name = data_biao1(end-num_target+1:end);
target_name = target_name(:)';

% 决策变量
X_decision = train_x_feature_label;
num_var = size(X_decision,2);
feature_name = feature_name(1:num_var);

% 变量范围：训练集5%~95%分位数
lb = prctile(X_decision,5,1);
ub = prctile(X_decision,95,1);

for i = 1:num_var
    if lb(i) == ub(i)
        lb(i) = min(X_decision(:,i));
        ub(i) = max(X_decision(:,i));
    end
    if lb(i) == ub(i)
        lb(i) = lb(i) - 1e-6;
        ub(i) = ub(i) + 1e-6;
    end
end

rng(G_out_data.random_seed)

%% ========== 公共目标函数 ==========
objfun = @(x) shield_multi_objective(x,model_all,x_mu,x_sig,y_mu,y_sig,num_target);

%% ========== 1. NSGA-II ==========
disp('========== 运行 NSGA-II ==========')

options_nsga = optimoptions('gamultiobj', ...
    'PopulationSize',200, ...
    'MaxGenerations',50, ...
    'CrossoverFraction',0.8, ...
    'ParetoFraction',0.6, ...
    'FunctionTolerance',1e-5, ...
    'Display','iter');

[x_nsga,f_nsga] = gamultiobj(objfun,num_var,[],[],[],[],lb,ub,[],options_nsga);

Y_nsga = [-f_nsga(:,1), f_nsga(:,2), f_nsga(:,3)];

%% ========== 2. MOEA/D ==========
disp('========== 运行 MOEA/D ==========')

moead_pop = 200;
moead_iter = 50;
neighbor_num = 20;

[x_moead,f_moead] = run_moead(objfun,lb,ub,moead_pop,moead_iter,neighbor_num);

Y_moead = [-f_moead(:,1), f_moead(:,2), f_moead(:,3)];

%% ========== 3. MOPSO ==========
disp('========== 运行 MOPSO ==========')

mopso_pop = 200;
mopso_iter = 50;
archive_size = 100;

[x_mopso,f_mopso] = run_mopso(objfun,lb,ub,mopso_pop,mopso_iter,archive_size);

Y_mopso = [-f_mopso(:,1), f_mopso(:,2), f_mopso(:,3)];

%% ========== 4. 合并所有算法 Pareto 解集 ==========

alg_nsga = repmat({'NSGA-II'},size(x_nsga,1),1);
alg_moead = repmat({'MOEA/D'},size(x_moead,1),1);
alg_mopso = repmat({'MOPSO'},size(x_mopso,1),1);

X_all = [x_nsga; x_moead; x_mopso];
Y_all = [Y_nsga; Y_moead; Y_mopso];
F_all = [-Y_all(:,1), Y_all(:,2), Y_all(:,3)];
alg_all = [alg_nsga; alg_moead; alg_mopso];

%% ========== 5. 提取合并非支配解集 ==========

nd_flag = get_nondominated_flag(F_all);

X_nd = X_all(nd_flag,:);
Y_nd = Y_all(nd_flag,:);
F_nd = F_all(nd_flag,:);
alg_nd = alg_all(nd_flag);

%% ========== 6. 导出所有算法解集到同一个 Excel ==========

header_all = [{'算法来源'}, feature_name, ...
    {'预测_刀盘贯入度'}, ...
    {'预测_刀盘磨损压力'}, ...
    {'预测_刀盘总功率'}];

all_cell = [header_all; [alg_all, num2cell([X_all,Y_all])]];

writecell(all_cell,multi_file,'Sheet','全部算法Pareto解集');

%% ========== 7. 导出合并非支配解集 ==========

nd_cell = [header_all; [alg_nd, num2cell([X_nd,Y_nd])]];

writecell(nd_cell,multi_file,'Sheet','合并非支配解集');

%% ========== 8. 导出变量范围 ==========

bound_cell = cell(num_var+1,3);
bound_cell(1,:) = {'变量名称','下界_5%分位数','上界_95%分位数'};

for i = 1:num_var
    bound_cell{i+1,1} = feature_name{i};
    bound_cell{i+1,2} = lb(i);
    bound_cell{i+1,3} = ub(i);
end

writecell(bound_cell,multi_file,'Sheet','变量取值范围');

%% ========== 9. 导出每种算法解集数量统计 ==========

stat_cell = {
    '算法','Pareto解数量';
    'NSGA-II',size(x_nsga,1);
    'MOEA/D',size(x_moead,1);
    'MOPSO',size(x_mopso,1);
    '合并全部解',size(X_all,1);
    '合并非支配解',size(X_nd,1)
};

writecell(stat_cell,multi_file,'Sheet','解集数量统计');

%% ========== 10. 每种算法单独绘制 Pareto 三维前沿 ==========

%% ========== 10. 三种算法 Pareto 前沿绘制在同一张图 ==========

fig_all_alg = figure('Position',[100,100,1000,760]);
set(fig_all_alg,'Color','w')

ax = axes(fig_all_alg);
hold(ax,'on')

s1 = scatter3(ax,Y_nsga(:,1),Y_nsga(:,2),Y_nsga(:,3),60,'filled');
s2 = scatter3(ax,Y_moead(:,1),Y_moead(:,2),Y_moead(:,3),60,'filled');
s3 = scatter3(ax,Y_mopso(:,1),Y_mopso(:,2),Y_mopso(:,3),60,'filled');

xlabel(ax,[target_name{1}, ' / 越大越好'], ...
    'FontName','Microsoft YaHei','FontSize',12)

ylabel(ax,[target_name{2}, ' / 越小越好'], ...
    'FontName','Microsoft YaHei','FontSize',12)

zlabel(ax,[target_name{3}, ' / 越小越好'], ...
    'FontName','Microsoft YaHei','FontSize',12)

title(ax,'三种多目标优化算法 Pareto 前沿对比', ...
    'FontName','Microsoft YaHei','FontSize',15,'FontWeight','bold')

legend(ax,[s1,s2,s3],{'NSGA-II','MOEA/D','MOPSO'}, ...
    'Location','best', ...
    'FontName','Microsoft YaHei', ...
    'FontSize',11)

grid(ax,'on')
box(ax,'on')
view(ax,135,28)

set(ax, ...
    'FontName','Microsoft YaHei', ...
    'FontSize',11, ...
    'LineWidth',1.1, ...
    'TickDir','out')

try
    ax.Toolbar.Visible = 'off';
catch
end

drawnow
exportgraphics(fig_all_alg,'三算法Pareto前沿对比图.png','Resolution',600);


%% ========== 11. 合并非支配解集三维图 ==========

fig_nd = figure('Position',[100,100,1000,760]);
set(fig_nd,'Color','w')

ax_nd = axes(fig_nd);
hold(ax_nd,'on')

alg_unique = unique(alg_nd);

marker_size = 70;

for i = 1:length(alg_unique)

    alg_now = alg_unique{i};
    idx = strcmp(alg_nd,alg_now);

    scatter3(ax_nd, ...
        Y_nd(idx,1), ...
        Y_nd(idx,2), ...
        Y_nd(idx,3), ...
        marker_size, ...
        'filled', ...
        'DisplayName',alg_now);

end

xlabel(ax_nd,[target_name{1}, ' / 越大越好'], ...
    'FontName','Microsoft YaHei','FontSize',12)

ylabel(ax_nd,[target_name{2}, ' / 越小越好'], ...
    'FontName','Microsoft YaHei','FontSize',12)

zlabel(ax_nd,[target_name{3}, ' / 越小越好'], ...
    'FontName','Microsoft YaHei','FontSize',12)

title(ax_nd,'多算法合并非支配 Pareto 前沿', ...
    'FontName','Microsoft YaHei','FontSize',15,'FontWeight','bold')

legend(ax_nd,'show', ...
    'Location','best', ...
    'FontName','Microsoft YaHei', ...
    'FontSize',11)

grid(ax_nd,'on')
box(ax_nd,'on')
view(ax_nd,135,28)

set(ax_nd, ...
    'FontName','Microsoft YaHei', ...
    'FontSize',11, ...
    'LineWidth',1.1, ...
    'TickDir','out')

try
    ax_nd.Toolbar.Visible = 'off';
catch
end

drawnow
exportgraphics(fig_nd,'多算法合并非支配Pareto前沿.png','Resolution',600);

disp(['多算法 Pareto 解集已导出至：',multi_file])
disp('========== 多算法多目标优化完成 ==========')


%% 



%% ========== 局部函数1：统一目标函数 ==========
function f = shield_multi_objective(x,model_all,x_mu,x_sig,y_mu,y_sig,num_target)

    x_norm = (x - x_mu) ./ x_sig;

    y_pred = zeros(1,num_target);

    for k = 1:num_target
        y_norm = predict(model_all{1,k},x_norm);
        y_pred(k) = y_norm .* y_sig(k) + y_mu(k);
    end

    % 目标：
    % 1 最大化刀盘贯入度 -> 最小化负贯入度
    % 2 最小化刀盘磨损压力
    % 3 最小化刀盘总功率
    f = [-y_pred(1), y_pred(2), y_pred(3)];

end


%% ========== 局部函数2：MOEA/D ==========
function [X_pareto,F_pareto] = run_moead(objfun,lb,ub,pop_size,max_iter,neighbor_num)

    num_var = length(lb);
    num_obj = 3;

    W = rand(pop_size,num_obj);
    W = W ./ sum(W,2);

    distW = squareform(pdist(W));
    [~,B] = sort(distW,2);
    B = B(:,1:neighbor_num);

    X = rand(pop_size,num_var).*(ub-lb)+lb;
    F = zeros(pop_size,num_obj);

    for i = 1:pop_size
        F(i,:) = objfun(X(i,:));
    end

    z = min(F,[],1);

    for iter = 1:max_iter

        for i = 1:pop_size

            P = B(i,randperm(neighbor_num,2));
            x1 = X(P(1),:);
            x2 = X(P(2),:);

            y = x1 + rand(1,num_var).*(x2-x1);
            y = min(max(y,lb),ub);

            mutation_rate = 1/num_var;
            for j = 1:num_var
                if rand < mutation_rate
                    y(j) = lb(j) + rand*(ub(j)-lb(j));
                end
            end

            fy = objfun(y);
            z = min(z,fy);

            for jj = 1:neighbor_num
                k = B(i,jj);

                g_old = max(W(k,:).*abs(F(k,:)-z));
                g_new = max(W(k,:).*abs(fy-z));

                if g_new <= g_old
                    X(k,:) = y;
                    F(k,:) = fy;
                end
            end
        end
    end

    nd = get_nondominated_flag(F);
    X_pareto = X(nd,:);
    F_pareto = F(nd,:);

end


%% ========== 局部函数3：MOPSO ==========
function [X_pareto,F_pareto] = run_mopso(objfun,lb,ub,pop_size,max_iter,archive_size)

    num_var = length(lb);
    num_obj = 3;

    X = rand(pop_size,num_var).*(ub-lb)+lb;
    V = zeros(pop_size,num_var);

    F = zeros(pop_size,num_obj);
    for i = 1:pop_size
        F(i,:) = objfun(X(i,:));
    end

    pbest_X = X;
    pbest_F = F;

    archive_X = X;
    archive_F = F;

    nd = get_nondominated_flag(archive_F);
    archive_X = archive_X(nd,:);
    archive_F = archive_F(nd,:);

    w = 0.6;
    c1 = 1.5;
    c2 = 1.5;

    for iter = 1:max_iter

        for i = 1:pop_size

            leader_id = randi(size(archive_X,1));
            leader = archive_X(leader_id,:);

            V(i,:) = w*V(i,:) ...
                + c1*rand(1,num_var).*(pbest_X(i,:)-X(i,:)) ...
                + c2*rand(1,num_var).*(leader-X(i,:));

            X(i,:) = X(i,:) + V(i,:);
            X(i,:) = min(max(X(i,:),lb),ub);

            F(i,:) = objfun(X(i,:));

            if dominates(F(i,:),pbest_F(i,:))
                pbest_X(i,:) = X(i,:);
                pbest_F(i,:) = F(i,:);
            elseif ~dominates(pbest_F(i,:),F(i,:)) && rand < 0.5
                pbest_X(i,:) = X(i,:);
                pbest_F(i,:) = F(i,:);
            end
        end

        archive_X = [archive_X; X];
        archive_F = [archive_F; F];

        nd = get_nondominated_flag(archive_F);
        archive_X = archive_X(nd,:);
        archive_F = archive_F(nd,:);

        if size(archive_X,1) > archive_size
            select_id = randperm(size(archive_X,1),archive_size);
            archive_X = archive_X(select_id,:);
            archive_F = archive_F(select_id,:);
        end
    end

    X_pareto = archive_X;
    F_pareto = archive_F;

end


%% ========== 局部函数4：非支配解判断 ==========
function nd_flag = get_nondominated_flag(F)

    n = size(F,1);
    nd_flag = true(n,1);

    for i = 1:n
        for j = 1:n
            if i ~= j
                if dominates(F(j,:),F(i,:))
                    nd_flag(i) = false;
                    break
                end
            end
        end
    end

end


%% ========== 局部函数5：支配关系 ==========
function flag = dominates(a,b)

    flag = all(a <= b) && any(a < b);

end


%% ========== 局部函数6：单算法Pareto图 ==========
function plot_single_pareto(Y,alg_name,save_name)

    fig = figure('Position',[100,100,900,720]);
    set(fig,'Color','w')

    scatter3(Y(:,1),Y(:,2),Y(:,3),60,Y(:,1),'filled')

    xlabel('刀盘贯入度 / 越大越好','FontName','Microsoft YaHei','FontSize',12)
    ylabel('刀盘磨损压力 / 越小越好','FontName','Microsoft YaHei','FontSize',12)
    zlabel('刀盘总功率 / 越小越好','FontName','Microsoft YaHei','FontSize',12)

    title([alg_name,' Pareto 前沿'], ...
        'FontName','Microsoft YaHei','FontSize',15,'FontWeight','bold')

    cb = colorbar;
    cb.Label.String = '刀盘贯入度';
    cb.Label.FontName = 'Microsoft YaHei';

    grid on
    box on
    view(135,28)

    set(gca,'FontName','Microsoft YaHei','FontSize',11,'LineWidth',1.1)

    try
        gca.Toolbar.Visible = 'off';
    catch
    end

    drawnow
    exportgraphics(fig,save_name,'Resolution',600);

end







%% 
%% ========== TOPSIS评分：从合并非支配Pareto解集中选择最优解 ==========

disp('========== 开始 TOPSIS 评分 ==========')

% 候选解：合并非支配 Pareto 解集
X_topsis = X_nd;
Y_topsis = Y_nd;
alg_topsis = alg_nd;

num_solution = size(Y_topsis,1);

% 三个目标：
% 目标1：刀盘贯入度，效益型，越大越好
% 目标2：刀盘磨损压力，成本型，越小越好
% 目标3：刀盘总功率，成本型，越小越好

% 权重，可根据论文需要调整
% 当前默认三目标等权
weight = [1/3, 1/3, 1/3];

%% ========== 1. TOPSIS指标归一化 ==========
% 将三个目标全部转化为"越大越好"的形式

Z = zeros(num_solution,3);

% 刀盘贯入度：越大越好
Z(:,1) = (Y_topsis(:,1) - min(Y_topsis(:,1))) ./ ...
         (max(Y_topsis(:,1)) - min(Y_topsis(:,1)) + eps);

% 刀盘磨损压力：越小越好
Z(:,2) = (max(Y_topsis(:,2)) - Y_topsis(:,2)) ./ ...
         (max(Y_topsis(:,2)) - min(Y_topsis(:,2)) + eps);

% 刀盘总功率：越小越好
Z(:,3) = (max(Y_topsis(:,3)) - Y_topsis(:,3)) ./ ...
         (max(Y_topsis(:,3)) - min(Y_topsis(:,3)) + eps);

% 加权归一化矩阵
V = Z .* weight;

%% ========== 2. 正理想解和负理想解 ==========

ideal_best = max(V,[],1);
ideal_worst = min(V,[],1);

D_best = sqrt(sum((V - ideal_best).^2,2));
D_worst = sqrt(sum((V - ideal_worst).^2,2));

% TOPSIS得分，越大越优
topsis_score = D_worst ./ (D_best + D_worst + eps);

[score_sort,score_id] = sort(topsis_score,'descend');

best3_id = score_id(1:min(3,num_solution));

%% ========== 3. 导出TOPSIS评分完整结果 ==========

target_export_name = strcat(target_name,'(预测)');

topsis_header = [{'TOPSIS排名','算法来源'}, ...
    feature_name, ...
    target_export_name, ...
    {'归一化_贯入度','归一化_磨损压力','归一化_刀盘总功率'}, ...
    {'距离正理想解','距离负理想解','TOPSIS得分'}];

topsis_data_cell = cell(num_solution,length(topsis_header));

for i = 1:num_solution

    idx = score_id(i);

    topsis_data_cell{i,1} = i;
    topsis_data_cell{i,2} = alg_topsis{idx};

    % 特征值
    for j = 1:size(X_topsis,2)
        topsis_data_cell{i,2+j} = X_topsis(idx,j);
    end

    base_col = 2 + size(X_topsis,2);

    % 三目标预测值
    for j = 1:3
        topsis_data_cell{i,base_col+j} = Y_topsis(idx,j);
    end

    base_col = base_col + 3;

    % 归一化指标
    for j = 1:3
        topsis_data_cell{i,base_col+j} = Z(idx,j);
    end

    base_col = base_col + 3;

    topsis_data_cell{i,base_col+1} = D_best(idx);
    topsis_data_cell{i,base_col+2} = D_worst(idx);
    topsis_data_cell{i,base_col+3} = topsis_score(idx);

end

topsis_cell = [topsis_header; topsis_data_cell];

writecell(topsis_cell,multi_file,'Sheet','TOPSIS评分结果');

%% ========== 4. 导出TOP3最佳解 ==========

top3_cell = [topsis_header; topsis_data_cell(1:length(best3_id),:)];

writecell(top3_cell,multi_file,'Sheet','TOPSIS最佳3个解');

disp('========== TOPSIS评分最佳3个解 ==========')
disp(top3_cell)

%% ========== 5. 绘制TOPSIS评分排序图 ==========

fig_topsis = figure('Position',[100,100,950,520]);
set(fig_topsis,'Color','w')

bar(score_sort,'FaceAlpha',0.85)

xlabel('Pareto解排名','FontName','Microsoft YaHei','FontSize',12)
ylabel('TOPSIS得分','FontName','Microsoft YaHei','FontSize',12)

title('合并非支配Pareto解集 TOPSIS评分排序', ...
    'FontName','Microsoft YaHei','FontSize',15,'FontWeight','bold')

grid on
box on

set(gca,'FontName','Microsoft YaHei','FontSize',11,'LineWidth',1.1)

try
    gca.Toolbar.Visible = 'off';
catch
end

drawnow
exportgraphics(fig_topsis,'TOPSIS评分排序图.png','Resolution',600);

disp(['TOPSIS评分结果已写入文件：',multi_file])
disp('========== TOPSIS评分完成 ==========')
%% 
%% ========== 显示TOPSIS最佳解在Pareto前沿中的位置 ==========

disp('========== 开始绘制TOPSIS最佳解在Pareto前沿中的位置 ==========')

% TOPSIS最佳3个解在原始合并非支配解集中的索引
best3_id = score_id(1:min(3,length(score_id)));

Y_best3 = Y_topsis(best3_id,:);
X_best3 = X_topsis(best3_id,:);
alg_best3 = alg_topsis(best3_id);
score_best3 = topsis_score(best3_id);

%% ========== 1. 三维Pareto前沿 + TOP3最佳解标注：颜色一致 + 美观标注 ==========

fig_best = figure('Position',[100,100,1080,780]);
set(fig_best,'Color','w')

ax_best = axes(fig_best);
hold(ax_best,'on')

%%颜色设置：与第10步/第11步保持一致
% 优先沿用第10步中的 ax.ColorOrder
if exist('ax','var') && isvalid(ax)
    color_order = ax.ColorOrder;
else
    color_order = get(groot,'defaultAxesColorOrder');
end

color_nsga  = color_order(1,:);
color_moead = color_order(2,:);
color_mopso = color_order(3,:);


idx_nsga = strcmp(alg_topsis,'NSGA-II');
idx_moead = strcmp(alg_topsis,'MOEA/D');
idx_mopso = strcmp(alg_topsis,'MOPSO');

s1 = scatter3(ax_best, ...
    Y_topsis(idx_nsga,1), ...
    Y_topsis(idx_nsga,2), ...
    Y_topsis(idx_nsga,3), ...
    46, ...
    'filled', ...
    'MarkerFaceColor',color_nsga, ...
    'MarkerEdgeColor',color_nsga, ...
    'MarkerFaceAlpha',0.35, ...
    'MarkerEdgeAlpha',0.35, ...
    'DisplayName','NSGA-II');

s2 = scatter3(ax_best, ...
    Y_topsis(idx_moead,1), ...
    Y_topsis(idx_moead,2), ...
    Y_topsis(idx_moead,3), ...
    46, ...
    'filled', ...
    'MarkerFaceColor',color_moead, ...
    'MarkerEdgeColor',color_moead, ...
    'MarkerFaceAlpha',0.35, ...
    'MarkerEdgeAlpha',0.35, ...
    'DisplayName','MOEA/D');

s3 = scatter3(ax_best, ...
    Y_topsis(idx_mopso,1), ...
    Y_topsis(idx_mopso,2), ...
    Y_topsis(idx_mopso,3), ...
    46, ...
    'filled', ...
    'MarkerFaceColor',color_mopso, ...
    'MarkerEdgeColor',color_mopso, ...
    'MarkerFaceAlpha',0.35, ...
    'MarkerEdgeAlpha',0.35, ...
    'DisplayName','MOPSO');



s_top = scatter3(ax_best, ...
    Y_best3(:,1), ...
    Y_best3(:,2), ...
    Y_best3(:,3), ...
    230, ...
    'p', ...
    'filled', ...
    'MarkerFaceColor',[1.00 0.84 0.00], ...
    'MarkerEdgeColor','k', ...
    'LineWidth',1.5, ...
    'DisplayName','TOPSIS最佳3个解');



x_range = max(Y_topsis(:,1)) - min(Y_topsis(:,1));
y_range = max(Y_topsis(:,2)) - min(Y_topsis(:,2));
z_range = max(Y_topsis(:,3)) - min(Y_topsis(:,3));

dx = 0.025 * x_range;
dy = 0.025 * y_range;
dz = 0.025 * z_range;

for i = 1:size(Y_best3,1)

    % 每个TOP解稍微不同方向偏移，减少重叠
    offset_x = dx * i;
    offset_y = dy * (-1)^(i);
    offset_z = dz * (1.2 - 0.25*i);

    text(ax_best, ...
        Y_best3(i,1) + offset_x, ...
        Y_best3(i,2) + offset_y, ...
        Y_best3(i,3) + offset_z, ...
        ['TOP',num2str(i)], ...
        'FontName','Microsoft YaHei', ...
        'FontSize',12, ...
        'FontWeight','bold', ...
        'Color','k', ...
        'BackgroundColor','w', ...
        'Margin',3, ...
        'EdgeColor',[0.25 0.25 0.25]);

    % 画一条短连线指向对应点
    plot3(ax_best, ...
        [Y_best3(i,1), Y_best3(i,1)+offset_x], ...
        [Y_best3(i,2), Y_best3(i,2)+offset_y], ...
        [Y_best3(i,3), Y_best3(i,3)+offset_z], ...
        'k-', ...
        'LineWidth',0.8, ...
        'HandleVisibility','off');

end

% 坐标轴与标题

xlabel(ax_best,[target_name{1}, ' / 越大越好'], ...
    'FontName','Microsoft YaHei','FontSize',12)

ylabel(ax_best,[target_name{2}, ' / 越小越好'], ...
    'FontName','Microsoft YaHei','FontSize',12)

zlabel(ax_best,[target_name{3}, ' / 越小越好'], ...
    'FontName','Microsoft YaHei','FontSize',12)

title(ax_best,'TOPSIS最佳解在合并非支配Pareto前沿中的位置', ...
    'FontName','Microsoft YaHei','FontSize',15,'FontWeight','bold')

legend(ax_best,[s1,s2,s3,s_top], ...
    {'NSGA-II','MOEA/D','MOPSO','TOPSIS最佳3个解'}, ...
    'Location','best', ...
    'FontName','Microsoft YaHei', ...
    'FontSize',10, ...
    'Box','off')

grid(ax_best,'on')
box(ax_best,'on')
view(ax_best,135,28)

set(ax_best, ...
    'FontName','Microsoft YaHei', ...
    'FontSize',11, ...
    'LineWidth',1.1, ...
    'TickDir','out')

try
    ax_best.Toolbar.Visible = 'off';
catch
end

drawnow
exportgraphics(fig_best,'TOPSIS最佳解_Pareto前沿位置.png','Resolution',600);


%% ========== 2. 二维投影图：TOP3位置 ==========

fig_best2 = figure('Position',[100,100,1250,420]);
set(fig_best2,'Color','w')

% 贯入度 - 磨损压力
ax1 = subplot(1,3,1);
hold(ax1,'on')
scatter(ax1,Y_topsis(:,1),Y_topsis(:,2),35,[0.65 0.65 0.65],'filled')
scatter(ax1,Y_best3(:,1),Y_best3(:,2),120,'p','filled','MarkerEdgeColor','k')
for i = 1:size(Y_best3,1)
    text(ax1,Y_best3(i,1),Y_best3(i,2),[' TOP',num2str(i)], ...
        'FontName','Microsoft YaHei','FontSize',10,'FontWeight','bold')
end
xlabel(ax1,[target_name{1}, ' / 越大越好'],'FontName','Microsoft YaHei')
ylabel(ax1,[target_name{2}, ' / 越小越好'],'FontName','Microsoft YaHei')
title(ax1,[target_name{1}, ' - ', target_name{2}],'FontName','Microsoft YaHei')
grid(ax1,'on')
box(ax1,'on')

% 贯入度 - 总功率
ax2 = subplot(1,3,2);
hold(ax2,'on')
scatter(ax2,Y_topsis(:,1),Y_topsis(:,3),35,[0.65 0.65 0.65],'filled')
scatter(ax2,Y_best3(:,1),Y_best3(:,3),120,'p','filled','MarkerEdgeColor','k')
for i = 1:size(Y_best3,1)
    text(ax2,Y_best3(i,1),Y_best3(i,3),[' TOP',num2str(i)], ...
        'FontName','Microsoft YaHei','FontSize',10,'FontWeight','bold')
end
xlabel(ax2,[target_name{1}, ' / 越大越好'],'FontName','Microsoft YaHei')
ylabel(ax2,[target_name{3}, ' / 越小越好'],'FontName','Microsoft YaHei')
title(ax2,[target_name{1}, ' - ', target_name{3}],'FontName','Microsoft YaHei')
grid(ax2,'on')
box(ax2,'on')

% 磨损压力 - 总功率
ax3 = subplot(1,3,3);
hold(ax3,'on')
scatter(ax3,Y_topsis(:,2),Y_topsis(:,3),35,[0.65 0.65 0.65],'filled')
scatter(ax3,Y_best3(:,2),Y_best3(:,3),120,'p','filled','MarkerEdgeColor','k')
for i = 1:size(Y_best3,1)
    text(ax3,Y_best3(i,2),Y_best3(i,3),[' TOP',num2str(i)], ...
        'FontName','Microsoft YaHei','FontSize',10,'FontWeight','bold')
end
xlabel(ax3,[target_name{2}, ' / 越小越好'],'FontName','Microsoft YaHei')
ylabel(ax3,[target_name{3}, ' / 越小越好'],'FontName','Microsoft YaHei')
title(ax3,[target_name{2}, ' - ', target_name{3}],'FontName','Microsoft YaHei')
grid(ax3,'on')
box(ax3,'on')

drawnow
exportgraphics(fig_best2,'TOPSIS最佳解_Pareto二维投影位置.png','Resolution',600);


%% ========== 3. 导出TOP3在Pareto前沿中的位置信息 ==========

position_header = [{'TOPSIS排名','算法来源'}, ...
    feature_name, ...
    strcat(target_name,'(预测)'), ...
    {'TOPSIS得分'}];

position_cell = cell(size(Y_best3,1)+1,length(position_header));
position_cell(1,:) = position_header;

for i = 1:size(Y_best3,1)

    position_cell{i+1,1} = i;
    position_cell{i+1,2} = alg_best3{i};

    for j = 1:size(X_best3,2)
        position_cell{i+1,2+j} = X_best3(i,j);
    end

    base_col = 2 + size(X_best3,2);

    for j = 1:3
        position_cell{i+1,base_col+j} = Y_best3(i,j);
    end

    position_cell{i+1,base_col+4} = score_best3(i);

end

writecell(position_cell,multi_file,'Sheet','TOPSIS最佳解位置');

disp('========== TOPSIS最佳解位置绘制完成 ==========')
