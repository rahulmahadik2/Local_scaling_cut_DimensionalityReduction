%{
We aim to implement Local Scaling Cut using various distance metrics such
as Euclidean, Minkowski, Cityblock, Cosine, Spearman etc.

The method is adapted from the following paper.
1. Mohanty, Ramanarayan, S. L. Happy, and Aurobinda Routray. "Graph scaling 
cut with l1-norm for classification of hyperspectral images." In 2017 25th 
European Signal Processing Conference (EUSIPCO), pp. 793-797. IEEE, 2017.

The local maximum is found using fmincon.

Fault are annotated by Gagandeep Singh.

Author: Rahul A. Mahadik and Aurobinda Routray
Department: Department of Electrical Engineering
Institute: Indian Institute of Technology, Kharagpur
Date: 08th September 2021
Last updated: 8th September 2021
%}
clc;
clearvars;
close all;
commandwindow

%% Read the patches coordinate points (time, crossline and inline information)
pathname1 = 'F:\Fault detection\Automatic Fault Extraction';
pathname2 = 'codes\Machine Learning Codes\Fault Labelling';
pathname = fullfile(pathname1, pathname2);
% filename = 'patch_extraction_coords.xlsx';
% xlfilepath = fullfile(pathname, filename);
% sheetnames = ["inline_1", "inline_10", "inline_11", "inline_2",...
%                 "inline_3", "inline_4", "inline_5"];
% coordpoints = cell(length(sheetnames),1);
% jj = 0;
% for ii = sheetnames
%     jj = jj + 1;
%     coordpoints{jj} = readmatrix(xlfilepath, 'Range', 'B:E', 'Sheet', ii);
% end
load('F:\Fault detection\Automatic Fault Extraction\codes\Machine Learning Codes\Dimensionality Reduction\required MAT files\coordpoints.mat')

%% extract Output label patches of seismic data
matfiles = dir([pathname, '\*.mat']);
faultpatches = [];
for onefile = 1:length(matfiles)
%    disp(matfiles(onefile).name)
   load([matfiles(onefile).folder, '/', matfiles(onefile).name])
   inl_coordpoints = coordpoints{onefile};
   for kk = 1:size(inl_coordpoints, 1)
       faultpatch = faultImage(inl_coordpoints(kk,1):inl_coordpoints(kk,2),...
                               inl_coordpoints(kk,3):inl_coordpoints(kk,4));
       indxs = find(faultpatch > 0);
       if isempty(indxs)
           disp(['filename:', matfiles(onefile).name])
           disp(['time range:', num2str(inl_coordpoints(kk,1))])
           disp(['inline range:', num2str(inl_coordpoints(kk,3))])
           fprintf('\n\n')
       end
       figure(1), imshow(faultpatch)
       faultpatches = vertcat(faultpatches,faultpatch(:));
   end
end

%% Choose the number of fault and non fault samples randomly
numelements = 10000;
faultindx = find(faultpatches>0);
allfaultindices = randperm(length(faultindx)); % get the randomly-selected indices
selected_faultindx = faultindx(allfaultindices(1:numelements));
nonfaultindx = find(faultpatches == 0);
allnonfaultindices = randperm(length(nonfaultindx)); % get the randomly-selected indices
selected_nonfaultindx = nonfaultindx(allnonfaultindices(1:numelements)); % choose the subset

%% read feature data
featuredatapath1 = 'F:\Fault detection\Automatic Fault Extraction\codes';
featuredatapath2 = 'Machine Learning Codes\Feature data_Rahul\Real data features';
featuredatapath = fullfile(featuredatapath1, featuredatapath2);
featurematfiles = dir([featuredatapath, '\*.mat']);
featurematrix = [];
for onefeatfile = 1 : length(featurematfiles)
   cohinlstruct = load([featurematfiles(onefeatfile).folder, '\',...
                        featurematfiles(onefeatfile).name]);
   cohinls = struct2cell(cohinlstruct);
   allpatches = [];
   for ii = 1:length(cohinls)
       inlpatches = coordpoints{ii};
       cohmat = cohinls{ii};
       for kk = 1:size(inlpatches, 1)
           patches = cohmat(inlpatches(kk,1):inlpatches(kk,2),...
                            inlpatches(kk,3):inlpatches(kk,4));
           
           allpatches = vertcat(allpatches, patches(:));
       end
   end
   featurematrix = horzcat(featurematrix, allpatches);
end
featurematrix = featurematrix.';
% featurecoluns = [1,2,3,4,5,12,13,22,24,25];
% featurematrix = featurematrix(featurecoluns, :);
featurmatrix_fault = featurematrix(:, selected_faultindx);
featurmatrix_nonfault = featurematrix(:, selected_nonfaultindx);

fault_samples = length(selected_faultindx);
nonfault_samples = length(selected_nonfaultindx);
num_features = size(featurematrix, 1);
featurmatrix_org = [featurmatrix_fault, featurmatrix_nonfault];
featurmatrix_org1 = featurmatrix_org;
labeled_org = [ones(1,fault_samples), zeros(1, nonfault_samples)];

%% Dimensionality reduction algorithm (Graph scaling cut with trace lasso regularization)
reduced_dim_num = 23;
num = 1;  K_nearest_sampls = 100; V_opt_mat = zeros(num_features, reduced_dim_num); 
dist_metric = 'euclidean';
vinit = rand(num_features, 1);
vinit = vinit/norm(vinit, 2);
exitflag = zeros(reduced_dim_num, 1);
all_iteration = zeros(reduced_dim_num, 1);
all_funccounts = zeros(reduced_dim_num, 1);
if reduced_dim_num > num_features
    error(['Please select the reduced dimension number less than number of features (',...
            num2str(num_features), ')'])
elseif reduced_dim_num == num_features
    warning('Reduced dimension number = Number of features.')
    warning('Only transfering to new subspace without dimensionality reduction')
end

while num <= reduced_dim_num
    kernelized = false;
    if kernelized
        sigma_f = 0.25;
        sigma_nf = 0.25;
        featurmatrix_fault = exp(-(featurmatrix_fault.^2)/(2*sigma_f^2));
        featurmatrix_nonfault = exp(-(featurmatrix_nonfault.^2)/(2*sigma_nf^2));
    end
    fprintf('\n Solving optimum vector: %d\n', num)
    % between class dissimilarity matrix 
    cov_betc_fault = zeros(num_features, fault_samples * K_nearest_sampls);
    cov_betc_nonfault = zeros(num_features, K_nearest_sampls * nonfault_samples);
%     wtbr = waitbar(0,'Please wait');
    for ii = 1 : fault_samples
        query_pt = featurmatrix_fault(:, ii).';
        Idxnfpts_betc = findclosepoints(featurmatrix_nonfault.', query_pt,...
                                            K_nearest_sampls, dist_metric);
        selectedfeaturmat_nonfault = featurmatrix_nonfault(:, Idxnfpts_betc);
        for jj = 1 : K_nearest_sampls
            cov_betc_fault(:, K_nearest_sampls * (ii - 1) + jj) = (featurmatrix_fault(:, ii)...
                                                           - selectedfeaturmat_nonfault(:, jj));
        end
%         waitbar(ii/fault_samples, wtbr, 'between class for fault points')
    end
%     close(wtbr)

%     wtbr = waitbar(0,'Please wait');
    for ii = 1 : nonfault_samples
        query_pt = featurmatrix_nonfault(:, ii).';
        Idxfpts_betc = findclosepoints(featurmatrix_fault.', query_pt,...
                                        K_nearest_sampls, dist_metric);
        selectedfeaturmat_fault = featurmatrix_fault(:, Idxfpts_betc);
        for jj = 1 : K_nearest_sampls
            cov_betc_nonfault(:, K_nearest_sampls * (ii - 1) + jj) = (featurmatrix_nonfault(:, ii)...
                                                           - selectedfeaturmat_fault(:, jj));
        end
%         waitbar(ii/nonfault_samples, wtbr, 'between class for non-fault points')
    end
%     close(wtbr)
    cov_betc = [cov_betc_fault, cov_betc_nonfault];
    
    
    % within class dissimilarity matrix
    cov_withinc_fault = zeros(num_features, K_nearest_sampls * fault_samples);
    cov_withinc_nonfault = zeros(num_features, K_nearest_sampls * nonfault_samples);
%     wtbr = waitbar(0,'Please wait');
    for kk = 1 : fault_samples
        query_pt = featurmatrix_fault(:, ii).';
        Idxfpts_withinc = findclosepoints(featurmatrix_fault.', query_pt,...
                                        K_nearest_sampls, dist_metric);
        selectedfeaturmat_fault = featurmatrix_fault(:, Idxfpts_withinc);
        for ll = 1 : K_nearest_sampls
            cov_withinc_fault(:, K_nearest_sampls * (kk - 1) + ll) = (featurmatrix_fault(:, kk)...
                                                            - selectedfeaturmat_fault(:, ll));
        end
%         waitbar(kk/fault_samples, wtbr, 'within class for fault points')
    end
%     close(wtbr)

%     wtbr = waitbar(0,'Please wait');
    for kk = 1 : nonfault_samples
        query_pt = featurmatrix_nonfault(:, ii).';
        Idxnfpts_withinc = findclosepoints(featurmatrix_nonfault.', query_pt,...
                                        K_nearest_sampls, dist_metric);
        selectedfeaturmat_nonfault = featurmatrix_nonfault(:, Idxnfpts_withinc);
        for ll = 1 : K_nearest_sampls
            cov_withinc_nonfault(:, K_nearest_sampls * (kk - 1) + ll) = (featurmatrix_nonfault(:, kk)...
                                                            - selectedfeaturmat_nonfault(:, ll));
        end
%         waitbar(kk/nonfault_samples, wtbr, 'within class for non-fault points')
    end
%     close(wtbr)
    cov_withinc = [cov_withinc_fault, cov_withinc_nonfault];
    
    %%
    % Algorithm: 1. 'active-set' 2. 'interior-point' 3. 'sqp', 4. 'sqp-legacy'. 
    options = optimoptions('fmincon','Display', 'iter', 'MaxFunctionEvaluations', 25000,...
                        'StepTolerance', 1e-20, 'Algorithm', 'interior-point');
    problem.options = options;
    problem.solver = 'fmincon';
%     problem.objective = @(v) -(sum(abs(v.' * cov_betc))/sum(abs(v.' * cov_betc)));
    problem.objective = @(v) -(norm(v.' * cov_betc,1)/(norm(v.' * cov_betc,1) + norm(v.' * cov_withinc,1)));
    problem.x0 = vinit;
    problem.nonlcon = @nonlinconstraint;
    [vopt, fval, exitflag(num), output] = fmincon(problem);
    all_iteration(num) = output.iterations;
    all_funccounts(num) = output.funcCount;
    V_opt_mat(:, num) = vopt;
    num = num + 1;
    if num <= reduced_dim_num
        fprintf('\n\n')
        fprintf('Optimum vector found.\nUpdating the dataset...\n')
        featurmatrix_fault = featurmatrix_fault - ((vopt * vopt.') * featurmatrix_fault);
        featurmatrix_nonfault = featurmatrix_nonfault - ((vopt * vopt.') * featurmatrix_nonfault);
        disp('Dataset updated!!!')
    end
end

%% transform the data to dimension reduced subspace 
trans_feature_mat = V_opt_mat .' * featurematrix;
transfeaturmatrix_fault = trans_feature_mat(:, selected_faultindx);
transfeaturmatrix_nonfault = trans_feature_mat(:, selected_nonfaultindx);

%% Visualize the original dataset and the dimesion reduced dataset
feature1 = 1;
feature2 = 2;
feature3 = 4;
featurmatrix_fault_org1 = featurmatrix_org1(:, 1 : fault_samples);
featurmatrix_nonfault_org1 = featurmatrix_org1(:, 1 + fault_samples: nonfault_samples + fault_samples);
figure('Name', 'original feature matrix')
% plot3(featurmatrix_fault_org1(feature1, :), featurmatrix_fault_org1(feature2, :),...
%     featurmatrix_fault_org1(feature3, :),'ro')
plot(featurmatrix_fault_org1(feature1, :), featurmatrix_fault_org1(feature2, :), 'ro')
hold on
% plot3(featurmatrix_nonfault_org1(feature1, :), featurmatrix_nonfault_org1(feature2, :),...
%     featurmatrix_nonfault_org1(feature3, :), 'gx')
plot(featurmatrix_nonfault_org1(feature1, :), featurmatrix_nonfault_org1(feature2, :), 'go')
hold off
legend('Fault', 'Non-fault')
xlabel(['feature ', num2str(feature1)])
ylabel(['feature ', num2str(feature2)])
title('original feature matrix')

figure('Name', 'dimension reduced matrix')
% plot3(transfeaturmatrix_fault(feature1, :), transfeaturmatrix_fault(feature2, :),...
%     transfeaturmatrix_fault(feature3, :),'ro')
plot(transfeaturmatrix_fault(feature1, :), transfeaturmatrix_fault(feature2, :), 'ro')
hold on
% plot3(transfeaturmatrix_nonfault(feature1, :), transfeaturmatrix_nonfault(feature2, :),...
%     transfeaturmatrix_nonfault(feature3, :), 'gx')
plot(transfeaturmatrix_nonfault(feature1, :), transfeaturmatrix_nonfault(feature2, :), 'go')
hold off
legend('Fault', 'Non-fault')
xlabel(['feature ', num2str(feature1)])
ylabel(['feature ', num2str(feature2)])
title(['dimension reduced matrix using ', dist_metric, ' distance metric'])

%% t-stochastic neighbourhood embedding using various distance metrics
species1 = repmat({'fault'}, fault_samples, 1);
species2 = repmat({'non-fault'}, nonfault_samples, 1);
species = vertcat(species1, species2);
v = double(categorical(species));
c = full(sparse(1:numel(v), v, ones(size(v)), numel(v), 3));
% options = statset('MaxIter',5000);
trans_feature_mat_selected = [transfeaturmatrix_fault, transfeaturmatrix_nonfault]';

% figure(3)
% rng('default') % for reproducibility
% [Y, loss] = tsne(trans_feature_mat_selected, 'Algorithm', 'exact', 'Distance', 'mahalanobis', 'Exaggeration', 4, ...
%     'Standardize', true, 'LearnRate', 1000, 'NumPCAComponents', 10, 'NumDimensions', 3, 'Perplexity', 20);
% subplot(2,2,1)
% gscatter(Y(:,1),Y(:,2),species)
% % scatter3(Y(:,1),Y(:,2),Y(:,3),15,c,'filled'); 
% title('Mahalanobis')
% fprintf('Loss in Mahalanobis distance = %2.4f\n', loss)
figure(3)
rng('default') % for fair comparison
[Y, loss] = tsne(trans_feature_mat_selected, 'Algorithm', 'exact', 'Distance', 'cosine', 'Exaggeration', 4, ...
    'Standardize', true, 'LearnRate', 1000, 'NumPCAComponents', 10, 'NumDimensions', 3, 'Perplexity', 20);
gscatter(Y(:,1), Y(:,2), species, 'bg', '..', 8, 'on', 'Attribute 1', 'Attribute 2')
% scatter3(Y(:,1),Y(:,2),Y(:,3),15,c,'filled'); 
title('Cosine')
fprintf('Loss in Cosine distance = %2.4f\n', loss)

% rng('default') % for fair comparison
% [Y, loss] = tsne(trans_feature_mat_selected,'Algorithm','exact','Distance','chebychev', 'Exaggeration', 4, ...
%     'Standardize', true, 'LearnRate', 1000, 'NumPCAComponents', 10, 'NumDimensions', 3, 'Perplexity', 20);
% subplot(2,2,3)
% gscatter(Y(:,1),Y(:,2),species)
% % scatter3(Y(:,1),Y(:,2),Y(:,3),15,c,'filled'); 
% title('Chebychev')
% fprintf('Loss in Chebychev distance = %2.4f\n', loss)

% rng('default') % for fair comparison
% [Y, loss] = tsne(trans_feature_mat_selected,'Algorithm','exact','Distance','euclidean', 'Exaggeration', 4, ...
%     'Standardize', true, 'LearnRate', 1000, 'NumPCAComponents', 10, 'NumDimensions', 3, 'Perplexity', 20);
% subplot(2,2,4)
% gscatter(Y(:,1),Y(:,2),species)
% % scatter3(Y(:,1),Y(:,2),Y(:,3),15,c,'filled'); 
% title('Euclidean')
% fprintf('Loss in Euclidean distance = %2.4f\n', loss)

figure(4)
% rng('default') % for fair comparison
% [Y, loss] = tsne(trans_feature_mat_selected,'Algorithm','exact','Distance','cityblock', 'Exaggeration', 4, ...
%     'Standardize', true, 'LearnRate', 1000, 'NumPCAComponents', 10, 'NumDimensions', 3, 'Perplexity', 20);
% subplot(2,2,1)
% gscatter(Y(:,1),Y(:,2),species)
% % scatter3(Y(:,1),Y(:,2),Y(:,3),15,c,'filled'); 
% title('cityblock')
% fprintf('Loss in cityblock distance = %2.4f\n', loss)

rng('default') % for fair comparison
[Y, loss] = tsne(trans_feature_mat_selected,'Algorithm','exact','Distance','spearman', 'Exaggeration', 4, ...
    'Standardize', true, 'LearnRate', 1000, 'NumPCAComponents', 10, 'NumDimensions', 3, 'Perplexity', 20);
% subplot(2,2,2)
gscatter(Y(:,1), Y(:,2), species, 'bg', '..', 8, 'on', 'Attribute 1', 'Attribute 2')
% scatter3(Y(:,1),Y(:,2),Y(:,3),15,c,'filled'); 
title('spearman')
fprintf('Loss in spearman distance = %2.4f\n', loss)

% rng('default') % for fair comparison
% [Y, loss] = tsne(trans_feature_mat_selected,'Algorithm','exact','Distance','correlation', 'Exaggeration', 4, ...
%     'Standardize', true, 'LearnRate', 1000, 'NumPCAComponents', 10, 'NumDimensions', 3, 'Perplexity', 20);
% subplot(2,2,3)
% gscatter(Y(:,1),Y(:,2),species)
% % scatter3(Y(:,1),Y(:,2),Y(:,3),15,c,'filled'); 
% title('correlation')
% fprintf('Loss in correlation distance = %2.4f\n', loss)

% rng('default') % for fair comparison
% [Y, loss] = tsne(trans_feature_mat_selected,'Algorithm','exact','Distance','minkowski', 'Exaggeration', 4, ...
%     'Standardize', true, 'LearnRate', 1000, 'NumPCAComponents', 10, 'NumDimensions', 3, 'Perplexity', 20);
% subplot(2,2,4)
% gscatter(Y(:,1),Y(:,2),species)
% % scatter3(Y(:,1),Y(:,2),Y(:,3),15,c,'filled'); 
% title('minkowski')
% fprintf('Loss in minkowski distance = %2.4f\n', loss)


figure(5)
rng('default') % for fair comparison
[Y, loss] = tsne(featurmatrix_org1', 'Algorithm', 'exact', 'Distance', 'cosine', 'Exaggeration', 4, ...
    'Standardize', true, 'LearnRate', 1000, 'NumPCAComponents', 10, 'NumDimensions', 3, 'Perplexity', 20);
% subplot(2,1,1)
gscatter(Y(:,1), Y(:,2), species, 'bg', '..', 8, 'on', 'Attribute 1', 'Attribute 2')
% scatter3(Y(:,1),Y(:,2),Y(:,3),15,c,'filled'); 
title('Cosine on original data')
fprintf('Loss in Cosine distance on original data= %2.4f\n', loss)

% rng('default') % for fair comparison
% [Y, loss] = tsne(trans_feature_mat_selected, 'Algorithm', 'exact', 'Distance', @canberrametric, 'Exaggeration', 4, ...
%     'Standardize', true, 'LearnRate', 1000, 'NumPCAComponents', 10, 'NumDimensions', 3, 'Perplexity', 20);
% subplot(2,1,2)
% gscatter(Y(:,1),Y(:,2),species)
% % scatter3(Y(:,1),Y(:,2),Y(:,3),15,c,'filled'); 
% title('canberra distance metric')
% fprintf('Loss in canberra distance = %2.4f\n', loss)

%% Supportive functions
function [c,ceq] = nonlinconstraint(x)
c = [];
ceq = norm(x, 2).^2 - 1;
end

function dist = canberrametric(p,q)
    if size(p,2) ~= size(q,2)
        error('The length of two vectors are not same!!!')
    end
    
        dist = sum(abs(p-q)./(abs(p) + abs(q)), 2);
    
end

function Idx = findclosepoints(search_matrix, query_pt, num_nearest_pts, dist_metric)
    switch dist_metric
        case 'seuclidean' 
            % Standardized Euclidean distance. Each coordinate difference 
            % between rows in X and the query matrix Y is scaled by dividing 
            % by the corresponding element of the standard deviation 
            % computed from X. To specify another scaling, use the 'Scale' 
            % name-value pair argument.
            Idx = knnsearch(search_matrix, query_pt, 'K', num_nearest_pts, ...
                'IncludeTies', false, 'NSMethod', 'exhaustive', 'Distance',...
                'seuclidean', 'Scale', quantile(search_matrix,0.9) - ...
                quantile(search_matrix,0.25));
            
        case 'euclidean' % Euclidean distance.
            Idx = knnsearch(search_matrix, query_pt, 'K', num_nearest_pts, ...
                'IncludeTies', false, 'NSMethod', 'exhaustive', 'Distance',...
                'euclidean');
            
        case 'cityblock' % City block distance.
            Idx = knnsearch(search_matrix, query_pt, 'K', num_nearest_pts, ...
                'IncludeTies', false, 'NSMethod', 'exhaustive', 'Distance',...
                'cityblock');
            
        case 'chebychev' % Chebychev distance (maximum coordinate difference).
            Idx = knnsearch(search_matrix, query_pt, 'K', num_nearest_pts, ...
                'IncludeTies', false, 'NSMethod', 'exhaustive', 'Distance',...
                'chebychev');
            
        case 'minkowski'	
            % Minkowski distance. The default exponent is 2 (euclidean). 
            % To specify a different exponent, use the 'P' name-value pair argument.
            Idx = knnsearch(search_matrix, query_pt, 'K', num_nearest_pts, ...
                'IncludeTies', false, 'NSMethod', 'exhaustive', 'Distance',...
                 'minkowski', 'P', 5);
            
        case 'mahalanobis'	
            % Mahalanobis distance, computed using a positive definite 
            % covariance matrix. To change the value of the covariance matrix, 
            % use the 'Cov' name-value pair argument.
            Idx = knnsearch(search_matrix, query_pt, 'K', num_nearest_pts, ...
                'IncludeTies', false, 'NSMethod', 'exhaustive', 'Distance',...
                'mahalanobis');
            
        case 'cosine'	
            % One minus the cosine of the included angle between observations 
            % (treated as vectors).
            Idx = knnsearch(search_matrix, query_pt, 'K', num_nearest_pts, ...
                'IncludeTies', false, 'NSMethod', 'exhaustive', 'Distance',...
                'cosine');

        case 'correlation'	
            % One minus the sample linear correlation between observations 
            % (treated as sequences of values).
            Idx = knnsearch(search_matrix, query_pt, 'K', num_nearest_pts, ...
                'IncludeTies', false, 'NSMethod', 'exhaustive', 'Distance',...
                'correlation');
            
        case 'spearman'	
            % One minus the sample Spearman's rank correlation 
            % between observations (treated as sequences of values).
            Idx = knnsearch(search_matrix, query_pt, 'K', num_nearest_pts, ...
                'IncludeTies', false, 'NSMethod', 'exhaustive', 'Distance',...
                'spearman');
            
        case 'hamming'	
            % Hamming distance, which is the percentage of coordinates that differ.
            Idx = knnsearch(search_matrix, query_pt, 'K', num_nearest_pts, ...
                'IncludeTies', false, 'NSMethod', 'exhaustive', 'Distance',...
                'hamming');
	
        case 'jaccard'
            % One minus the Jaccard coefficient, which is the percentage of 
            % nonzero coordinates that differ.
            Idx = knnsearch(search_matrix, query_pt, 'K', num_nearest_pts, ...
                'IncludeTies', false, 'NSMethod', 'exhaustive', 'Distance',...
                'jaccard');
            
        case 'canberra'
            % One minus the Jaccard coefficient, which is the percentage of 
            % nonzero coordinates that differ.
            Idx = knnsearch(search_matrix, query_pt, 'K', num_nearest_pts, ...
                'IncludeTies', false, 'NSMethod', 'exhaustive', 'Distance',...
                @canberrametric);
            
        otherwise
            error('Unknown distance metric. Please check again !!!')
    end
	
end