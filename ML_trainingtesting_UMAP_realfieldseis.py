# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 00:01:44 2021

@author: IIT
"""
#%% Load librariers
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import time
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from skimage.morphology import remove_small_objects, binary_opening, binary_dilation, binary_erosion
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.manifold import TSNE
import umap

print('Libraries loaded successfully!!!')

#%% Load Feature data and the optimized projection matrix V
pathname = "F:\\Fault detection\\Automatic Fault Extraction\\codes\\Machine Learning Codes\\Dimensionality Reduction\\required MAT files" 
# localscaling_4000samp_k900.mat; localscaling_3000samp_k900.mat; 
# localscaling_2000samp_k800.mat; localscaling_1600samp_k700.mat;
# localscaling_10000samp_k100.mat
filename = 'localscaling_4000samp_k900.mat' 
var_mats = loadmat(os.path.join(pathname,filename))
V_opt_mat = var_mats['V_opt_mat']
q, r = np.linalg.qr(V_opt_mat, mode='reduced')
V_opt_mat = q.copy()
num_dim_proj_vecs = 23
V_opt_mat = V_opt_mat[:, 0:num_dim_proj_vecs]
featurmatrix_org1 = var_mats['featurmatrix_org1']
total_samples = featurmatrix_org1.shape[1]
fault_samples = total_samples//2
nonfault_samples = total_samples//2
labeled_org = np.hstack((np.ones((fault_samples,)), np.zeros((nonfault_samples,))))
transformed_featmat = np.matmul(V_opt_mat.T, featurmatrix_org1)

print('The shape of transformed featmat is ' + str(transformed_featmat.shape))

#%% Kernal PCA on transformed data
distance_metric = 'cosine'
# kernPCA_model = KernelPCA(n_components=10, kernel='rbf', 
#                         gamma=0.058, fit_inverse_transform=True, alpha=100,
#                   random_state=42)
kernPCA_model = KernelPCA(n_components=10, kernel='sigmoid', random_state=42)
transfeaturmatrix_PCA = kernPCA_model.fit_transform(transformed_featmat.T)

#%% UMAP on transformed data using Local scaling cut and PCA
# UMAP_model = umap.UMAP(n_neighbors=50, min_dist=0.0151, n_components=3, 
#                         metric=distance_metric, learning_rate=0.01, 
#                         random_state=42, verbose=True, 
#                         n_epochs=2000)
UMAP_model = umap.UMAP(n_neighbors=50, min_dist=0.0151, n_components=3, 
                        metric=distance_metric, learning_rate=1e-2, 
                        random_state=42, verbose=True, 
                        n_epochs=2000)
Y_embedded_umap_LSC = UMAP_model.fit_transform(transfeaturmatrix_PCA);
print('The shape of Y_embedded_LSC is ' + str(Y_embedded_umap_LSC.shape))

transformed_fault_umap_LSC = Y_embedded_umap_LSC[0 : fault_samples, :]
transformed_nonfault_umap_LSC = Y_embedded_umap_LSC[fault_samples: nonfault_samples 
                                                    + fault_samples, :]

#%% Plot the UMAP data
def scatterplotfaultnonfault(transformed_faultdata, transformed_nonfaultdata, 
                             features2plot):
    if transformed_faultdata.shape[1] == 2:
        plt.plot(transformed_faultdata[:, features2plot[0]], 
                 transformed_faultdata[:, features2plot[1]], 'ro')
        plt.plot(transformed_nonfaultdata[:, features2plot[0]], 
                 transformed_nonfaultdata[:, features2plot[1]], 'go')
        plt.xlabel('Attribute 1'), plt.ylabel('Attribute 2')
    else:
        ax = plt.axes(projection='3d')
        ax.plot3D(transformed_faultdata[:, features2plot[0]], 
                  transformed_faultdata[:, features2plot[1]], 
                   transformed_faultdata[:, features2plot[2]], 'ro')
        ax.plot3D(transformed_nonfaultdata[:, features2plot[0]], 
                  transformed_nonfaultdata[:, features2plot[1]], 
                   transformed_nonfaultdata[:, features2plot[2]], 'go')
        ax.set_xlabel('Attribute 1')
        ax.set_ylabel('Attribute 2')
        ax.set_zlabel('Attribute 3')
    plt.legend(["fault samples", "non-fault samples"], loc ="upper right")
        
features2plot = (0, 1, 2)
plt.figure()
scatterplotfaultnonfault(transformed_fault_umap_LSC, 
                          transformed_nonfault_umap_LSC, features2plot)
plt.title('UMAP on Local scaling cut Data "'+ distance_metric + '" distance metric')
plt.show()

#%% Support Vector Machine training on UMAP data
print('Training the UMAP dimensionality reduced labelled data using SVM...')
X_train_umap = Y_embedded_umap_LSC
y_train_umap = labeled_org

# start_time = time.time()
# SVMclassifier = SVC(C = 70.0, kernel = 'sigmoid', gamma = 5e-4, 
#               verbose = True, probability=True)
SVMclassifier = SVC(C = 70.0, kernel = 'sigmoid', gamma = 5e-4, 
              verbose = True, probability=True)
SVMclassifier.fit(X_train_umap, y_train_umap)
# print("SVM training time UMAP--- %s seconds ---" % (time.time() - start_time))

#%% K nearest Neighbour on UMAP data
print('Training the UMAP dimensionality reduced labelled data using KNN...')
X_train_umap_knn = Y_embedded_umap_LSC
y_train_umap_knn = labeled_org
start_time = time.time()
knnclassifier = KNeighborsClassifier(n_neighbors = 50, 
                                  metric = 'cosine')
knnclassifier.fit(X_train_umap_knn, y_train_umap_knn)
print("KNN training time UMAP--- %s seconds ---" % (time.time() - start_time))

#%% Random Forest algorithm on on UMAP data
print('Training the UMAP dimensionality reduced labelled data using Random forest...')
X_train_umap_RF = Y_embedded_umap_LSC
y_train_umap_RF = labeled_org
start_time = time.time()
rfclassifier = RandomForestClassifier(n_estimators=500, 
                       random_state=42, verbose=0, 
                       max_samples=100)
rfclassifier.fit(X_train_umap_RF, y_train_umap_RF)
print("Random Forest training time UMAP--- %s seconds ---" % (time.time() - start_time))


#%% read Real field feature data and its corresponding labels
filename_realseis = 'fieldseis_features_crop_BigData_2_specbal.mat' 
var_featsLabelsmats = loadmat(os.path.join(pathname,filename_realseis))

#%% extract feature matrix and its corresponding fault labels
fault_labels = var_featsLabelsmats['faultpatches']
realseis_featurematrix = var_featsLabelsmats['featurematrix']

patch_num = 10 #25, 54, 55, 10
test_feat = realseis_featurematrix[:, 40000*patch_num : 40000*(patch_num+1)]
test_label = fault_labels[40000*patch_num : 40000*(patch_num + 1)]

test_seis_groundtruth = np.reshape(test_label, (200,200), order = 'F')
plt.figure()
plt.imshow(test_seis_groundtruth, cmap = 'gray')

#%% Transform the test feature matrix to another subspace
transformed_test_feat = np.matmul(V_opt_mat.T, test_feat)
num_testsamples = transformed_test_feat.shape[1]

#%% Predict the output on transformed subspace using SVM, KNN and Random forest
rf_faultseis = []
knn_faultseis = []
svm_faultseis = []
for ii in range(0, num_testsamples, total_samples):
    if (ii + total_samples) < num_testsamples:
        patch = transformed_test_feat[:, ii:ii+total_samples]
        ii_range = (ii, ii + total_samples)
    else:
        patch = transformed_test_feat[:, ii:]
        ii_range = (ii, num_testsamples)
        
    patch_pca = kernPCA_model.fit_transform(patch.T)
    
    patch_umap = UMAP_model.fit_transform(patch_pca);
    # features2plot = (0, 1, 2)
    # plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot3D(patch_umap[:, features2plot[0]], 
    #           patch_umap[:, features2plot[1]], 
    #           patch_umap[:, features2plot[2]], 'ro')
    # ax.set_xlabel('Feature 1')
    # ax.set_ylabel('Feature 2')
    # ax.set_zlabel('Feature 3')
    rf_patch = rfclassifier.predict_proba(patch_umap)
    knn_patch = knnclassifier.predict_proba(patch_umap)
    svm_patch = SVMclassifier.predict_proba(patch_umap)
    
    rf_patch = rf_patch[:, 1] # take fault probability
    knn_patch = knn_patch[:, 1] # take fault probability
    svm_patch = svm_patch[:, 1] # take fault probability
    
    rf_faultseis = np.concatenate((rf_faultseis, rf_patch), axis=0)
    knn_faultseis = np.concatenate((knn_faultseis, knn_patch), axis=0)
    svm_faultseis = np.concatenate((svm_faultseis, svm_patch), axis=0)

#%% Visualize real field data testing results using Random forest, 
# Support Vector Machine, K-NN
rf_faultseisdata = np.reshape(rf_faultseis, (200,200), order = 'F')
rf_faultseisdata_new = rf_faultseisdata.copy()
rf_faultseisdata_new[rf_faultseisdata_new < 0.5625] = 0.0
rf_faultseisdata_new[rf_faultseisdata_new >= 0.5625] = 1.0
fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].imshow(rf_faultseisdata, cmap = 'gray')
axs[0].set_title('Probabilistic Fault Prediction using Random Forest')
axs[1].imshow(rf_faultseisdata_new, cmap = 'gray')
axs[1].set_title('Fault Prediction using Random Forest using thresholding')


knn_faultseisdata = np.reshape(knn_faultseis, (200,200), order = 'F')
knn_faultseisdata_new = knn_faultseisdata.copy()
knn_faultseisdata_new[knn_faultseisdata_new < 0.5625] = 0.0
knn_faultseisdata_new[knn_faultseisdata_new >= 0.5625] = 1.0
fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].imshow(knn_faultseisdata, cmap = 'gray')
axs[0].set_title('Probabilistic Fault Prediction using K-nearest neighbour')
axs[1].imshow(knn_faultseisdata_new, cmap = 'gray')
axs[1].set_title('Fault Prediction using K-nearest neighbour using thresholding')


svm_faultseisdata = np.reshape(svm_faultseis, (200,200), order = 'F')
svm_faultseisdata_new = svm_faultseisdata.copy()
svm_faultseisdata_new[svm_faultseisdata_new < 0.5625] = 0.0
svm_faultseisdata_new[svm_faultseisdata_new >= 0.5625] = 1.0
fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].imshow(svm_faultseisdata, cmap = 'gray')
axs[0].set_title('Probabilistic Fault Prediction using Support Vector Machine')
axs[1].imshow(svm_faultseisdata_new, cmap = 'gray')
axs[1].set_title('Fault Prediction using Support Vector Machine using thresholding')


#%% Post Processing on Fault detected data
time, nxlines = svm_faultseisdata_new.shape
# newseisdata = svm_faultseisdata.copy()
newseisdata = np.zeros((time,nxlines))
time_step = 2
xline_step = 15
for ii in range(0, time):
    for jj in range(0, nxlines):
        window = svm_faultseisdata_new[ii:ii+time_step, jj:jj+xline_step]
        
        if np.sum(window)==time_step*xline_step:
            newseisdata[ii:ii+time_step, jj:jj+xline_step] = np.zeros(window.shape)
        else:
            newseisdata[ii:ii+time_step, jj:jj+xline_step] = window

newseisdata = newseisdata.astype(bool)
final_newseisdata = remove_small_objects(newseisdata, min_size=30, 
                                           connectivity=2)
final_newseisdata_org = final_newseisdata.copy()
# kernel = np.ones((25,1), dtype=bool)
kernel = np.identity(10, dtype=bool)
erode_im = binary_erosion(final_newseisdata_org, selem = kernel)

#%% Visualize the post processing results
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(svm_faultseisdata, cmap = 'gray')
axs[0, 0].set_title('Fault Prediction using Support Vector Machine')

axs[0, 1].imshow(newseisdata, cmap = 'gray')
axs[0, 1].set_title('Post processing on fault predicetd data using SVM')

# axs[1, 0].imshow(final_newseisdata*test_seis_groundtruth, cmap = 'gray')
# axs[1, 0].set_title('Removing small pixel and spurious events')

axs[1, 0].imshow(final_newseisdata, cmap = 'gray')
axs[1, 0].set_title('Removing small pixel and spurious events')

axs[1, 1].imshow(erode_im, cmap = 'gray')
axs[1, 1].set_title('Erosion operation')

#%% 
plt.figure()
plt.imshow(final_newseisdata*test_seis_groundtruth, cmap = 'gray')
