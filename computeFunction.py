from collections import Counter
import numpy as np 
import cv2
import time, json
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os, time, json
import joblib
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.color import rgb2lab
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
import sys
from numpy import array, reshape, shape, matrix, ones, zeros, sqrt, sort, arange
from numpy import nonzero, fromfile, tile, append, prod, double, argsort, sign
from numpy import kron, multiply, divide, abs, reshape, asarray
from scipy import rand
from scipy.sparse import csc_matrix, spdiags
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm, svd, LinAlgError
import os
import xml.etree.ElementTree as ET
import cv2
import math
import numpy as np
from sklearn.cluster import KMeans
from skimage import io
from sklearn import preprocessing
from collections import defaultdict
from scipy import stats
from skimage import io
from numpy.linalg import inv


def box_prior(train_data):
    saliency = saliency_map_from_set(train_data)
    v = []
    for s in saliency:
        v.append(np.mean(s))
    return np.array(v)

def test_and_make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def currentTime():
    return time.strftime("%H_%M_%S", time.localtime())

def test_postfix_dir(root):
    seplen = len(os.sep)
    if root[-seplen:] != os.sep:
        return root + os.sep
    return root

def save_opt(root, opt):
    json_dump(opt._get_kwargs(), root + "config.json")

def json_dump(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f)

class SpatialPyramidClassifier:
    """ 
    :params:
        M: size of vocabulary, number of clusters' centers.
        L: number of levels.
        cluster_batch: batch of samples used in training clusters
        do_train_clusters: train cluster or not. (if not, need `load_model` first)
        do_train_classifier: train classifier or not. (if not, need `load_model` first)
        max_num_sample_train_cluster: maximum number of samples used in training clusters, 
            since it only use the subset of trainset. 
        save_root: save folder
        use_matrix_in_kernel: when doing matching in kernel function, using Matrix Way or not. 
            Note that Matrix way **consume large memory**.
    """
    def __init__(self, M = 200, L = 2, cluster_batch = 1000, 
        do_train_clusters = True, do_train_classifier = True,
        max_num_sample_train_cluster = 20000, save_root = "save", 
        use_matrix_in_kernel = False, save_sift_feature = True):

        self.saveroot = test_postfix_dir(save_root)
        test_and_make_dir(self.saveroot)

        self.do_train_clusters = do_train_clusters
        self.do_train_classifier = do_train_classifier
        self.save_sift_feature = save_sift_feature

        self.use_matrix_in_kernel = use_matrix_in_kernel

        self.L = L 
        self.M = M
        self.num_cells = (4 **(L + 1) - 1) // 3
        self.sp_dim = M * self.num_cells # dim of spatial-pyramid-feature
        # self.clusters = KMeans(M) # clusters as vocabulary
        self.clusters_batch = cluster_batch
        self.clusters = MiniBatchKMeans(M, batch_size=self.clusters_batch)
        # self.classifier = SVC(kernel="precomputed")
        self.classifier = SVC(kernel=self.spatial_pyramid_matching_kernel)
        self.MAX_NUM_SAMPLE_TRAIN_CLUSTER  = max_num_sample_train_cluster # maximum number of training samples in training KMeans


    def save_model(self):
        joblib.dump(self.clusters, self.saveroot + "clusters_" + currentTime() + ".pt")
        joblib.dump(self.classifier, self.saveroot + "classifier_" + currentTime() + ".pt")

    def load_model(self, cluster_file = None, classifier_file= None):
        if cluster_file:
            self.clusters =  joblib.load(cluster_file)
        if classifier_file:
            self.classifier = joblib.load(classifier_file)

    def save_feature(self, feature):
        np.save(self.saveroot + "feature_" + currentTime() + ".npy", feature)

    def load_feature(self, filename):
        return np.load(filename)

    def _set_feature_hw(self, images):
        h, w = images.shape[1:]
        patch_size = (16, 16)
        step_size = 8

        coord_y = [y for y in range(step_size, h, step_size)]
        coord_x = [x for x in range(step_size, w, step_size)]
        self.num_feature_h = len(coord_y)
        self.num_feature_w = len(coord_x)

    def feature_dense_sift_one_image(self, image):
        """ Extract dense-sift features from one image
        """
        sift = cv2.xfeatures2d.SIFT_create()
        h, w = image.shape
        patch_size = (16, 16)
        step_size = 8

        coord_y = [y for y in range(step_size, h, step_size)]
        coord_x = [x for x in range(step_size, w, step_size)]
        self.num_feature_h = len(coord_y)
        self.num_feature_w = len(coord_x)

        kp = [cv2.KeyPoint(x,y, patch_size[0]) for y in coord_y
                                        for x in coord_x]
        _, dense_sift = sift.compute(image, kp)
        return dense_sift

    def feature_dense_sift(self, images):
        """ Extract Dense sift features from a batch of images
        """
        sift = cv2.xfeatures2d.SIFT_create()
        h, w = images.shape[1:]
        patch_size = (16, 16)
        step_size = 8

        coord_y = [y for y in range(step_size, h, step_size)]
        coord_x = [x for x in range(step_size, w, step_size)]
        self.num_feature_h = len(coord_y)
        self.num_feature_w = len(coord_x)

        kp = [cv2.KeyPoint(x,y, patch_size[0]) for y in coord_y
                                        for x in coord_x]
        features = np.zeros([images.shape[0], self.num_feature_h * self.num_feature_w, 128])
        for i, image in enumerate(tqdm(images)):
            _, dense_sift = sift.compute(image, kp)
            features[i] = dense_sift

        return features

    def train_clusters(self, features):
        """ use random subset of patch features to train KMeans as dictionary.
        """
        # sample from features
        num_samples, num_points = features.shape[:2]
        size_train_set = min( num_samples * num_points , self.MAX_NUM_SAMPLE_TRAIN_CLUSTER)
        indices = np.random.choice(num_samples * num_points , size_train_set)

        # todo: ref might consume large additional memory
        trainset = features.reshape(num_points * num_samples, -1)[indices, :]

        # train and predict
        # self.clusters.fit(trainset)
        print("Training MiniBatch KMeans")
        for i in tqdm(range(size_train_set // self.clusters_batch + 1)):
            start_idx = self.clusters_batch * i
            end_idx = min(self.clusters_batch * (i + 1), size_train_set)
            if end_idx - start_idx == 0:
                break
            batch = trainset[start_idx:end_idx, :]
            self.clusters.partial_fit(batch)

    def toBOF(self, features):
        """ convert lower feature to bags of feature
        """
        num_samples, num_points = features.shape[:2]
        vocab = self.clusters.predict(features.reshape(num_samples * num_points, -1))
        return vocab.reshape(num_samples, num_points, )

    def spatial_pyramid_matching_precomputed(self, x):
        """ precomputed function of svm, calculate the matching score of two vector
        """
        num_samples, num_features = x.shape
        x = x.reshape(-1, self.num_feature_h, self.num_feature_w)
        x_feature = np.zeros((num_samples, self.num_cells, self.M))

        # extract feature
        for level in range(0, self.L + 1):
            if level == 0:
                coef = 1 / (2 ** self.L)
            else:
                coef = 1 / (2 ** (self.L + 1 - level))

            num_level_cells = 4 ** level
            num_segments = 2 ** level
            cell_hs = np.linspace(0, self.num_feature_h, 2 ** level + 1, dtype=int)
            cell_ws = np.linspace(0, self.num_feature_w, 2 ** level + 1, dtype=int)

            # cells in each level
            for cell in range(num_level_cells):

                idx_y = cell // num_segments
                idx_x = cell % num_segments

                # histogram of BOF in one cell
                x_block_feature = x[:, cell_hs[idx_y] :cell_hs[idx_y + 1], cell_ws[idx_x]:cell_ws[idx_x + 1]]
                for s in range(num_samples):
                    counter = Counter(x_block_feature[s].flatten())
                    counts = np.array(counter.most_common(self.M))
                    level_feature = np.zeros((self.M, )) # 注意可能在cell内有未出现的 类
                    level_feature[counts[:,0]] = counts[:,1]
                    x_feature[s, count, :] = level_feature * coef
            
        x_feature = x_feature.reshape(num_samples, -1)
        xf1 = x_feature.reshape(num_samples, 1, -1).repeat(num_samples, axis = 1)
        xf2 = np.transpose(xf1, (1, 0, 2))

        return np.min([xf1, xf2], axis = 0).sum(axis = -1)

    def spatial_pyramid_matching_kernel(self, x, y):
        """ spatial pyramid matching kernel function of svm, 
            calculate the matching score of two vector
        """
        num_samples_x, num_features = x.shape
        num_samples_y, num_features = y.shape

        x = x.reshape(-1, self.num_feature_h, self.num_feature_w)
        y = y.reshape(-1, self.num_feature_h, self.num_feature_w)

        x_feature = np.zeros((num_samples_x, self.num_cells, self.M))
        y_feature = np.zeros((num_samples_y, self.num_cells, self.M))

        # extract feature
        count = 0
        for level in range(0, self.L + 1):
            if level == 0:
                coef = 1 / (2 ** self.L)
            else:
                coef = 1 / (2 ** (self.L + 1 - level))

            num_level_cells = 4 ** level
            num_segments = 2 ** level
            cell_hs = np.linspace(0, self.num_feature_h, 2 ** level + 1, dtype=int)
            cell_ws = np.linspace(0, self.num_feature_w, 2 ** level + 1, dtype=int)

            # cells in each level
            for cell in range(num_level_cells):

                idx_y = cell // num_segments
                idx_x = cell % num_segments

                # histogram of BOF in one cell
                x_block_feature = x[:, cell_hs[idx_y] :cell_hs[idx_y + 1], cell_ws[idx_x]:cell_ws[idx_x + 1]]
                for s in range(num_samples_x):
                    counter = Counter(x_block_feature[s].flatten())
                    counts = np.array(counter.most_common(self.M), dtype = int)
                    level_feature = np.zeros((self.M, )) # 注意可能在cell内有未出现的类
                    if counts.size != 0:
                        level_feature[counts[:,0]] = counts[:,1]
                    x_feature[s, count, :] = level_feature * coef

                y_block_feature = y[:, cell_hs[idx_y] :cell_hs[idx_y + 1], cell_ws[idx_x]:cell_ws[idx_x + 1]]
                for s in range(num_samples_y):
                    counter = Counter(y_block_feature[s].flatten())
                    counts = np.array(counter.most_common(self.M), dtype = int)
                    level_feature = np.zeros((self.M, ))
                    if counts.size != 0:
                        level_feature[counts[:, 0]] = counts[:, 1]
                    y_feature[s, count, :] = level_feature * coef
            
                count += 1
            
        x_feature = x_feature.reshape(num_samples_x, -1)
        y_feature = y_feature.reshape(num_samples_y, -1)

        # 此处直接用matrix计算的话两个repeat的内存占用会非常大，仅能对非常小的样本使用
        # 如果改成循环内存占用会小很多，但是牺牲时间效率
        if self.use_matrix_in_kernel:
            xf = x_feature.reshape(num_samples_x, 1, -1).repeat(num_samples_y, axis = 1)

            yf = y_feature.reshape(num_samples_y, 1, -1).repeat(num_samples_x, axis = 1)
            yf = np.transpose(yf, (1, 0, 2))
            t = np.min([xf, yf], axis = 0).sum(axis = -1)
        else:
            t = np.zeros((num_samples_x, num_samples_y))
            for i in range(num_samples_x):
                for j in range(num_samples_y):
                    a = x_feature[i,:]
                    b = y_feature[j,:]
                    t[i][j] = np.min([a, b], axis = 0).sum(axis = -1)

        return t

    def train_classifier(self, X_train, y_train):

        print(X_train.shape)
        self.classifier.fit(X_train, y_train)
        y_predict = self.classifier.predict(X_train)

        report = classification_report(y_train, y_predict)
        print("Classifier Training Report: \n {}".format(report))


    def predict_clss(self, X):
        return self.classifier.predict(X)
    
    def get_descriptors(self, images, labels, precompute_feature = None):

        if precompute_feature is None:
            print("Extracting Dense sift feature")
            low_features = self.feature_dense_sift(images)
            if self.save_sift_feature:
                self.save_feature(low_features)
        else:
            print("Using pre-computed feature")
            low_features = precompute_feature
            self._set_feature_hw(images)

        print("Traing vocabulary")
        if self.do_train_clusters:
            self.train_clusters(low_features)

        print("Extracting BOF feature")
        bof = self.toBOF(low_features)

        return bof
        # print("Training classifier")
        # if self.do_train_classifier:
        #     self.train_classifier(bof, labels)

        # self.save_model()

    def train(self, images, labels, precompute_feature = None):

        if precompute_feature is None:
            print("Extracting Dense sift feature")
            low_features = self.feature_dense_sift(images)
            if self.save_sift_feature:
                self.save_feature(low_features)
        else:
            print("Using pre-computed feature")
            low_features = precompute_feature
            self._set_feature_hw(images)

        print("Traing vocabulary")
        if self.do_train_clusters:
            self.train_clusters(low_features)

        print("Extracting BOF feature")
        bof = self.toBOF(low_features)

        print("Training classifier")
        if self.do_train_classifier:
            self.train_classifier(bof, labels)

        self.save_model()

    def test(self, images, labels, use_batch = True, batch_size = 100, precompute_feature = None):

        predict = self.inference(images, use_batch=use_batch, 
                    batch_size=batch_size, precompute_feature= precompute_feature)
        report = classification_report(labels, predict, output_dict = True)
        print(report)
        return report


    def inference(self, images, use_batch = True, batch_size = 100, precompute_feature = None):
        """ inference procedure, able to use batch to accelerate the progress of inference.
        """

        if precompute_feature is None:
            print("Extracting Dense sift feature")
            low_features = self.feature_dense_sift(images)
            if self.save_sift_feature:
                self.save_feature(low_features)
        else:
            print("Using pre-computed feature")
            low_features = precompute_feature
            self._set_feature_hw(images)

        print("Extracting BOF feature")
        bof = self.toBOF(low_features)

        num_samples = bof.shape[0]
        predict = np.zeros((num_samples, ), dtype = int)
        iternum = num_samples // batch_size  + 1

        print("Inference on classifier")
        for i in tqdm(range(iternum)):
            start_idx = batch_size * i 
            end_idx = min(batch_size * (i + 1), num_samples)
            if end_idx - start_idx == 0:
                break
            data = bof[start_idx: end_idx, ]
            out = self.predict_clss(data)
            predict[start_idx: end_idx, ] = out

        return predict
    

sigmap = 0.25
sigmac = 20
nSegments=500

def giveSSD(x1,x2):
    return np.sqrt(np.sum(np.square(x1-x2)))

def wp(pi,pj):
    return np.exp((-1/2*(sigmap**2))*np.square(giveSSD(pi,pj)))

def wc(ci,cj):
    return np.exp((-1/2*(sigmac**2))*np.square(giveSSD(ci,cj)))

def abstract(i):  
    image=i
    image_norm=img_as_float(image)
    # print(image_norm.shape)
    lab = rgb2lab(image_norm)
    # lab_norm=(lab + np.array([0, 128, 128])) / np.array([100, 255, 255])
    segments = slic(image_norm, n_segments = nSegments, compactness=20, sigma = 1, convert2lab=True)
    #segments = slic(image, n_segments=self.__n_segments,enforce_connectivity=True, compactness=30.0,convert2lab=False)

    # show the output of SLIC

    # plt.imshow(mark_boundaries(image, segments))
    # plt.savefig(segFileName)
    #plt.show()
    n_segments = segments.max() + 1

    # construct position matrix
    max_y, max_x = np.array(segments.shape) - 1
    x = np.linspace(0, max_x, image.shape[1]) / max_x
    y = np.linspace(0, max_y, image.shape[0]) / max_y
    position = np.dstack((np.meshgrid(x, y)))

    mean_colors = np.zeros((n_segments, 3))
    mean_position = np.zeros((n_segments, 2))
    clrs=np.zeros((n_segments, 3))
    for i in np.unique(segments):
        mask = segments == i
        mean_colors[i,:]=lab[mask,:].mean(axis=0)
        mean_position[i, :] = position[mask, :].mean(axis=0)

#     mean_position
#     mean_colors
    d_abstract={}

    d={}
    d1={}
    d_abst={}
    d_uniqueness={}
    r,c=segments.shape
    for i in range (r):
        for j in range (c):
            d[segments[i,j]]=[]
            d_uniqueness[segments[i,j]]=[]
            d1[segments[i,j]]=[]


    for i in range (r):
        for j in range (c):
            d[segments[i,j]].append([i,j])
            d1[segments[i,j]].append(image_norm[i,j])

    image_copy=np.copy(image_norm)
    
    for i in range (r):
        for j in range (c):
            image_copy[i,j]= np.median(d1[segments[i,j]],axis=0)
            d_uniqueness[segments[i,j]].append([image_copy[i,j]])
            
            
    output=(image_copy*255).astype('uint8')
    for i in d_uniqueness.keys():
        
        clrs[i]=d_uniqueness[i][0][0]

    
    return clrs,mean_position,segments,d,output,d_uniqueness

    

def uniquenessAssignment(c,p):
    U = np.empty(len(c))
    for i in range(len(c)):
        Zi=0
        pi = p[i]
        ci = c[i]
        tUniq = 0;
        
        for j in range(len(c)):
            #if i != j:
            pj = p[j] 
            cj=c[j]
            Zi+= wp(pi,pj)

        
        for j in range(len(c)):
            #if i != j:
            pj = p[j] 
            cj=c[j]
            tUniq += np.square(giveSSD(ci,cj))*(1/Zi)*wp(pi,pj)    

        U[i] = tUniq
    
    return U

def distributionAssignment(c,p):
    D = np.zeros(len(c))
    
    
    for i in range(len(c)):
        Zi=0    
        meani = 0
        tDist = 0
    
        ci = c[i]
        pi = p[i]
        
        for j in range(len(c)):
            cj = c[j] 
            Zi+= wc(ci,cj)
        
        for j in range(len(c)):
            cj = c[j] 
            pj = p[j]
            #if (i != j):
            meani += (1/Zi) * wc(ci,cj)*pj
        
        
        for j in range(len(c)):
            cj = c[j] 
            pj = p[j]
            tDist += np.square(giveSSD(pj,meani))*(1/Zi)*wc(ci,cj)
            
        D[i] = tDist
        
        
    return D

def saliency_Assignment(U_norm,dist_norm,colors,positions):
    k=3
    Si = np.ones(len(colors))
    S=np.ones(len(colors))
    Si = U_norm* np.exp(-k*dist_norm);
    for i in range(len(colors)):
        Zi=0
        pi = positions[i]
        ci = colors[i]
        tUniq = 0;

        for j in range(len(colors)):
            #if i != j:
            pj = positions[j] 
            cj = colors[j]
            Zi+=np.exp((-1/2*(1/10))*np.square(giveSSD(ci,cj)))*np.exp((-1/2*(1/30))*np.square(giveSSD(pi,pj)))


        for j in range(len(colors)):
            #if i != j:
            pj = positions[j] 
            cj = colors[j]
            tUniq +=(1/Zi)*np.exp((-1/2*(1/10))*np.square(giveSSD(ci,cj)))*np.exp((-1/2*(1/30))*np.square(giveSSD(pi,pj)))*Si[i] 

        S[i] = tUniq
    
    return S


# exception hander for singular value decomposition
class SVDError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


# (eigen_val, eigen_vec) = ncut( W, nbEigenValues ):
#
# This function performs the first step of normalized cut spectral clustering. The normalized LaPlacian
# is calculated on the similarity matrix W, and top nbEigenValues eigenvectors are calculated. The number of 
# eigenvectors corresponds to the maximum number of classes (K) that will be produced by the clustering algorithm. 
#
#    W:             symmetric #feature x #feature sparse matrix representing the similarity between voxels, 
#                   traditionally this matrix should be positive semidefinite, but a trick is implemented to 
#                   allow negative matrix entries
#    nvEigenValues: number of eigenvectors that should be calculated, this determines the maximum number of 
#                   clusters (K) that can be derived from the result
#    eigen_val:     (output) eigenvalues from the eigen decomposition of the LaPlacian of W
#    eigen_vec:     (output) eigenvectors from the eign decomposition of the LaPlacian of W
#
def ncut(W, nbEigenValues):
    # parameters
    offset = .5
    maxiterations = 100
    eigsErrorTolerence = 1e-6
    truncMin = 1e-6
    eps = 2.2204e-16

    m = shape(W)[1]

    # make sure that W is symmetric, this is a computationally expensive operation, only use for debugging
    #if (W-W.transpose()).sum() != 0:
    #	print "W should be symmetric!"
    #	exit(0)

    # degrees and regularization
    # S Yu Understanding Popout through Repulsion CVPR 2001
    # Allows negative values as well as improves invertability
    # of d for small numbers
    # i bet that this is what improves the stability of the eigen
    d = abs(W).sum(0)
    dr = 0.5 * (d - W.sum(0))
    d = d + offset * 2
    dr = dr + offset

    # calculation of the normalized LaPlacian
    W = W + spdiags(dr, [0], m, m, "csc")
    Dinvsqrt = spdiags((1.0 / sqrt(d + eps)), [0], m, m, "csc")
    P = Dinvsqrt * (W * Dinvsqrt);

    # perform the eigen decomposition
    #eigen_val, eigen_vec = eigsh(P, nbEigenValues, maxiter=maxiterations, tol=eigsErrorTolerence, which='LA')
    eigen_val, eigen_vec = eigsh(P, nbEigenValues, tol=eigsErrorTolerence, which='LA')

    # sort the eigen_vals so that the first
    # is the largest
    i = argsort(-eigen_val)
    eigen_val = eigen_val[i]
    eigen_vec = eigen_vec[:, i]

    # normalize the returned eigenvectors
    eigen_vec = Dinvsqrt * matrix(eigen_vec)
    norm_ones = norm(ones((m, 1)))
    for i in range(0, shape(eigen_vec)[1]):
        eigen_vec[:, i] = (eigen_vec[:, i] / norm(eigen_vec[:, i])) * norm_ones
        if eigen_vec[0, i] != 0:
            eigen_vec[:, i] = -1 * eigen_vec[:, i] * sign(eigen_vec[0, i])

    return (eigen_val, eigen_vec)


# eigenvec_discrete=discretisation( eigen_vec ):
#
# This function performs the second step of normalized cut clustering which assigns features to clusters 
# based on the eigen vectors from the LaPlacian of a similarity matrix. There are a few different ways to
# perform this task. Shi and Malik (2000) iteratively bisect the features based on the positive and 
# negative loadings of the eigenvectors. Ng, Jordan and Weiss (2001) proposed to perform K-means clustering
# on the rows of the eigenvectors. The method implemented here was proposed by Yu and Shi (2003) and it finds
# a discrete solution by iteratively rotating a binaised set of vectors until they are maximally similar to
# the the eigenvectors (for more information, the full citation is at the top of this file). An advantage
# of this method over K-means is that it is _more_ deterministic, i.e. you should get very similar results
# every time you run the algorithm on the same data.
#
# The number of clusters that the features are clustered into is determined by the number of eignevectors 
# (number of columns) in the input array eigen_vec. A caveat of this method, is that number of resulting
# clusters is bound by the number of eignevectors, but it may contain less.
#
#    eigen_vec:          Eigenvectors of the normalized LaPlacian calculated from the similarity matrix 
#                        for the corresponding clustering problem
#    eigen_vec_discrete: (output) discretised eigenvectors, i.e. vectors of 0 and 1 which indicate 
#                        wether or not a feature belongs to the cluster defined by the eigen vector.
#                        I.E. a one in the 10th row of the 4th eigenvector (column) means that feature
#                        10 belongs to cluster #4.
# 
def discretisation(eigen_vec):
    eps = 2.2204e-16

    # normalize the eigenvectors
    [n, k] = shape(eigen_vec)
    vm = kron(ones((1, k)), sqrt(multiply(eigen_vec, eigen_vec).sum(1)))
    eigen_vec = divide(eigen_vec, vm)

    svd_restarts = 0
    exitLoop = 0

    ### if there is an exception we try to randomize and rerun SVD again
    ### do this 30 times
    while (svd_restarts < 30) and (exitLoop == 0):

        # initialize algorithm with a random ordering of eigenvectors
        c = zeros((n, 1))
        R = matrix(zeros((k, k)))
        R[:, 0] = eigen_vec[int(rand(1) * (n)), :].transpose()

        for j in range(1, k):
            c = c + abs(eigen_vec * R[:, j - 1])
            R[:, j] = eigen_vec[c.argmin(), :].transpose()

        lastObjectiveValue = 0
        nbIterationsDiscretisation = 0
        nbIterationsDiscretisationMax = 20

        # iteratively rotate the discretised eigenvectors until they
        # are maximally similar to the input eignevectors, this
        # converges when the differences between the current solution
        # and the previous solution differs by less than eps or we
        # we have reached the maximum number of itarations
        while exitLoop == 0:
            nbIterationsDiscretisation = nbIterationsDiscretisation + 1

            # rotate the original eigen_vectors
            tDiscrete = eigen_vec * R

            # discretise the result by setting the max of each row=1 and
            # other values to 0
            j = reshape(asarray(tDiscrete.argmax(1)), n)
            eigenvec_discrete = csc_matrix((ones(len(j)), (range(0, n), array(j))), shape=(n, k))

            # calculate a rotation to bring the discrete eigenvectors cluster to the
            # original eigenvectors
            tSVD = eigenvec_discrete.transpose() * eigen_vec
            # catch a SVD convergence error and restart
            try:
                U, S, Vh = svd(tSVD)
                svd_restarts += 1
            except LinAlgError:
                # catch exception and go back to the beginning of the loop
                print >> sys.stderr, "SVD did not converge, randomizing and trying again"
                break

            # test for convergence
            NcutValue = 2 * (n - S.sum())
            if ((abs(NcutValue - lastObjectiveValue) < eps ) or
                    ( nbIterationsDiscretisation > nbIterationsDiscretisationMax )):
                exitLoop = 1
            else:
                # otherwise calculate rotation and continue
                lastObjectiveValue = NcutValue
                R = matrix(Vh).transpose() * matrix(U).transpose()

    if exitLoop == 0:
        raise SVDError("SVD did not converge after 30 retries")
    else:
        return (eigenvec_discrete)
    

def round_box(box):
    xmin, ymin, xmax, ymax, score = box
    return (round(float(xmin)), round(float(ymin)), round(float(xmax)), round(float(ymax)), score)

def extract_boxes(run_objectness, M, params, data_dir, imgs):
    box_data = []
    box_coordinates = []
    for img in imgs:
        img_id = data_dir + img
        img_example = cv2.imread(img_id)[:, :, ::-1]
        boxes = run_objectness(img_example, M, params)
        print(boxes)
        box_coordinates = box_coordinates + boxes.tolist()
        box_data += boxes_data_from_img(boxes, img_id)
    return box_data, box_coordinates

def boxes_data_from_img(boxes, img_id):
    box_data = []
    img_data = cv2.imread(img_id, cv2.COLOR_BGR2RGB)
    for box in boxes:
        xmin, ymin, xmax, ymax, score = round_box(box)
        box_data.append(img_data[ymin:ymax, xmin:xmax, :])
    return box_data

"""
    function caculate is the discriminative clustering term 
    CM: central projection matrix
    I: ones matrix
    X: similarity natrix (n x d)
"""

def discriminative_optimial(central_matrix, X, nbox, I, k):
    X = np.vstack(X)
    print("X shape {}".format(X.shape))
    _in_ = X.T @ central_matrix @ X + nbox * k * I
    I1 = np.identity(X.shape[0])
    _in1_ = I1 - X @ inv(np.matrix(_in_)) @ X.T
    A = (1/nbox) * central_matrix @ _in1_ @ central_matrix 
    return A

def normalize_laplacian(W):
    # parameters
    offset = .5
    maxiterations = 100
    eigsErrorTolerence = 1e-6
    truncMin = 1e-6
    eps = 2.2204e-16

    m = shape(W)[1]

    # make sure that W is symmetric, this is a computationally expensive operation, only use for debugging
    #if (W-W.transpose()).sum() != 0:
    #	print "W should be symmetric!"
    #	exit(0)

    # degrees and regularization
    # S Yu Understanding Popout through Repulsion CVPR 2001
    # Allows negative values as well as improves invertability
    # of d for small numbers
    # i bet that this is what improves the stability of the eigen
    d = abs(W).sum(0)
    dr = 0.5 * (d - W.sum(0))
    d = d + offset * 2
    dr = dr + offset

    # calculation of the normalized LaPlacian
    W = W + spdiags(dr, [0], m, m, "csc")
    Dinvsqrt = spdiags((1.0 / sqrt(d + eps)), [0], m, m, "csc")
    P = Dinvsqrt * (W * Dinvsqrt);
    return P


def to_rgb(x):
    x_rgb = np.zeros((x.shape[0], 28, 28, 3))
    for i in range(3):
        x_rgb[..., i] = x[..., 0]
    return x_rgb

def load_dataset(img_dir, annot_file, num_per_class=-1):
    data = []
    labels = []
    k = 0
    for filename in os.listdir(img_dir):
        if filename.endswith(".asm") or filename.endswith(".jpg") and k < 2: 
            label_filename = annot_file + filename.split('.')[0] + '.xml'
            tree = ET.parse(label_filename)
            root = tree.getroot()
            objects= tree.findall('.//object')
            for obj in objects:
                x = int(obj.find('.//xmin').text)
                y = int(obj.find('.//ymin').text)
                x_max = int(obj.find('.//xmax').text)
                y_max = int(obj.find('.//ymax').text)
                data.append(cv2.imread(img_dir + filename, cv2.COLOR_BGR2RGB)[y:y_max, x:x_max, :])
                # print(cv2.imread(img_dir + filename, cv2.COLOR_RGB2BGR)[y:y_max, x:x_max, :].shape)
            labels.append(1)
            k = k + 1
            continue
        else:
            continue
    return data, labels


# compute dense SIFT 
def computeSIFT(data):
    x = []
    # print(data)
    for i in range(0, len(data)):
        sift = cv2.xfeatures2d.SIFT_create()
        img = data[i]
        # print(img)
        step_size = 15
        kp = [cv2.KeyPoint(x, y, step_size) for x in range(0, img.shape[0], step_size) for y in range(0, img.shape[1], step_size)]
        dense_feat = sift.compute(img, kp)
        x.append(dense_feat[1])
        
    return x


def extract_denseSIFT(img):
    DSIFT_STEP_SIZE = 2
    sift = cv2.xfeatures2d.SIFT_create()
    disft_step_size = DSIFT_STEP_SIZE
    keypoints = [cv2.KeyPoint(x, y, disft_step_size)
            for y in range(0, img.shape[0], disft_step_size)
                for x in range(0, img.shape[1], disft_step_size)]

    descriptors = sift.compute(img, keypoints)[1]
    
    #keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


# form histogram with Spatial Pyramid Matching upto level L with codebook kmeans and k codewords
def getImageFeaturesSPM(L, img, kmeans, k):
    W = img.shape[1]
    H = img.shape[0]   
    h = []
    for l in range(L+1):
        w_step = math.floor(W/(2**l))
        h_step = math.floor(H/(2**l))
        x, y = 0, 0
        for i in range(1,2**l + 1):
            x = 0
            for j in range(1, 2**l + 1):                
                desc = extract_denseSIFT(img[y:y+h_step, x:x+w_step])                
                #print("type:",desc is None, "x:",x,"y:",y, "desc_size:",desc is None)
                predict = kmeans.predict(desc)
                histo = np.bincount(predict, minlength=k).reshape(1,-1).ravel()
                weight = 2**(l-L)
                h.append(weight*histo)
                x = x + w_step
            y = y + h_step
            
    hist = np.array(h).ravel()
    # normalize hist
    dev = np.std(hist)
    hist -= np.mean(hist)
    hist /= dev
    return hist


# get histogram representation for training/testing data
def getHistogramSPM(L, data, kmeans, k):    
    x = []
    for i in range(len(data)):        
        hist = getImageFeaturesSPM(L, data[i], kmeans, k)        
        x.append(hist)
    return np.array(x)


# build BoW presentation from SIFT of training images 
def clusterFeatures(all_train_desc, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(all_train_desc)
    return kmeans


def saliency_map_from_set(imgs):
    maps = []
    for image in imgs:
        # image=io.imread(ipFileName)
        # print("Image being read.")
        # image_uniqueness=np.zeros((image.shape[0],image.shape[1],3))
        # image_distribution=np.zeros((image.shape[0],image.shape[1],3))
        # image_saliency=np.zeros((image.shape[0],image.shape[1],3))
        # image=io.imread(image)
        img_size = image.shape[0] * image.shape[1]

        colors=[]
        positions=[]

        # print("Started Abstraction.")
        colors,positions,seg,d_p,o,d_u = abstract(image)
        # print("Abstraction successful.")

        # print("Started Uniqueness Assignment.")
        Uniqueness=uniquenessAssignment(colors,positions)
        U_norm=Uniqueness/max(Uniqueness)
        # print("Uniqueness Assignment successful.")

        # print("Starting Distribution Assignment.")
        dist=distributionAssignment(colors,positions)
        dist_norm=dist/max(dist)
        # print("Distribution Assignment successful.")

        # print("Starting Saliency Assignment.")
        sal=saliency_Assignment(U_norm,dist_norm,colors,positions)    
        # print("Saliency Assignment successful.")

        # for i in range(len(d_p)):
        #     for k in range(len(d_p[i])):
                
        #         row=d_p[i][k][0]
        #         col=d_p[i][k][1]
        #         image_uniqueness[row,col]=Uniqueness[i]
        #         image_distribution[row,col]=dist_norm[i]
        #         image_saliency[row,col]=sal[i]
        maps.append(sal / img_size)
    return maps

def compute_purity(C_computed,C_grndtruth,R):
    """
    Clustering accuracy can be defined with the purity measure, defined here:
      Yang-Hao-Dikmen-Chen-Oja, Clustering by nonnegative matrix factorization
      using graph random walk, 2012.

    Usages:
      accuracy = compute_clustering_accuracy(C_computed,C_grndtruth,R)

    Notations:
      n = nb_data

    Input variables:
      C_computed = Computed clusters. Size = n x 1. Values in [0,1,...,R-1].
      C_grndtruth = Ground truth clusters. Size = n x 1. Values in [0,1,...,R-1].
      R = Number of clusters.

    Output variables:
      accuracy = Clustering accuracy of computed clusters.
    """

    N = C_grndtruth.size
    nb_of_dominant_points_in_class = np.zeros((R, 1))
    w = defaultdict(list)
    z = defaultdict(list)       
    for k in range(R):
        for i in range(N):
            if C_computed[i]==k:
                w[k].append(C_grndtruth[i])
        if len(w[k])>0:
            val,nb = stats.mode(w[k])
            z[k] = int(nb.squeeze()) 
        else:
            z[k] = 0
    sum_dominant = 0
    for k in range(R):
        sum_dominant = sum_dominant + z[k]
    purity = float(sum_dominant) / float(N)* 100.0
    return purity

def compute_ncut(W, Cgt, R):
    """
    Graph spectral clustering technique NCut:
      Yu-Shi, Multiclass spectral clustering, 2003
      Code available here: http://www.cis.upenn.edu/~jshi/software

    Usages:
      C,acc = compute_ncut(W,Cgt,R)

    Notations:
      n = nb_data

    Input variables:
      W = Adjacency matrix. Size = n x n.
      R = Number of clusters.
      Cgt = Ground truth clusters. Size = n x 1. Values in [0,1,...,R-1].

    Output variables:
      C = NCut solution. Size = n x 1. Values in [0,1,...,R-1].
      acc = Accuracy of NCut solution.
    """

    # Apply ncut
    eigen_val, eigen_vec = ncut( W, R )
    
    # Discretize to get cluster id
    eigenvec_discrete = discretisation( eigen_vec )
    res = eigenvec_discrete.dot(np.arange(1, R + 1)) 
    C = np.array(res-1,dtype=np.int64)
    
    # Compute accuracy
    computed_solution = C
    ground_truth = Cgt
    acc = compute_purity( computed_solution,ground_truth, R)

    return C, acc