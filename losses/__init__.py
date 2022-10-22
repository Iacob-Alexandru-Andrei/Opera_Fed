import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# define custom losses
class Contrastive_multiview_loss(nn.Module):

    def __init__(self,loss_func):
        """
        Simple implemntation of Full Graph Contrastive Multiview Coding (Tian et.al. 2020)
        Attribute:
        loss_func (torch.nn.Module): the contrastive loss function
        """
        super(Contrastive_multiview_loss, self).__init__()
        self.loss_func = loss_func

    def forward(self,*vecs):
        loss = sum([self.loss_func(vecs[i],vecs[j]) for i in range(0,len(vecs)-1) for j in range(i+1,len(vecs))])
        return loss
    
    
class NT_Xent(nn.Module):
    """
    NT_Xent for SimCLR. Work created by Spijkervet
    """

    def __init__(self, batch_size, temperature, world_size=1,device='cuda'):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        assert z_i.shape == z_j.shape
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
#         if self.world_size > 1:
#             z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)
#         print(z_i.shape,z.shape,sim.shape,sim_i_j.shape,sim_j_i.shape)
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        labels = torch.zeros(N).long().to(logits.device)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
##########################################################################################################################
# https://github.com/paulxiong/SimCLR-4/blob/master/modules/nt_xent.py
class NT_Xent_2(nn.Module):

    def __init__(self, batch_size, temperature, mask, device):
        super(NT_Xent_2, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = mask
        self.device = device

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """

        p1 = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(p1.unsqueeze(1), p1.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(self.batch_size * 2, 1)
        negative_samples = sim[self.mask].reshape(self.batch_size * 2, -1)

        labels = torch.zeros(self.batch_size * 2).to(self.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size
        return loss       
    
##########################################################################################################################
# class Contrastive_multiview_loss(nn.Module):

#     def __init__(self,loss_func):
#         super(Contrastive_multiview_loss, self).__init__()
#         self.loss_func = loss_func

#     def forward(self,*vecs):
#         loss = sum([self.loss_func(vecs[i],vecs[j]) for i in range(0,len(vecs)-1) for j in range(i+1,len(vecs))])
# #         loss = torch.tensor(0)
# #         for i in range(0,len(vecs)-1):
# #             for j in range(i+1,len(vecs)):
# #                 cur = self.loss_func(vecs[i],vecs[j])
# #                 loss = loss.add(cur)
#         return loss

########################################################################################################################

class SupConLoss(nn.Module):
    """


    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR

    origin: https://github.com/HobbitLong/SupContrast.git
    """

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, stack = True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.stack = stack

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # features = F.normalize(features,dim=1)

        if self.stack == True:
            features = features.unsqueeze(1)
            features = torch.cat((features,features),dim=1)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
    
    
    
############################ sim - siam ################################    

def negative_cosine_similarity(
    p: torch.Tensor,
    z: torch.Tensor
) -> torch.Tensor:
    """ D(p, z) = -(p*z).sum(dim=1).mean() """
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    
#########################################################################
class NT_Xent_3(nn.Module):

    def __init__(self,batch_size,num_repre=2,temperature=0.1):
        """
        NT_Xent, support for num_repre (number of representation) >= 2
        Attributes:
        batch_size (int): batch size
        num_repre (int): number of representation
        temperature (float): smoothness
        """
        super(NT_Xent_3, self).__init__()
        if num_repre < 2:
            raise ValueError('num_repre must be >=2')
        self.batch_size = batch_size
        self.num_repre = num_repre
        self.mtx_size = batch_size*num_repre
        self.positive_mask = self.generate_mask(1)
        self.negative_mask = self.generate_mask(0)
        self.temperature = temperature
        self.xent = nn.CrossEntropyLoss(reduction="sum")
        self.cossim = nn.CosineSimilarity(dim=2)


    def forward(self,*tensors):
        """
        Argument
        tensors (torch.Tensor):
        """
        assert tensors[0].shape[0] == self.batch_size,f'batch size not matching, expecting size: {self.batch_size}, get batch size {tensors[0].shape[0]}'
        assert len(tensors) == self.num_repre,f'number of representation not matching, expecting size: {self.num_repre}, get {len(tensors)} '
        z = torch.cat(tensors, dim=0)
        s = self.cossim(z.unsqueeze(1),z.unsqueeze(0))/self.temperature
        positive_samples = s[self.positive_mask]
        positive_samples = positive_samples.reshape(self.mtx_size,-1)
        positive_samples = torch.cat(torch.split(positive_samples,1,1))
        # print(f'positive shape {positive_samples.shape}')
        negative_samples = s[self.negative_mask]
        negative_samples = negative_samples.reshape(self.mtx_size,-1)
        negative_samples = negative_samples.repeat(self.num_repre-1,1)
        # print(f'negative shape {negative_samples.shape}')
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        labels = torch.zeros(logits.shape[0]).long().to(logits.device)
        # print(f'logit shape {logits.shape}')
        loss = self.xent(logits, labels)
        loss /= self.mtx_size
        return loss

    def generate_mask(self,mask_type=0):
        if mask_type != 1 and mask_type != 0:
            raise ValueError('Mask_type must be equal {0,1}')
        if mask_type:
            mask = torch.zeros((self.mtx_size, self.mtx_size), dtype=bool)
        else:
            mask = torch.ones((self.mtx_size, self.mtx_size), dtype=bool)
            mask = mask.fill_diagonal_(0)
        for i in range(self.mtx_size):
            for j in range(i,self.mtx_size,self.batch_size):
                if i != j:
                    mask[i,j] = mask_type
                    mask[j,i] = mask_type
        return mask
#################################################################################################################
    
#!wget https://raw.githubusercontent.com/wangz10/contrastive_loss/master/losses.py
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def pdist_euclidean(A):
    # Euclidean pdist
    # https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    r = tf.reduce_sum(A*A, 1)

    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
    return tf.sqrt(D)


def square_to_vec(D):
    '''Convert a squared form pdist matrix to vector form.
    '''
    n = D.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    d_vec = tf.gather_nd(D, list(zip(triu_idx[0], triu_idx[1])))
    return d_vec


def get_contrast_batch_labels(y):
    '''
    Make contrast labels by taking all the pairwise in y
    y: tensor with shape: (batch_size, )
    returns:   
        tensor with shape: (batch_size * (batch_size-1) // 2, )
    '''
    y_col_vec = tf.reshape(tf.cast(y, tf.float32), [-1, 1])
    D_y = pdist_euclidean(y_col_vec)
    d_y = square_to_vec(D_y)
    y_contrasts = tf.cast(d_y == 0, tf.int32)
    return y_contrasts


def get_contrast_batch_labels_regression(y):
    '''
    Make contrast labels for regression by taking all the pairwise in y
    y: tensor with shape: (batch_size, )
    returns:   
        tensor with shape: (batch_size * (batch_size-1) // 2, )
    '''
    raise NotImplementedError


def max_margin_contrastive_loss(z, y, margin=1.0, metric='euclidean'):
    '''
    Wrapper for the maximum margin contrastive loss (Hadsell et al. 2006)
    `tfa.losses.contrastive_loss`
    Args:
        z: hidden vector of shape [bsz, n_features].
        y: ground truth of shape [bsz].
        metric: one of ('euclidean', 'cosine')
    '''
    # compute pair-wise distance matrix
    if metric == 'euclidean':
        D = pdist_euclidean(z)
    elif metric == 'cosine':
        D = 1 - tf.matmul(z, z, transpose_a=False, transpose_b=True)
    # convert squareform matrix to vector form
    d_vec = square_to_vec(D)
    # make contrastive labels
    y_contrasts = get_contrast_batch_labels(y)
    loss = tfa.losses.contrastive_loss(y_contrasts, d_vec, margin=margin)
    # exploding/varnishing gradients on large batch?
    return tf.reduce_mean(loss)


def multiclass_npairs_loss(z, y):
    '''
    Wrapper for the multiclass N-pair loss (Sohn 2016)
    `tfa.losses.npairs_loss`
    Args:
        z: hidden vector of shape [bsz, n_features].
        y: ground truth of shape [bsz].
    '''
    # cosine similarity matrix
    S = tf.matmul(z, z, transpose_a=False, transpose_b=True)
    loss = tfa.losses.npairs_loss(y, S)
    return loss


def triplet_loss(z, y, margin=1.0, kind='hard'):
    '''
    Wrapper for the triplet losses 
    `tfa.losses.triplet_hard_loss` and `tfa.losses.triplet_semihard_loss`
    Args:
        z: hidden vector of shape [bsz, n_features], assumes it is l2-normalized.
        y: ground truth of shape [bsz].    
    '''
    if kind == 'hard':
        loss = tfa.losses.triplet_hard_loss(y, z, margin=margin, soft=False)
    elif kind == 'soft':
        loss = tfa.losses.triplet_hard_loss(y, z, margin=margin, soft=True)
    elif kind == 'semihard':
        loss = tfa.losses.triplet_semihard_loss(y, z, margin=margin)
    return loss


def supervised_nt_xent_loss(z, y, temperature=0.5, base_temperature=0.07):
    '''
    Supervised normalized temperature-scaled cross entropy loss. 
    A variant of Multi-class N-pair Loss from (Sohn 2016)
    Later used in SimCLR (Chen et al. 2020, Khosla et al. 2020).
    Implementation modified from: 
        - https://github.com/google-research/simclr/blob/master/objective.py
        - https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Args:
        z: hidden vector of shape [bsz, n_features].
        y: ground truth of shape [bsz].
    '''
    batch_size = tf.shape(z)[0]
    contrast_count = 1
    anchor_count = contrast_count
    y = tf.expand_dims(y, -1)

    # mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
    #     has the same class as sample i. Can be asymmetric.
    mask = tf.cast(tf.equal(y, tf.transpose(y)), tf.float32)
    anchor_dot_contrast = tf.divide(
        tf.matmul(z, tf.transpose(z)),
        temperature
    )
    # # for numerical stability
    logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
    logits = anchor_dot_contrast - logits_max
    # # tile mask
    logits_mask = tf.ones_like(mask) - tf.eye(batch_size)
    mask = mask * logits_mask
    # compute log_prob
    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - \
        tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True))

    # compute mean of log-likelihood over positive
    # this may introduce NaNs due to zero division,
    # when a class only has one example in the batch
    mask_sum = tf.reduce_sum(mask, axis=1)
    mean_log_prob_pos = tf.reduce_sum(
        mask * log_prob, axis=1)[mask_sum > 0] / mask_sum[mask_sum > 0]

    # loss
    loss = -(temperature / base_temperature) * mean_log_prob_pos
    # loss = tf.reduce_mean(tf.reshape(loss, [anchor_count, batch_size]))
    loss = tf.reduce_mean(loss)
    return loss

#############################################################################################################################

def get_NT_Xent_loss_gradients(model, samples_transform_1, samples_transform_2, normalize=True, temperature=1.0, weights=1.0):
    """
    A wrapper function for the NT_Xent_loss function which facilitates back propagation
    Parameters:
        model
            the deep learning model for feature learning 
        samples_transform_1
            inputs samples subject to transformation 1
        
        samples_transform_2
            inputs samples subject to transformation 2
        normalize = True
            normalise the activations if true
        temperature = 1.0
            hyperparameter, the scaling factor of the logits
            (see NT_Xent_loss)
        
        weights = 1.0
            weights of different samples
            (see NT_Xent_loss)
    Return:
        loss
            the value of the NT_Xent_loss
        gradients
            the gradients for backpropagation
    """
    with tf.GradientTape() as tape:
        hidden_features_transform_1 = model(samples_transform_1)
        hidden_features_transform_2 = model(samples_transform_2)
        loss = NT_Xent_loss(hidden_features_transform_1, hidden_features_transform_2, normalize=normalize, temperature=temperature, weights=weights)

    gradients = tape.gradient(loss, model.trainable_variables)
    return loss, gradients


@tf.function
def NT_Xent_loss(hidden_features_transform_1, hidden_features_transform_2, normalize=True, temperature=1.0, weights=1.0):
    """
    The normalised temperature-scaled cross entropy loss function of SimCLR Contrastive training
    Reference: Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. arXiv preprint arXiv:2002.05709.
    https://github.com/google-research/simclr/blob/master/objective.py
    Parameters:
        hidden_features_transform_1
            the features (activations) extracted from the inputs after applying transformation 1
            e.g. model(transform_1(X))
        
        hidden_features_transform_2
            the features (activations) extracted from the inputs after applying transformation 2
            e.g. model(transform_2(X))
        normalize = True
            normalise the activations if true
        temperature
            hyperparameter, the scaling factor of the logits
        
        weights
            weights of different samples
    Return:
        loss
            the value of the NT_Xent_loss
    """
    LARGE_NUM = 1e9
    entropy_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    batch_size = tf.shape(hidden_features_transform_1)[0]
    
    h1 = hidden_features_transform_1
    h2 = hidden_features_transform_2
    if normalize:
        h1 = tf.math.l2_normalize(h1, axis=1)
        h2 = tf.math.l2_normalize(h2, axis=1)

    labels = tf.range(batch_size)
    masks = tf.one_hot(tf.range(batch_size), batch_size)
    
    logits_aa = tf.matmul(h1, h1, transpose_b=True) / temperature
    # Suppresses the logit of the repeated sample, which is in the diagonal of logit_aa
    # i.e. the product of h1[x] . h1[x]
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(h2, h2, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(h1, h2, transpose_b=True) / temperature
    logits_ba = tf.matmul(h2, h1, transpose_b=True) / temperature

    
    loss_a = entropy_function(labels, tf.concat([logits_ab, logits_aa], 1), sample_weight=weights)
    loss_b = entropy_function(labels, tf.concat([logits_ba, logits_bb], 1), sample_weight=weights)
    loss = loss_a + loss_b

    return loss

#############################################################################################
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        return losses.NTXentLoss(temperature=0.05)(logits, torch.squeeze(labels))
################################################################################
# class SupervisedContrastiveLoss(losses.Loss):
#     def __init__(self, temperature=1, name=None):
#         super(SupervisedContrastiveLoss, self).__init__(name=name)
#         self.temperature = temperature

#     def __call__(self, labels, feature_vectors, sample_weight=None):
#         # Normalize feature vectors
#         feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
#         # Compute logits
#         logits = tf.divide(
#             tf.matmul(
#                 feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
#             ),
#             temperature,
#         )
#         return tfa.losses.npairs_loss(tf.squeeze(labels), logits)
################################################################################################    
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    
    
    def __init__(self, temperature=0.05, contrast_mode='all',
                 base_temperature=0.05):
        super(SupConLoss, self).__init__()
        
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
                 
#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))        
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
     
def mask_correlated_samples(batch_size):
    mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask    



# define loss function
def loss_func_moco(q,k,queue,t=0.05):        
    # t: temperature
    N = q.shape[0] # batch_size
    C = q.shape[1] # channel
    # bmm: batch matrix multiplication
    pos = torch.exp(torch.div(torch.bmm(q.view(N,1,C), k.view(N,C,1)).view(N,1),t))
    neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N,C),torch.t(queue)),t)),dim=1)
    # denominator is sum over pos and neg
    denominator = pos + neg

    return torch.mean(-torch.log(torch.div(pos,denominator)))
#####################################################################################################################

from torch.optim.optimizer import Optimizer, required
import re

EETA_DEFAULT = 0.001


class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0.9,
        use_nesterov=False,
        weight_decay=0.0,
        exclude_from_weight_decay=None,
        exclude_from_layer_adaptation=None,
        classic_momentum=True,
        eeta=EETA_DEFAULT,
    ):
        """Constructs a LARSOptimizer.
        Args:
        lr: A `float` for learning rate.
        momentum: A `float` for momentum.
        use_nesterov: A 'Boolean' for whether to use nesterov momentum.
        weight_decay: A `float` for weight decay.
        exclude_from_weight_decay: A list of `string` for variable screening, if
            any of the string appears in a variable's name, the variable will be
            excluded for computing weight decay. For example, one could specify
            the list like ['batch_normalization', 'bias'] to exclude BN and bias
            from weight decay.
        exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
            for layer adaptation. If it is None, it will be defaulted the same as
            exclude_from_weight_decay.
        classic_momentum: A `boolean` for whether to use classic (or popular)
            momentum. The learning rate is applied during momeuntum update in
            classic momentum, but after momentum for popular momentum.
        eeta: A `float` for scaling of learning rate when computing trust ratio.
        name: The name for the scope.
        """

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            exclude_from_weight_decay=exclude_from_weight_decay,
            exclude_from_layer_adaptation=exclude_from_layer_adaptation,
            classic_momentum=classic_momentum,
            eeta=eeta,
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eeta = eeta
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
        # arg is None.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def step(self, epoch=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eeta = group["eeta"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                # TODO: get param names
                # if self._use_weight_decay(param_name):
                grad += self.weight_decay * param

                if self.classic_momentum:
                    trust_ratio = 1.0

                    # TODO: get param names
                    # if self._do_layer_adaptation(param_name):
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    device = g_norm.get_device()
                    trust_ratio = torch.where(
                        w_norm.gt(0),
                        torch.where(
                            g_norm.gt(0),
                            (self.eeta * w_norm / g_norm),
                            torch.Tensor([1.0]).to(device),
                        ),
                        torch.Tensor([1.0]).to(device),
                    ).item()

                    scaled_lr = lr * trust_ratio
                    if "momentum_buffer" not in param_state:
                        next_v = param_state["momentum_buffer"] = torch.zeros_like(
                            p.data
                        )
                    else:
                        next_v = param_state["momentum_buffer"]

                    next_v.mul_(momentum).add_(scaled_lr, grad)
                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v

                    p.data.add_(-update)
                else:
                    raise NotImplementedError

        return loss

    def _use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True


###########################################################################################################
class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    
    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]

        output:
            - loss: loss computed according to SimCLR 
        """

        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        return loss