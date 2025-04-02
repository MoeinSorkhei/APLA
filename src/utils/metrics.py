import torch
import numpy as np
from sklearn import metrics
from easydict import EasyDict as edict

from ._utils import *


def get_activation(is_multiclass):
    if is_multiclass:
        act = nn.Softmax(dim=1)
    else:
        act = nn.Sigmoid()
    return act


def mean_roc_auc(truths, predictions):
    """
    Calculating mean ROC-AUC:
        Asuuming that the last dimension represent the classes
    """
    _truths = np.array(deepcopy(truths))
    _predictions = np.array(deepcopy(predictions))  
    n_classes = _predictions.shape[-1]
    avg_roc_auc = 0 
    for class_num in range(n_classes):
        auc = 0.5
        tar = (_truths[:,class_num] + _truths[:,class_num]**2 ) / 2
        if tar.sum() > 0:
            auc = metrics.roc_auc_score(tar, _predictions[:,class_num], 
                                        average='macro', 
                                        sample_weight=_truths[:, class_num] ** 2 + 1e-06, 
                                        multi_class = 'ovo')            
        avg_roc_auc += auc 
    return avg_roc_auc / n_classes


class ClassificationMetrics:
    def __init__(self, n_classes, mode="", raw=True) -> None:
        self.n_classes = n_classes
        self.prefix = mode + "_" if mode else ""
        self.raw = raw  # meaning that the metric gets the raw logits, instead of actual probabilities
        self.act_fn = torch.nn.Softmax(dim=1) if self.raw else lambda x: x  # no activation needed if values are already probabilities
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))  # row: true, column: pred
        self.truths = []
        self.predictions = []
        self.roc_preds = []
    
    def add_preds(self, preds, truths):
        # convertiing to probabilities for roc_auc metric
        if self.n_classes == 2:
            roc_pred = self.act_fn(preds)[:, -1].clone().detach().cpu().numpy()
        else:
            roc_pred = self.act_fn(preds).clone().detach().cpu().numpy()
            # raise NotImplementedError('I need to investigate more (the roc_pred array etc) the roc_auc for more than two calsses')
        self.roc_preds.extend(roc_pred)
            
        preds = self.act_fn(preds).max(dim=1)[1].data.flatten().detach().cpu().numpy()  # output of max is a tuple of two output tensors (max, max_indices)
        truths = truths.flatten().detach().cpu().numpy()
        self.predictions.extend(preds)
        self.truths.extend(truths)
        np.add.at(self.confusion_matrix, (truths, preds), 1)  # adds to conf matrix in-place - for each index, gets the rows from y_true, and the columns from y_pred
        
    @staticmethod
    def calc_mean_per_class_acc(confusion_matrix):
        with np.errstate(divide='ignore', invalid='ignore'):  # so it does not war for division by zero
            divided = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)  # this might have inf values, if no sample of a class is not present in the current batch
        result = np.mean(np.ma.fix_invalid(np.ma.masked_invalid(divided), fill_value=0).data)  # replace invalid values with 0
        return result

    def get_values(self, use_dist=False, do_reset=True, return_conf_matrix=False):
        if use_dist:
            synchronize()
            truths = sum(dist_gather(self.truths), [])
            predictions = sum(dist_gather(self.predictions), [])
            roc_preds = np.asarray(sum(dist_gather(self.roc_preds), []))
        else:
            truths = self.truths
            predictions = self.predictions 
            roc_preds = self.roc_preds
            
        accuracy = metrics.accuracy_score(y_true=truths, y_pred=predictions)
        accuracy_mean_per_class = self.calc_mean_per_class_acc(self.confusion_matrix)
        if self.n_classes > 2:
            quadratic_kappa = metrics.cohen_kappa_score(y1=truths, y2=predictions, weights='quadratic')
        else:  # skip -- sklearn cannot handle this
            quadratic_kappa = 0.
            
        recall = metrics.recall_score(y_true=truths, y_pred=predictions, average='macro', zero_division=0)
        try:
            # if all classes are present in the batch compute roc_auc
            roc_auc = metrics.roc_auc_score(y_true=truths, y_score=np.asarray(roc_preds), average='macro', multi_class='ovo')
        except:
            # if NOT all classes are present just give the values of 0.5 to avoid extensive error handling
            roc_auc = 0.5

        if do_reset:
            self.reset()
        # return metrics as dictionary
        results = edict({
            self.prefix + "accuracy": round(accuracy, 3),
            self.prefix + "mean_per_class_accuracy": round(accuracy_mean_per_class, 3),
            self.prefix + "quadratic_kappa": round(quadratic_kappa, 3),
            self.prefix + "roc_auc": round(roc_auc, 3),
            self.prefix + "recall": round(recall, 3)
        })
        if return_conf_matrix:
            results['confusion_matrix'] = self.confusion_matrix
        return results


class MultiLabelClassificationMetrics:
    def __init__(self, n_classes, int_to_labels=None, act_threshold=0.5, mode=""):
        self.mode = mode
        self.prefix = ""
        if mode:
            self.prefix = mode + "_"       
        self.n_classes = n_classes
        if int_to_labels is None:
            int_to_labels = {val:'class_'+str(val) for val in range(n_classes)}
        self.labels = np.arange(n_classes)
        self.int_to_labels = int_to_labels
        self.truths = []
        self.predictions = []
        self.activation = nn.Sigmoid()
        self.act_threshold = act_threshold
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.truths = []
        self.predictions = []        
    
    # add predictions to confusion matrix etc
    def add_preds(self, y_pred, y_true, using_knn=False):       
        y_true = y_true.int().detach().cpu().numpy()
        y_pred = self.preds_from_logits(y_pred)
        self.truths += (y_true.tolist())
        self.predictions += (y_pred.tolist())
    
    # pass signal through activation and thresholding
    def preds_from_logits(self, preds):
        preds = self.activation(preds)
        return preds.detach().cpu().numpy()
    
    def threshold_preds(self, preds):
        preds = preds > self.act_threshold
        if isinstance(preds, torch.Tensor):
            return preds.int().detach().cpu().numpy()
        else:
            return preds * 1
    
    # Calculate and report metrics
    def get_value(self, use_dist=True):
        if use_dist:
            synchronize()
            truths = np.array(sum(dist_gather(self.truths), []))
            predictions = np.array(sum(dist_gather(self.predictions), []))
        else:
            truths = np.array(self.truths)
            predictions = np.array(self.predictions) 
            
        try:
            mAP = metrics.average_precision_score(truths, predictions, average='macro')
        except:
            mAP = 0.                    
        roc_auc = mean_roc_auc(truths, predictions)        
        
        predictions = self.threshold_preds(predictions)
        self.confusion_matrix = metrics.multilabel_confusion_matrix(truths, predictions)     
        
        accuracy = metrics.accuracy_score(truths, predictions)
        precision = metrics.precision_score(truths, predictions, average='macro', 
                                            labels=self.labels, zero_division=0)
        recall = metrics.recall_score(truths, predictions, average='macro', 
                                            labels=self.labels, zero_division=0)
        f1 = metrics.f1_score(truths, predictions, average='macro', 
                                            labels=self.labels, zero_division=0)
        
        # return metrics as dictionary
        return edict({self.prefix + "accuracy" : round(accuracy, 3),
                        self.prefix + "mAP" : round(mAP, 3),
                        self.prefix + "precision" : round(precision, 3),
                        self.prefix + "recall" : round(recall, 3),
                        self.prefix + "f1" : round(f1, 3),
                        self.prefix + "roc_auc" : round(roc_auc, 3)}
                    )    
