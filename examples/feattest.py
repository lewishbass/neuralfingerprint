# Example regression script using neural fingerprints.
#
# Compares Morgan fingerprints to neural fingerprints.

import autograd.numpy as np
import autograd.numpy.random as npr

from neuralfingerprint import load_data, load_pickle, load_pickle_comp
from neuralfingerprint import build_morgan_deep_net
from neuralfingerprint import build_conv_deep_net
from neuralfingerprint import normalize_array, adam
from neuralfingerprint import build_batched_grad
from neuralfingerprint.util import rmse
from neuralfingerprint import Atom, Bond, Mol, MolFrom
from nnvis import NNvis

from autograd import grad


nvis = NNvis()

model_params = dict(fp_length=50,    # Usually neural fps need far fewer dimensions than morgan.
                    fp_depth=4,      # The depth of the network equals the fingerprint radius.
                    conv_width=20,   # Only the neural fps need this parameter.
                    h1_size=100,     # Size of hidden layer of network on top of fps.
                    L2_reg=np.exp(1))
train_params = dict(num_iters=300,
                    batch_size=100,
                    init_scale=np.exp(-4),
                    step_size=np.exp(-6))

# Define the architecture of the network that sits on top of the fingerprints.
vanilla_net_params = dict(
    layer_sizes = [model_params['fp_length'], model_params['h1_size'], model_params['h1_size']],  # One hidden layer.
    normalize=True, L2_reg = model_params['L2_reg'], nll_func = rmse)

def train_nn(pred_fun, loss_fun, num_weights, train_smiles, train_raw_targets, train_params,
             validation_smiles, validation_raw_targets, seed=0):
    """loss_fun has inputs (weights, smiles, targets)"""
    init_weights = npr.RandomState(seed).randn(num_weights) * train_params['init_scale']

    train_targets, undo_norm = normalize_array(train_raw_targets)

    def callback(weights, iter):
        if iter%10 == 0:
            print(iter)
            dn = 50
            train_preds = undo_norm(pred_fun(weights, train_smiles[:dn]))
            validation_preds = undo_norm(pred_fun(weights, validation_smiles[:dn]))

            
            vr = rmse(validation_preds, validation_raw_targets[:dn])
            tr = rmse(train_preds, train_raw_targets[:dn])
            nvis.graph( np.array([iter, tr,vr]))# + list(np.abs(np.array(validation_preds)-np.array(validation_raw_targets[:dn])))))
            nvis.draw()

    # Build gradient using autograd.
    grad_fun = grad(loss_fun)
    grad_fun_with_data = build_batched_grad(grad_fun, train_params['batch_size'],
                                            train_smiles, train_targets)

    # Optimize weights.
    trained_weights = adam(grad_fun_with_data, init_weights, callback=callback,
                           num_iters=train_params['num_iters'], step_size=train_params['step_size'])

    def predict_func(new_smiles):
        """Returns to the original units that the raw targets were in."""
        return undo_norm(pred_fun(trained_weights, new_smiles))
    return predict_func, trained_weights


def main():
    #print "Loading data..."
    #traindata, valdata, testdata = load_pickle_comp(
    #    'consol.pickle2', (100, 30, 30),
    #    input_name='key', target_name2='BGB+', target_name1='expt')
    traindata, valdata, testdata = load_pickle(
        'consol.pickle2', (100, 50, 3),
        input_name='key', target_name='expt', filter='none')

    train_inputs, train_targets = traindata
    val_inputs,   val_targets   = valdata
    test_inputs,  test_targets  = testdata

    print len(train_inputs), len(test_inputs), len(val_inputs)

    #print(len(train_inputs), len(test_inputs), len(val_inputs))

    def run_conv_experiment():
        conv_layer_sizes = [model_params['conv_width']] * model_params['fp_depth']
        conv_arch_params = {'num_hidden_features' : conv_layer_sizes,
                            'fp_length' : model_params['fp_length'], 'normalize' : 1}
        loss_fun, pred_fun, conv_parser = \
            build_conv_deep_net(conv_arch_params, vanilla_net_params, model_params['L2_reg'])
        num_weights = len(conv_parser)
        predict_func, trained_weights = \
            train_nn(pred_fun, loss_fun, num_weights, train_inputs, train_targets,
                     train_params, validation_smiles=val_inputs, validation_raw_targets=val_targets)
        #test_predictions = predict_func(test_inputs)

        #print(nvis.free.keys())
        #for i in range(1, 2):
            #print(nvis.free.keys()[:i])
            #print(predict_func([nvis.free.keys()[:i]]))
            #print(i)
            #print np.array(predict_func(val_inputs[:i]))- np.array(val_targets[:i])
        inputs = nvis.free.keys()
        predictions = predict_func(inputs)
        for i in range(len(nvis.free.keys())):
            nvis.free[nvis.free.keys()[i]]['ML'] = predictions[i]
            nvis.free[nvis.free.keys()[i]]['MLgroup'] = 'train' if nvis.free.keys()[i] in train_inputs else 'test'
        return predict_func

    return run_conv_experiment()

if __name__ == '__main__':
    
    pf = main()
    


    while(nvis.running):
        nvis.draw()
    nvis.kill()
