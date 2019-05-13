#coding=utf-8
# 
import json
import numpy
import time

import cPickle as pkl
import h5py

from nlvr_prepare import load_data, int_data, resize_data
from operator import itemgetter


def dump_data():  
    # these three params only used for debugging and irrelevant to real nums used in train_model.
    num_train_samples = 80000
    num_val_samples = 10000
    num_test_samples = 10000
    
    rnn_wordvec_dim = 200
    load_start_time = time.time()
    
    train_image, train_sent, train_labels = load_data('train')
    dev_image, dev_sent, dev_labels = load_data('dev')
    test_image, test_sent, test_labels = load_data('test')
    load_end_time = time.time()
    print 'load data time: %ds' %(load_end_time - load_start_time)
    
    train_labels = numpy.argmax(numpy.array(train_labels)[:num_train_samples], 1)
    dev_labels = numpy.argmax(numpy.array(dev_labels)[:num_val_samples], 1)
    test_labels = numpy.argmax(numpy.array(test_labels)[:num_test_samples], 1)
            
    train_image = resize_data(train_image, (14,56,3))[:num_train_samples].transpose(0,3,1,2)
    dev_image = resize_data(dev_image, (14,56,3))[:num_val_samples].transpose(0,3,1,2)
    test_image = resize_data(test_image, (14,56,3))[:num_test_samples].transpose(0,3,1,2)
    
    print "image shape:"
    print train_image.shape

    vocab = {}
    inv_vocab = {}
    freq_vocab = {}
    train_sent_int, train_sent_mask, train_lengths, max_train_len = int_data(train_sent, vocab, inv_vocab, freq_vocab, rnn_wordvec_dim)
    dev_sent_int, dev_sent_mask, dev_lengths, max_dev_len = int_data(dev_sent, vocab, inv_vocab, freq_vocab, rnn_wordvec_dim)
    test_sent_int, test_sent_mask, test_lengths, max_test_len = int_data(test_sent, vocab, inv_vocab, freq_vocab, rnn_wordvec_dim)
    
    # print freq_vocab
    
    inv_freq_vocab = []
    for k,v in freq_vocab.iteritems():
        inv_freq_vocab.append((k,v))
        
    # print inv_freq_vocab
    # print sorted(inv_freq_vocab, key=itemgetter(1))
    
    f_vocab_freq = open('vocab_freqs_new', 'w')
    for frq in sorted(inv_freq_vocab, key=itemgetter(1)):
        f_vocab_freq.write('%s, %d\n' %(frq[0], frq[1])) 
    f_vocab_freq.close()
    

    train_sent_int = train_sent_int[:num_train_samples]
    dev_sent_int = dev_sent_int[:num_val_samples]
    test_sent_int = test_sent_int[:num_test_samples]
    train_sent_mask = train_sent_mask[:num_train_samples]
    dev_sent_mask = dev_sent_mask[:num_val_samples]
    test_sent_mask = test_sent_mask[:num_test_samples]
    
    train_lengths = train_lengths[:num_train_samples].astype(int)
    dev_lengths = dev_lengths[:num_val_samples].astype(int)
    test_lengths = test_lengths[:num_test_samples].astype(int)

    rnn_time_step = max([max_test_len, max_train_len, max_dev_len])
    print 'rnn_time_step:'
    print rnn_time_step
    
    new_list = []
    for set_data in [train_sent_int, dev_sent_int, test_sent_int]:
        temp_data = numpy.zeros((len(set_data),rnn_time_step))
        temp_data[:,:len(set_data[0])] = set_data
        new_list.append(temp_data)
        
    train_sent_int = new_list[0].astype(int)
    dev_sent_int = new_list[1].astype(int)
    test_sent_int = new_list[2].astype(int)
    
    new_list = []
    for set_data in [train_sent_mask, dev_sent_mask, test_sent_mask]:
        temp_data = numpy.zeros((len(set_data),rnn_time_step, rnn_wordvec_dim))
        temp_data[:,:len(set_data[0]),:] = set_data
        new_list.append(temp_data)
    train_sent_mask = new_list[0]
    dev_sent_mask = new_list[1]
    test_sent_mask = new_list[2]
    
    
    num_train_samples = len(train_sent_int)
    num_val_samples = len(dev_sent_int)
    num_test_samples = len(test_sent_int)
    # random idx
    train_rand_idx = numpy.random.permutation(num_train_samples)
    dev_rand_idx = numpy.random.permutation(num_val_samples)
    test_rand_idx = numpy.random.permutation(num_test_samples)
    
    
    # permute all data
    train_image, train_sent_int, train_labels, train_lengths, train_sent_mask = (train_image[train_rand_idx,:], train_sent_int[train_rand_idx,:], train_labels[train_rand_idx], train_lengths[train_rand_idx], train_sent_mask[train_rand_idx,:])
    dev_image, dev_sent_int, dev_labels, dev_lengths, dev_sent_mask = (dev_image[dev_rand_idx,:], dev_sent_int[dev_rand_idx,:], dev_labels[dev_rand_idx], dev_lengths[dev_rand_idx], dev_sent_mask[dev_rand_idx,:])
    test_image, test_sent_int, test_labels, test_lengths, test_sent_mask = (test_image[test_rand_idx,:], test_sent_int[test_rand_idx,:], test_labels[test_rand_idx], test_lengths[test_rand_idx], test_sent_mask[test_rand_idx,:])
    
    print "writing h5 files:"
    start_time = time.time()
    # write h5 files
    with h5py.File('../../../dataset/NLVR/h5/nlvr_train_raw.h5', 'w') as f:
        img = f.create_dataset('image', train_image.shape, dtype=train_image.dtype)
        img[:] = train_image
        sent = f.create_dataset('sent', train_sent_int.shape, dtype=train_sent_int.dtype)
        sent[:] = train_sent_int
        labels = f.create_dataset('labels', train_labels.shape, dtype=train_labels.dtype)
        labels[:] = train_labels
        lengths = f.create_dataset('lengths', train_lengths.shape, dtype=train_lengths.dtype)
        lengths[:] = train_lengths
        mask = f.create_dataset('mask', train_sent_mask.shape, dtype=train_sent_mask.dtype)
        mask[:] = train_sent_mask
        
    with h5py.File('../../../dataset/NLVR/h5/nlvr_dev_raw.h5', 'w') as f:
        img = f.create_dataset('image', dev_image.shape, dtype=dev_image.dtype)
        img[:] = dev_image
        sent = f.create_dataset('sent', dev_sent_int.shape, dtype=dev_sent_int.dtype)
        sent[:] = dev_sent_int
        labels = f.create_dataset('labels', dev_labels.shape, dtype=dev_labels.dtype)
        labels[:] = dev_labels
        lengths = f.create_dataset('lengths', dev_lengths.shape, dtype=dev_lengths.dtype)
        lengths[:] = dev_lengths
        mask = f.create_dataset('mask', dev_sent_mask.shape, dtype=dev_sent_mask.dtype)
        mask[:] = dev_sent_mask
        
    with h5py.File('../../../dataset/NLVR/h5/nlvr_test_raw.h5', 'w') as f:
        img = f.create_dataset('image', test_image.shape, dtype=test_image.dtype)
        img[:] = test_image
        sent = f.create_dataset('sent', test_sent_int.shape, dtype=test_sent_int.dtype)
        sent[:] = test_sent_int
        labels = f.create_dataset('labels', test_labels.shape, dtype=test_labels.dtype)
        labels[:] = test_labels
        lengths = f.create_dataset('lengths', test_lengths.shape, dtype=test_lengths.dtype)
        lengths[:] = test_lengths
        mask = f.create_dataset('mask', test_sent_mask.shape, dtype=test_sent_mask.dtype)
        mask[:] = test_sent_mask
    
    print "write cost time:"
    print time.time() - start_time
    
    
    start_time = time.time()
    print "loading from h5:"
    test_img_loaded = h5py.File('../../../dataset/NLVR/h5/nlvr_test_raw.h5', 'r')['image']
    test_sent_loaded = h5py.File('../../../dataset/NLVR/h5/nlvr_test_raw.h5', 'r')['sent']
    test_labels_loaded = h5py.File('../../../dataset/NLVR/h5/nlvr_test_raw.h5', 'r')['labels']
    test_lengths_loaded = h5py.File('../../../dataset/NLVR/h5/nlvr_test_raw.h5', 'r')['lengths']
    test_mask_loaded = h5py.File('../../../dataset/NLVR/h5/nlvr_test_raw.h5', 'r')['mask']
    print "loaded shape:"
    print test_img_loaded[0]
    print test_sent_loaded[:]
    print test_labels_loaded[:]
    print test_lengths_loaded[:]
    print test_mask_loaded[:]
    
    print "load cost time:"
    print time.time() - start_time
    
        
    
    pkl.dump((vocab, inv_vocab), open('data/nlvr_vocab_raw.pkl', 'wb'))
        

if __name__ == '__main__':
    dump_data()
        
    
    
    
    
    

        


    
    
