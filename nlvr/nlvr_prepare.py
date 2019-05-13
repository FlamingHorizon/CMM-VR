#coding=utf-8
# 
import json
import numpy as np
from matplotlib import pyplot as plt
from skimage import transform

def load_data(mode_name):
    data_path = '../../../dataset/NLVR/nlvr-master/' + mode_name + '/'
    json_file_name = data_path + '%s.json'%mode_name
    png_path = data_path + 'images/all/'
    
    n_samples = 0
    image_data = []
    sent_data = []
    labels = []
    
    f = open(json_file_name, 'r')
    line = f.readline().strip()
    
    typo_dict = {}
    f_fix_typos = open('fix_typos','r')
    l_tps = f_fix_typos.readline().strip()
    while l_tps:
        k,v = l_tps.split(',')
        typo_dict[k.strip()] = v.strip()
        l_tps = f_fix_typos.readline().strip()
    
    print "typo_dict:"
    print typo_dict
    
    while line:
        info = json.loads(line)
        sent = info['sentence']
        if 'label' in info.keys():
            if info['label'] == 'true':
                label = [1,0]
            else:
                label = [0,1]
            
        else:
            label = ''
        image_id = info['identifier']
        # cut sent
        sent = sent.replace('then,idle', 'the middle')
        sent = sent.replace('a sthe', 'as the')
        sent = sent.replace('ablue', 'a blue')
        sent = sent.replace('isa', 'is a')
        sent = sent.replace('atleast', 'at least')
        sent = sent.replace('tleast', 'at least')
        sent = sent.replace('abox', 'a box')
        sent_words = sent.strip().lower().split()
        real_sent = []
        for word_idx in range(len(sent_words)):
            if sent_words[word_idx][-1] in ['.','/']:
                word = sent_words[word_idx][:-1]
                end_token = '.'
            elif sent_words[word_idx][-1] in [','] and sent_words[word_idx] != ',':
                word = sent_words[word_idx][:-1]
                end_token = ',' 
            else:
                word = sent_words[word_idx]
                end_token = None
                
            if word in typo_dict:
                word = typo_dict[word]
            
            if word.strip() != '':
                real_sent.append(word)
            if end_token is not None:
                real_sent.append(end_token)
        
        if real_sent[-1] != '.':
            real_sent.append('.')
        # print real_sent
                
        # load image
        for i in range(6):
            image_mat = plt.imread(png_path + mode_name + '-' + image_id + '-' + str(i) + '.png')[:,:,:3]
            # if n_samples == 10:
                # print image_mat
            image_data.append(image_mat)
            sent_data.append(real_sent)
            labels.append(label)
            n_samples += 1
        # if n_samples > 300:
            # break
        line = f.readline().strip()
    f.close()
    print n_samples
    return image_data, sent_data, labels
    
def int_data(sent_data, vocab, inv_vocab, freq_vocab, emb_dim):
    n = len(sent_data)
    max_sent_len = max([len(x) for x in sent_data])
    vocab_size = len(vocab)
    for s in sent_data:
        for w in s:
            if w not in vocab.keys():
                vocab_size += 1
                vocab[w] = vocab_size
                inv_vocab[str(vocab_size)] = w
                freq_vocab[w] = 1
            else:
                freq_vocab[w] += 1
    sent_data_int = np.zeros((n, max_sent_len))
    sent_data_mask = np.zeros((n, max_sent_len, emb_dim))
    data_lengths = []
    sent_id = 0
    for s in sent_data:
        int_sent = [vocab[w] for w in s]
        data_lengths.append(len(int_sent))
        sent_data_int[sent_id,:len(int_sent)] = np.array(int_sent)
        sent_data_mask[sent_id,:len(int_sent),:] = np.ones((len(int_sent),emb_dim))
        sent_id += 1
    return sent_data_int, sent_data_mask, np.array(data_lengths), max_sent_len
        
def resize_data(data, shape):
    out = []
    '''for i in range(len(data)):
        pic1 = data[i][:,:100,:]
        pic2 = data[i][:,150:250,:]
        pic3 = data[i][:,300:400,:]
        # print pic1.shape, pic2.shape, pic3.shape
        new_pic = np.concatenate([transform.resize(pic1, shape), transform.resize(pic2, shape), transform.resize(pic3, shape)], -1)
        out.append(new_pic)'''
    
    for i in range(len(data)):
        new_pic = transform.resize(data[i], shape)
        out.append(new_pic)
    return np.array(out)
                
            
    
    
if __name__ == '__main__':    
    image_data, sent_data, labels = load_data('dev')
    vocab = {}
    inv_vocab = {}
    sent_data_int, sent_data_mask, max_sent_len = int_data(sent_data, vocab, inv_vocab)
    for i in range(len(sent_data)):
        print sent_data_int[i,:]
    print vocab
    print inv_vocab
    

        


    
    
