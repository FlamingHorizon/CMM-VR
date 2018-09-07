#!/usr/bin/env python3

import math
import ipdb as pdb
import pprint
from termcolor import colored
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models

from vr.models.layers import init_modules, GlobalAveragePool, Flatten
from vr.models.layers import build_classifier, build_stem, build_cnn_proj
import vr.programs

# added 2017-12-01
from torch.nn.parameter import Parameter


class FiLM(nn.Module):
  """
  A Feature-wise Linear Modulation Layer from
  'FiLM: Visual Reasoning with a General Conditioning Layer'
  """
  def forward(self, x, gammas, betas):
    gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
    betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
    return (gammas * x) + betas


class FiLMedNet(nn.Module):
  def __init__(self, vocab, feature_dim=(1024, 14, 14),
               stem_num_layers=2,
               stem_batchnorm=False,
               stem_kernel_size=3,
               stem_stride=1,
               stem_padding=None,
               num_modules=4,
               module_num_layers=1,
               module_dim=128,
               module_residual=True,
               module_batchnorm=False,
               module_batchnorm_affine=False,
               module_dropout=0,
               module_input_proj=1,
               module_kernel_size=3,
               classifier_proj_dim=512,
               classifier_downsample='maxpool2',
               classifier_fc_layers=(1024,),
               classifier_batchnorm=False,
               classifier_dropout=0,
               condition_method='bn-film',
               condition_pattern=[],
               use_gamma=True,
               use_beta=True,
               use_coords=1,
               
               # for Language part:
               null_token=0,
               start_token=1,
               end_token=2,
               encoder_embed=None,
               encoder_vocab_size=100,
               decoder_vocab_size=100,
               wordvec_dim=200,
               hidden_dim=512,
               rnn_num_layers=1,
               rnn_dropout=0,
               rnn_time_step=None,
               output_batchnorm=False,
               bidirectional=False,
               encoder_type='gru',
               decoder_type='linear',
               gamma_option='linear',
               gamma_baseline=1,
               parameter_efficient=False,
                
               debug_every=float('inf'),
               print_verbose_every=float('inf'),
               verbose=True,
               ):
    super(FiLMedNet, self).__init__()

    self.vocab = vocab
    num_answers = len(vocab['answer_idx_to_token'])

    self.stem_times = []
    self.module_times = []
    self.classifier_times = []
    self.timing = False
    # for image part
    self.num_modules = num_modules
    self.module_num_layers = module_num_layers
    self.module_batchnorm = module_batchnorm
    self.module_dim = module_dim # 128
    self.condition_method = condition_method
    self.use_gamma = use_gamma
    self.use_beta = use_beta
    self.use_coords_freq = use_coords # == 1
    self.feature_dim = feature_dim
    
    # for language part
    self.encoder_type = encoder_type
    self.decoder_type = decoder_type
    self.output_batchnorm = output_batchnorm
    self.bidirectional = bidirectional
    self.rnn_time_step = rnn_time_step
    self.hidden_dim = hidden_dim
    self.num_dir = 2 if self.bidirectional else 1
    self.gamma_option = gamma_option
    self.gamma_baseline = gamma_baseline # =1
    self.debug_every = debug_every
    self.NULL = null_token
    self.START = start_token
    self.END = end_token
    
    self.debug_every = debug_every
    self.print_verbose_every = print_verbose_every

    # initialize rnn
    if self.bidirectional: # yes
      if decoder_type != 'linear':
        raise(NotImplementedError)
      hidden_dim = (int) (hidden_dim / self.num_dir)

    self.func_list = {
      'linear': None,
      'sigmoid': F.sigmoid,
      'tanh': F.tanh,
      'exp': torch.exp,
      'relu': F.relu
    }

    self.cond_feat_size = 2 * self.module_dim * self.module_num_layers  # FiLM params per ResBlock
    if not parameter_efficient:  # parameter_efficient=False only used to load older trained models
      self.cond_feat_size = 4 * self.module_dim + 2 * self.num_modules

    self.encoder_embed = nn.Embedding(encoder_vocab_size, wordvec_dim)
    self.encoder_rnn = init_rnn(self.encoder_type, wordvec_dim, hidden_dim, rnn_num_layers,
                                dropout=rnn_dropout, bidirectional=self.bidirectional)
                                
    # Initialize stem for rnn
    self.use_rnn_stem = False
    self.stem_rnn_size = int(256 / 2)
    if self.use_rnn_stem:
        self.stem_rnn = init_rnn(self.encoder_type, self.hidden_dim, self.stem_rnn_size, rnn_num_layers,
                                dropout=rnn_dropout, bidirectional=self.bidirectional)
        self.hidden_dim = self.stem_rnn_size * 2
        hidden_dim = self.stem_rnn_size
    
    self.condition_block = {}
    for fn_num in range(self.num_modules):
        mod = nn.Linear(
            hidden_dim * self.num_dir, self.cond_feat_size) # gamma, beta for each block.
        self.condition_block[fn_num] = mod
        self.add_module('condition_block_'+str(fn_num), mod)
    
    # build sentence conditioning for each module:
    self.condition_rnn = {}
    self.cond_rnn_pool = False
    self.modulewise_cond = False
    self.cond_cnn_proj = True
    self.cond_rnn_pool_size = 3
    self.cond_rnn_flatten = Flatten()
    
    if self.cond_rnn_pool:
        self.cond_rnn_dim_in = self.module_dim * np.floor((feature_dim[1] - self.cond_rnn_pool_size) / self.cond_rnn_pool_size + 1) * np.floor((feature_dim[1] - self.cond_rnn_pool_size) / self.cond_rnn_pool_size + 1)
        self.cond_rnn_dim_in = int(self.cond_rnn_dim_in)
    else:
        self.cond_rnn_dim_in = self.module_dim * feature_dim[1] * feature_dim[2]
    
    
    
    self.full_pooling = nn.MaxPool2d(kernel_size=self.cond_rnn_pool_size, padding=0)
    
    if self.output_batchnorm:
      self.output_bn = nn.BatchNorm1d(self.cond_feat_size, affine=True)
        
    # Initialize helper variables
    self.stem_use_coords = (stem_stride == 1) and (self.use_coords_freq > 0) 
    self.condition_pattern = condition_pattern 
    if len(condition_pattern) == 0:
      self.condition_pattern = []
      for i in range(self.module_num_layers * self.num_modules):
        self.condition_pattern.append(self.condition_method != 'concat')
    else:
      self.condition_pattern = [i > 0 for i in self.condition_pattern] 
    self.extra_channel_freq = self.use_coords_freq 
    self.block = FiLMedResBlock

    self.num_cond_maps = 2 * self.module_dim if self.condition_method == 'concat' else 0 
    self.fwd_count = 0
    self.num_extra_channels = 2 if self.use_coords_freq > 0 else 0
    if self.debug_every <= -1:
      self.print_verbose_every = 1
    module_H = feature_dim[1] // (stem_stride ** stem_num_layers)  
    module_W = feature_dim[2] // (stem_stride ** stem_num_layers) 
    self.coords = coord_map((module_H, module_W)) 
    self.default_weight = Parameter(torch.ones(1, 1, self.module_dim).type(torch.cuda.FloatTensor), requires_grad=False) # to instead film, not used.
    self.default_bias = Parameter(torch.zeros(1, 1, self.module_dim).type(torch.cuda.FloatTensor), requires_grad=False) # not used.

    # Initialize stem
    stem_feature_dim = feature_dim[0] + self.stem_use_coords * self.num_extra_channels # 1024 + 2
    self.stem = build_stem(stem_feature_dim, module_dim,
                           num_layers=stem_num_layers, with_batchnorm=stem_batchnorm,
                           kernel_size=stem_kernel_size, stride=stem_stride, padding=stem_padding) # stem_batchnorm == 1, kernel_size=3, stride=1, padding=None
    # stem: 1-layer CNN converting 1026 channels into 128 channels.   


    
                         
    # Initialize FiLMed network body
    for fn_num in range(self.num_modules):
      with_cond = self.condition_pattern[self.module_num_layers * fn_num:
                                          self.module_num_layers * (fn_num + 1)]
      mod = self.block(module_dim, with_residual=module_residual, with_batchnorm=module_batchnorm,
                       with_cond=with_cond,
                       dropout=module_dropout, # 0e-2
                       num_extra_channels=self.num_extra_channels,
                       extra_channel_freq=self.extra_channel_freq,
                       with_input_proj=module_input_proj, 
                       num_cond_maps=self.num_cond_maps, 
                       kernel_size=module_kernel_size, 
                       batchnorm_affine=module_batchnorm_affine, 
                       num_layers=self.module_num_layers,
                       condition_method=condition_method, 
                       debug_every=self.debug_every)
      self.add_module('block_' + str(fn_num), mod)
    
    for fn_num in range(self.num_modules):
        mem = LangMemBlock(self.hidden_dim)
        self.add_module('lang_mem_' + str(fn_num), mem)
    
    # proj rnn hidden state to common latent space.
    self.rnn_proj = nn.Linear(hidden_dim * self.num_dir, classifier_proj_dim)
    self.cnn_proj = build_cnn_proj(module_dim + self.num_extra_channels, module_H, module_W, classifier_proj_dim,
                                   classifier_downsample, with_batchnorm=classifier_batchnorm, dropout=classifier_dropout)
    
    # cond_proj_out_dim = self.hidden_dim if self.att_mode == 'simple' else 2 * self.hidden_dim
    cond_proj_out_dim = 2 * self.hidden_dim
    
    if self.cond_cnn_proj:
        if self.modulewise_cond == False:
            self.cnn_proj_for_cond = build_cnn_proj(module_dim + self.num_extra_channels, module_H, module_W, cond_proj_out_dim,
                                   classifier_downsample, with_batchnorm=classifier_batchnorm, dropout=classifier_dropout)
        else:
            for fn_num in range(self.num_modules):
                mod = build_cnn_proj(module_dim + self.num_extra_channels, module_H, module_W, cond_proj_out_dim,
                                   classifier_downsample, with_batchnorm=classifier_batchnorm, dropout=classifier_dropout)
                self.add_module('cnn_proj_for_cond_'+str(fn_num), mod)
    else:
        for fn_num in range(self.num_modules):
            mod = nn.Linear(self.cond_rnn_dim_in, 2 * self.hidden_dim) # gamma, beta for bi-direct
            self.condition_rnn[fn_num] = mod
            self.add_module('condition_rnn_'+str(fn_num), mod)
    
    # Initialize output classifier
    self.classifier = build_classifier(num_answers, classifier_fc_layers, classifier_proj_dim, 
                                        with_batchnorm=classifier_batchnorm, dropout=classifier_dropout)

    init_modules(self.modules())

  def get_dims(self, x=None):
    V_in = self.encoder_embed.num_embeddings
    V_out = self.cond_feat_size
    D = self.encoder_embed.embedding_dim
    H = self.encoder_rnn.hidden_size
    H_full = self.encoder_rnn.hidden_size * self.num_dir
    L = self.encoder_rnn.num_layers * self.num_dir

    N = x.size(0) if x is not None else None
    T_in = x.size(1) if x is not None else None
    T_out = self.num_modules
    return V_in, V_out, D, H, H_full, L, N, T_in, T_out

  def before_rnn(self, x, replace=0):
    N, T = x.size()
    idx = torch.LongTensor(N).fill_(T - 1)

    # Find the last non-null element in each sequence.
    x_cpu = x.cpu()
    for i in range(N):
      for t in range(T - 1):
        if x_cpu.data[i, t] != self.NULL and x_cpu.data[i, t + 1] == self.NULL:
          idx[i] = t
          break
    idx = idx.type_as(x.data)
    x[x.data == self.NULL] = replace
    return x, Variable(idx)

  def encoder(self, x):
    V_in, V_out, D, H, H_full, L, N, T_in, T_out = self.get_dims(x=x)
    x, idx = self.before_rnn(x)  # Tokenized word sequences (questions), end index
    embed = self.encoder_embed(x)
    h0 = Variable(torch.zeros(L, N, H).type_as(embed.data)).cuda()

    if self.encoder_type == 'lstm':
      c0 = Variable(torch.zeros(L, N, H).type_as(embed.data))
      out, _ = self.encoder_rnn(embed, (h0, c0))
    elif self.encoder_type == 'gru':
      out, _ = self.encoder_rnn(embed, h0)
    
    if self.use_rnn_stem:
      h0_2 =  Variable(torch.zeros(L, N, self.stem_rnn_size).type_as(embed.data)).cuda()
      out, _ = self.stem_rnn(out, h0_2)

    # Pull out the hidden state for the last non-null value in each input
    idx = idx.view(N, 1, 1).expand(N, 1, self.hidden_dim)
    return out.gather(1, idx).view(N, self.hidden_dim), out

  def decoder(self, encoded, dims, h0=None, c0=None):
    V_in, V_out, D, H, H_full, L, N, T_in, T_out = dims

    if self.decoder_type == 'linear':
      # (N x H) x (H x T_out*V_out) -> (N x T_out*V_out) -> N x T_out x V_out
      return self.decoder_linear(encoded).view(N, T_out, V_out), (None, None)

    
  def trainable_params(self):
    for name, param in self.named_parameters():
        if param.requires_grad:
            yield param
            
  def forward(self, questions, image_features, save_activations=False):
    # Initialize forward pass and externally viewable activations
    self.fwd_count += 1
    if save_activations:
      self.feats = None
      self.module_outputs = []
      self.cf_input = None

    if self.debug_every <= -2:
      pdb.set_trace()

      
    # encode questions
    q_encode, q_encode_seq = self.encoder(questions)

    
    # encode image features with stem()
    batch_coords = None
    if self.use_coords_freq > 0: # 1
      batch_coords = self.coords.unsqueeze(0).expand(torch.Size((image_features.size(0),) + self.coords.size()))
      
    if self.stem_use_coords: # 1
      image_features = torch.cat([image_features, batch_coords], 1)
    feats = self.stem(image_features)
    if save_activations:
      self.feats = feats
    N, _, H, W = feats.size()    
    
    # condition values
    gammas_rnn = Variable(torch.zeros(N, self.num_modules, self.rnn_time_step)).type(torch.cuda.FloatTensor)
    betas_rnn = Variable(torch.zeros(N, self.num_modules, self.rnn_time_step)).type(torch.cuda.FloatTensor)
    # module input values
    module_inputs_block = Variable(torch.zeros(feats.size()).unsqueeze(1).expand(
      N, self.num_modules, self.module_dim, H, W)).type(torch.cuda.FloatTensor)
    module_inputs_block[:,0] = feats
    module_inputs_rnn = Variable(torch.zeros(q_encode_seq.size()).unsqueeze(1).expand(
      N, self.num_modules, self.rnn_time_step, self.hidden_dim)).type(torch.cuda.FloatTensor)
    module_inputs_rnn[:,0] = q_encode_seq
    module_inputs_rnn_single = Variable(torch.zeros(q_encode.size()).unsqueeze(1).expand(
      N, self.num_modules, self.hidden_dim)).type(torch.cuda.FloatTensor)
    module_inputs_rnn_single[:,0] = q_encode        
    
    
    # print_flag = ((self.fwd_count % self.debug_every) in [1,2]) or (self.debug_every <= -1)
    print_flag = False
    
    # Propagate up the network from low-to-high numbered blocks
    for fn_num in range(self.num_modules): 
        film_b = self._modules['condition_block_'+str(fn_num)](module_inputs_rnn_single[:,fn_num])
        gammas_block, betas_block = torch.split(film_b, self.module_dim, dim=-1)
        # block forward pass with film_b
        block_output = self._modules['block_'+str(fn_num)](module_inputs_block[:,fn_num],
          gammas_block, betas_block, batch_coords)
        # compute condition values used by rnn hidden states
        if self.cond_rnn_pool:
            cond_rnn_in = self.cond_rnn_flatten(self.full_pooling(block_output))
            film_r = self._modules['condition_rnn_'+str(fn_num)](cond_rnn_in)
        else:
            if self.cond_cnn_proj:
                if self.use_coords_freq > 0:
                    proj_input = torch.cat([block_output, batch_coords], 1)
                if self.modulewise_cond:
                    film_r = self._modules['cnn_proj_for_cond_'+str(fn_num)](proj_input)
                else:
                    film_r = self.cnn_proj_for_cond(proj_input)
            else:
                film_r = self._modules['condition_rnn_'+str(fn_num)](block_output.view(N, self.module_dim * self.feature_dim[1] * self.feature_dim[2]))
        
        gammas_rnn, keys_rnn = torch.split(film_r, self.hidden_dim, dim=-1)

            
        # weighted sum of rnn features with film_r (or other forms)
        mem_output, mem_output_single = self._modules['lang_mem_'+str(fn_num)](module_inputs_rnn[:,fn_num], gammas_rnn, keys_rnn, module_inputs_rnn_single[:,fn_num], fn_num, print_flag)
        
        
        # for debugging
        if print_flag:
        # if 0:
            print "rnn input: %d" %fn_num
            print module_inputs_rnn[0,fn_num]
            print "gammas_block: %d" %fn_num
            print gammas_block[0]
            print "betas_block: %d" %fn_num
            print betas_block[0]
            print "block input: %d" %fn_num
            print module_inputs_block[0,fn_num]
            print "block_output: %d" %fn_num
            print block_output[0]
            # print "gammas_rnn: %d" %fn_num
            # print gammas_rnn[0]
            print "keys_rnn: %d" %fn_num
            print keys_rnn[0]
            print "mem_output: %d" %fn_num
            print mem_output[0]
            print "mem_out_single: %d" %fn_num
            print mem_output_single[0]
        if fn_num == (self.num_modules - 1):
            final_block = block_output
            final_mem = mem_output_single
        else:
            # update blocks
            module_inputs_block_updated = module_inputs_block.clone()
            module_inputs_block_updated[:,fn_num+1] = module_inputs_block_updated[:,fn_num+1] + block_output
            module_inputs_block = module_inputs_block_updated
            # update mems
            module_inputs_rnn_updated = module_inputs_rnn.clone()
            module_inputs_rnn_updated[:,fn_num+1] = module_inputs_rnn_updated[:,fn_num+1] + mem_output
            module_inputs_rnn = module_inputs_rnn_updated
            
            module_inputs_rnn_updated_single = module_inputs_rnn_single.clone()
            module_inputs_rnn_updated_single[:,fn_num+1] = module_inputs_rnn_updated_single[:,fn_num+1] + mem_output_single
            module_inputs_rnn_single = module_inputs_rnn_updated_single

    if self.debug_every <= -2:
      pdb.set_trace()

    # Run the final classifier over the resultant, post-modulated features.
    if self.use_coords_freq > 0:
        final_block = torch.cat([final_block, batch_coords], 1)
    if save_activations:
        self.cf_input = final_block
        
    # proj image and language states
    final_block_proj = self.cnn_proj(final_block)
    final_rnn_proj = self.rnn_proj(final_mem)
    final_rnn_proj = final_rnn_proj * F.sigmoid(final_rnn_proj)
    # final_rnn_proj = self.func_list['tanh'](final_rnn_proj)
    # final_repr = torch.nn.functional.normalize(final_block_proj * final_rnn_proj) 
    # final_repr = final_block_proj * final_rnn_proj
    final_repr = final_block_proj
    # final_repr = final_rnn_proj
    out = self.classifier(final_repr)

    if print_flag:
    # if 0:
        # pdb.set_trace()
        print "final_block:"
        print final_block[0]
        print "final_mem:"
        print final_mem[0]
        print "final_block_proj:"
        print final_block_proj[0]
        print "final_rnn_proj:"
        print final_rnn_proj[0]
        print "final_repr:"
        print final_repr[0]
        print "out:"
        print out[0]
    return out


class FiLMedResBlock(nn.Module):
  def __init__(self, in_dim, out_dim=None, with_residual=True, with_batchnorm=True,
               with_cond=[False], dropout=0, num_extra_channels=0, extra_channel_freq=1,
               with_input_proj=0, num_cond_maps=0, kernel_size=3, batchnorm_affine=False,
               num_layers=1, condition_method='bn-film', debug_every=float('inf')):
    if out_dim is None:
      out_dim = in_dim # 128
    super(FiLMedResBlock, self).__init__()
    self.with_residual = with_residual
    self.with_batchnorm = with_batchnorm
    self.with_cond = with_cond
    self.dropout = dropout
    self.extra_channel_freq = 0 if num_extra_channels == 0 else extra_channel_freq
    self.with_input_proj = with_input_proj  # Kernel size of input projection
    self.num_cond_maps = num_cond_maps
    self.kernel_size = kernel_size
    self.batchnorm_affine = batchnorm_affine
    self.num_layers = num_layers
    self.condition_method = condition_method
    self.debug_every = debug_every

    if self.with_input_proj % 2 == 0:
      raise(NotImplementedError)
    if self.kernel_size % 2 == 0:
      raise(NotImplementedError)
    if self.num_layers >= 2:
      raise(NotImplementedError)

    if self.condition_method == 'block-input-film' and self.with_cond[0]:
      self.film = FiLM()
    if self.with_input_proj:
      self.input_proj = nn.Conv2d(in_dim + (num_extra_channels if self.extra_channel_freq >= 1 else 0),
                                  in_dim, kernel_size=self.with_input_proj, padding=self.with_input_proj // 2)

    self.conv1 = nn.Conv2d(in_dim + self.num_cond_maps +
                           (num_extra_channels if self.extra_channel_freq >= 2 else 0),
                            out_dim, kernel_size=self.kernel_size,
                            padding=self.kernel_size // 2)
    if self.condition_method == 'conv-film' and self.with_cond[0]:
      self.film = FiLM()
    if self.with_batchnorm:
      self.bn1 = nn.BatchNorm2d(out_dim, affine=((not self.with_cond[0]) or self.batchnorm_affine))
    if self.condition_method == 'bn-film' and self.with_cond[0]:
      self.film = FiLM()
    if dropout > 0:
      self.drop = nn.Dropout2d(p=self.dropout)
    if ((self.condition_method == 'relu-film' or self.condition_method == 'block-output-film')
         and self.with_cond[0]):
      self.film = FiLM()

    init_modules(self.modules())

  def forward(self, x, gammas=None, betas=None, extra_channels=None, cond_maps=None):
    if self.debug_every <= -2:
      pdb.set_trace()

    if self.condition_method == 'block-input-film' and self.with_cond[0]:
      x = self.film(x, gammas, betas)

    # ResBlock input projection
    if self.with_input_proj:
      if extra_channels is not None and self.extra_channel_freq >= 1:
        x = torch.cat([x, extra_channels], 1)
      x = F.relu(self.input_proj(x))
    out = x

    # ResBlock body
    if cond_maps is not None:
      out = torch.cat([out, cond_maps], 1)
    if extra_channels is not None and self.extra_channel_freq >= 2:
      out = torch.cat([out, extra_channels], 1)
    out = self.conv1(out)
    if self.condition_method == 'conv-film' and self.with_cond[0]:
      out = self.film(out, gammas, betas)
    if self.with_batchnorm:
      out = self.bn1(out)
    if self.condition_method == 'bn-film' and self.with_cond[0]:
      out = self.film(out, gammas, betas)
    if self.dropout > 0:
      out = self.drop(out)
    out = F.relu(out)
    if self.condition_method == 'relu-film' and self.with_cond[0]:
      out = self.film(out, gammas, betas)

    # ResBlock remainder
    if self.with_residual:
      out = x + out
    if self.condition_method == 'block-output-film' and self.with_cond[0]:
      out = self.film(out, gammas, betas)
    return out

class LangMemBlock(nn.Module):
    def __init__(self, hidden_dim, with_batchnorm=True):
        super(LangMemBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.with_batchnorm = with_batchnorm
        # batch norm for rnn feature maps
        self.rnn_batch_norm = nn.BatchNorm1d(self.hidden_dim, affine=False)
        
        # self.att_mode = 'simple'
        self.att_mode = 'paramed'
        self.compute_attention = nn.Linear(self.hidden_dim, 1)
        
            
        init_modules(self.modules())
        
    def forward(self, state_seq, gs, ks, in_single, fn_num, print_flag):
        N, seq_len, dim = state_seq.size()
        
        if self.att_mode == 'paramed': 
            states_mult = state_seq * gs.unsqueeze(1).expand(N,seq_len,dim) + ks.unsqueeze(1).expand(N,seq_len,dim)
            att_values = F.softmax(self.compute_attention(states_mult).squeeze()).unsqueeze(2).expand(N,seq_len,dim)
        else: 
            att_values = F.softmax(torch.matmul(state_seq, ks.unsqueeze(2)).squeeze()).unsqueeze(2).expand(N,seq_len,dim)
            
        out_single = torch.sum(state_seq * att_values, 1)
        out_states = state_seq
       
        if print_flag:
            print "att_values:"
            print att_values[0][:,0]
            print torch.sum(att_values[0][:,0])
            print "out_single:"
            print out_single[0]
            print "in_single:"
            print in_single[0]

        return out_states, out_single
        

    

    

def coord_map(shape, start=-1, end=1):
  """
  Gives, a 2d shape tuple, returns two mxn coordinate maps,
  Ranging min-max in the x and y directions, respectively.
  """
  m, n = shape
  x_coord_row = torch.linspace(start, end, steps=n).type(torch.cuda.FloatTensor)
  y_coord_row = torch.linspace(start, end, steps=m).type(torch.cuda.FloatTensor)
  x_coords = x_coord_row.unsqueeze(0).expand(torch.Size((m, n))).unsqueeze(0)
  y_coords = y_coord_row.unsqueeze(1).expand(torch.Size((m, n))).unsqueeze(0)
  return Parameter(torch.cat([x_coords, y_coords], 0), requires_grad=False)
  
def init_rnn(rnn_type, hidden_dim1, hidden_dim2, rnn_num_layers,
             dropout=0, bidirectional=False):
  if rnn_type == 'gru':
    return nn.GRU(hidden_dim1, hidden_dim2, rnn_num_layers, dropout=dropout,
                  batch_first=True, bidirectional=bidirectional)
  elif rnn_type == 'lstm':
    return nn.LSTM(hidden_dim1, hidden_dim2, rnn_num_layers, dropout=dropout,
                   batch_first=True, bidirectional=bidirectional)
  elif rnn_type == 'linear':
    return None
  else:
    print 'RNN type ' + str(rnn_type) + ' not yet implemented.'
    raise(NotImplementedError)

