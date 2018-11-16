"""
Utilities for implementing the Sum of Conditional Variances as
described by Richa et. al.

R. Richa, R. Sznitman, R. Taylor, and G. Hager, "Visual tracking
using the sum of conditional variance," Intelligent Robots and
Systems (IROS), 2011 IEEE/RSJ International Conference on, pp.
2953-2958, 2011.

Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import numpy as np
from scipy import weave
from scipy.weave import converters

#from ImageUtils import *

def getSCVIntensityMap(src, dst):
    conditional_probability = np.zeros((256,256))
    intensity_map = np.arange(256, dtype=np.float64)
    n = len(src)
    for k in xrange(n):
        conditional_probability[src[k], dst[k]]+=1
    for i in xrange(256):
        normalizer=0
        weighted_sum=0
        for j in xrange(256):
            weighted_sum += j * conditional_probability[i,j]
            normalizer += conditional_probability[i,j]
        if normalizer>0:
            intensity_map[i] = weighted_sum / normalizer
    return intensity_map

def scv_intensity_map(src, dst):
    #print 'starting scv_intensity_map'
    #log_file=open("temp_data.txt","w")
    #log_file.write("src:\n")
    #log_file.write(src)
    #log_file.write("\ndst:\n")
    #log_file.write(dst)
    #np.savetxt(log_file, src)
    #log_file.close()

    conditional_probability = np.zeros((256,256))
    intensity_map = np.arange(256, dtype=np.float64)
    n = len(src)
    code = \
    """
    for (int k = 0; k < n; k++) {
      int i = int(src(k));
      int j = int(dst(k));
      conditional_probability(i,j) += 1;
    }
    for (int i = 0; i < 256; i++) {
      double normalizer = 0;
      double total = 0;
      for (int j = 0; j < 256; j++) {
        total += j * conditional_probability(i,j);
        normalizer += conditional_probability(i,j);
      }
      if (normalizer > 0) {
        intensity_map(i) = total / normalizer;
      }
    }
    """
    #print "executing weave"
    weave.inline(code, ['conditional_probability', 'intensity_map', 'n', 'src', 'dst'],
                 type_converters=converters.blitz,
                 compiler='gcc')
    #print "Done executing weave"
    #print 'done scv_intensity_map'
    return intensity_map

def scv_intensity_map_vec(src, dst):
    # print 'src.shape: ', src.shape
    # print 'dst.shape: ', dst.shape

    if len(src.shape)!=2 or len(dst.shape)!=2:
        print 'src.shape: ', src.shape
        print 'dst.shape: ', dst.shape
        raise SystemExit('Error in scv_intensity_map_vec:\nSource and/or destination images are not multichannel')

    #print 'starting scv_intensity_map'
    #log_file=open("temp_data.txt","w")
    #log_file.write("src:\n")
    #log_file.write(src)
    #log_file.write("\ndst:\n")
    #log_file.write(dst)
    #np.savetxt(log_file, src)
    #log_file.close()

    nchannels=src.shape[0]
    # print 'nchannels=', nchannels

    conditional_probability = np.zeros((256,256))
    intensity_map=np.empty((nchannels, 256))
    for i in xrange(nchannels):
        intensity_map[i, :] = np.arange(256, dtype=np.float64)
    np.savetxt('intensity_map_scv.txt', intensity_map, fmt='%10.5f')
    n = src.shape[1]
    code = \
    """
    for (int ch = 0; ch < nchannels; ch++) {
        for (int k = 0; k < n; k++) {
          int i = int(src(ch, k));
          int j = int(dst(ch, k));
          conditional_probability(i,j) += 1;
        }
        for (int i = 0; i < 256; i++) {
          double normalizer = 0;
          double total = 0;
          for (int j = 0; j < 256; j++) {
            total += j * conditional_probability(i,j);
            normalizer += conditional_probability(i,j);
            conditional_probability(i,j)=0
          }
          if (normalizer > 0) {
            intensity_map(ch, i) = total / normalizer;
          }
        }
    }
    """
    # print "executing weave in scv_intensity_map_vec"
    weave.inline(code, ['conditional_probability', 'intensity_map', 'n', 'src', 'dst', 'nchannels'],
                 type_converters=converters.blitz,
                 compiler='gcc')
    # print "Done executing weave"
    #print 'done scv_intensity_map'
    return intensity_map

def scv_intensity_map_vec2(src_vec, dst_vec):
    #print 'src_vec.shape: ', src_vec.shape
    #print 'dst_vec.shape: ', dst_vec.shape

    if len(src_vec.shape)!=2 or len(dst_vec.shape)!=2:
        print 'src.shape: ', src_vec.shape
        print 'dst.shape: ', dst_vec.shape
        raise SystemExit('Error in scv_intensity_map_vec:\nSource and/or destination images are not multichannel')

    #print 'nchannels=', nchannels

    nchannels=src_vec.shape[0]
    intensity_map_vec=np.empty((nchannels, 256))
    conditional_probability = np.zeros((256,256))
    n = src_vec.shape[1]
    #print 'n=', n
    code = \
    """
    for (int k = 0; k < n; k++) {
      int i = int(src(k));
      int j = int(dst(k));
      conditional_probability(i,j) += 1;
    }
    for (int i = 0; i < 256; i++) {
      double normalizer = 0;
      double total = 0;
      for (int j = 0; j < 256; j++) {
        total += j * conditional_probability(i,j);
        normalizer += conditional_probability(i,j);
      }
      if (normalizer > 0) {
        intensity_map(i) = total / normalizer;
      }
    }
    """
    # print 'src_vec.shape:', src_vec.shape
    # print 'dst_vec.shape:', dst_vec.shape

    # print 'executing weave in scv_intensity_map_vec2'
    for i in xrange(nchannels):
        src=src_vec[i,:]
        dst=dst_vec[i,:]
        # print 'src.shape:', src.shape
        # print 'dst.shape:', dst.shape
        intensity_map = np.arange(256, dtype=np.float64)
        weave.inline(code, ['conditional_probability', 'intensity_map', 'n', 'src', 'dst'],
                     type_converters=converters.blitz,
                     compiler='gcc')
        intensity_map_vec[i,:]=intensity_map
    # print 'Done'
    #np.savetxt('intensity_map_scv.txt', intensity_map_vec, fmt='%10.5f')
    #print "Done executing weave"
    #print 'done scv_intensity_map'
    return intensity_map_vec

def scv_expectation(original, intensity_map):
    return intensity_map[np.floor(original).astype(np.int)]

def scv_expectation_vec(original, intensity_map):
    #print 'original.shape:', original.shape
    expectation=np.empty(original.shape)
    for i in xrange(original.shape[0]):
        ch_map=intensity_map[i,:]
        expectation[i, :]=ch_map[np.floor(original[i, :]).astype(np.int)]
    return expectation
