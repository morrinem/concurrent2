#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <stdint.h>
#include <x86intrin.h>

#include "conv-test.h"

void defaultconv(float *** image, int16_t **** kernels,
		       float *** output, int width, int height,
		       int nchannels, int nkernels, int kernel_order)
{
  int h, w, x, y, c, m;
  
  for ( m = 0; m < nkernels; m++ ) {
    for ( w = 0; w < width; w++ ) {
      for ( h = 0; h < height; h++ ) {
        double sum = 0.0;
        for ( c = 0; c < nchannels; c++ ) {
          for ( x = 0; x < kernel_order; x++) {
            for ( y = 0; y < kernel_order; y++ ) {
              sum += image[w+x][h+y][c] * kernels[m][c][x][y];
            }
          }
          output[m][w][h] = (float) sum;
        }
        
      }
    }
  }
}

void noparallel(float *** image, int16_t **** kernels, float *** output,
               int width, int height, int nchannels, int nkernels,
               int kernel_order)
{
  // this call here is just dummy code that calls the slow, simple, correct version.
  // insert your own code instead
  
  //multichannel_conv(image, kernels, output, width, height, nchannels, nkernels, kernel_order);
  //return;
  
  //image and kernels both contain channel restriction, can rearrange kernels matrix so that c is last, and can be vectorised
  //kernels[m][c][x][y] -> kernelsRearranged[m][x][y][c]
  
  //int h, w, x, y, c, m;
  int16_t ****kernelsRearranged = new_empty_4d_matrix_int16(nkernels, kernel_order, kernel_order, nchannels);
  
  for (int m = 0; m < nkernels; m++)
  {
    for (int c = 0; c < nchannels; c++)
    {
      for (int x = 0; x < kernel_order; x++)
      {
        for (int y = 0; y < kernel_order; y++)
        {
          kernelsRearranged[m][x][y][c] = kernels[m][c][x][y];
        }
      }
    }
  }
  
  for (int m = 0; m < nkernels; m++ ) {
    for (int w = 0; w < width; w++ ) {
      for (int h = 0; h < height; h++ ) {
        __m128d vec_sum[2] = {_mm_setzero_pd(), _mm_setzero_pd()};
        
        for (int x = 0; x < kernel_order; x++) {
          for (int y = 0; y < kernel_order; y++ ) {
            for (int c = 0; c < nchannels; c+=4 ) {
              
              __m128 vec_image = _mm_loadu_ps(&image[w+x][h+y][c]);
              const float kernelfloat[4] = {(float) kernelsRearranged[m][x][y][c], (float) kernelsRearranged[m][x][y][c+1], (float) kernelsRearranged[m][x][y][c+2], (float) kernelsRearranged[m][x][y][c+3]};
              __m128 vec_kernel = _mm_loadu_ps(&kernelfloat[0]);
              __m128 vec_mul = _mm_mul_ps(vec_image, vec_kernel);
              
              vec_sum[0] = _mm_add_pd(vec_sum[0], _mm_cvtps_pd(vec_mul));
              vec_mul = _mm_shuffle_ps(vec_mul, vec_mul, _MM_SHUFFLE(1, 0, 3, 2));
              vec_sum[1] = _mm_add_pd(vec_sum[1], _mm_cvtps_pd(vec_mul));
            }
          }
        }
        vec_sum[0] = _mm_hadd_pd(vec_sum[0], vec_sum[1]);
        vec_sum[0] = _mm_hadd_pd(vec_sum[0], vec_sum[0]);
        
        output[m][w][h] = (float) _mm_cvtsd_f64(vec_sum[0]);
      }
    }
  }
}

void outsideparallel(float *** image, int16_t **** kernels, float *** output,
               int width, int height, int nchannels, int nkernels,
               int kernel_order)
{
  // this call here is just dummy code that calls the slow, simple, correct version.
  // insert your own code instead
  
  //multichannel_conv(image, kernels, output, width, height, nchannels, nkernels, kernel_order);
  //return;
  
  //image and kernels both contain channel restriction, can rearrange kernels matrix so that c is last, and can be vectorised
  //kernels[m][c][x][y] -> kernelsRearranged[m][x][y][c]
  
  //int h, w, x, y, c, m;
  int16_t ****kernelsRearranged = new_empty_4d_matrix_int16(nkernels, kernel_order, kernel_order, nchannels);
  
  for (int m = 0; m < nkernels; m++)
  {
    for (int c = 0; c < nchannels; c++)
    {
      for (int x = 0; x < kernel_order; x++)
      {
        for (int y = 0; y < kernel_order; y++)
        {
          kernelsRearranged[m][x][y][c] = kernels[m][c][x][y];
        }
      }
    }
  }
  
  #pragma omp parallel for
  for (int m = 0; m < nkernels; m++ ) {
    for (int w = 0; w < width; w++ ) {
      for (int h = 0; h < height; h++ ) {
        __m128d vec_sum[2] = {_mm_setzero_pd(), _mm_setzero_pd()};
        
        for (int x = 0; x < kernel_order; x++) {
          for (int y = 0; y < kernel_order; y++ ) {
            for (int c = 0; c < nchannels; c+=4 ) {
              
              __m128 vec_image = _mm_loadu_ps(&image[w+x][h+y][c]);
              const float kernelfloat[4] = {(float) kernelsRearranged[m][x][y][c], (float) kernelsRearranged[m][x][y][c+1], (float) kernelsRearranged[m][x][y][c+2], (float) kernelsRearranged[m][x][y][c+3]};
              __m128 vec_kernel = _mm_loadu_ps(&kernelfloat[0]);
              __m128 vec_mul = _mm_mul_ps(vec_image, vec_kernel);
              
              vec_sum[0] = _mm_add_pd(vec_sum[0], _mm_cvtps_pd(vec_mul));
              vec_mul = _mm_shuffle_ps(vec_mul, vec_mul, _MM_SHUFFLE(1, 0, 3, 2));
              vec_sum[1] = _mm_add_pd(vec_sum[1], _mm_cvtps_pd(vec_mul));
            }
          }
        }
        vec_sum[0] = _mm_hadd_pd(vec_sum[0], vec_sum[1]);
        vec_sum[0] = _mm_hadd_pd(vec_sum[0], vec_sum[0]);
        
        output[m][w][h] = (float) _mm_cvtsd_f64(vec_sum[0]);
      }
    }
  }
}

void maximumparallel(float *** image, int16_t **** kernels, float *** output,
               int width, int height, int nchannels, int nkernels,
               int kernel_order)
{
  // this call here is just dummy code that calls the slow, simple, correct version.
  // insert your own code instead
  
  //multichannel_conv(image, kernels, output, width, height, nchannels, nkernels, kernel_order);
  //return;
  
  //image and kernels both contain channel restriction, can rearrange kernels matrix so that c is last, and can be vectorised
  //kernels[m][c][x][y] -> kernelsRearranged[m][x][y][c]
  
  //int h, w, x, y, c, m;
  int16_t ****kernelsRearranged = new_empty_4d_matrix_int16(nkernels, kernel_order, kernel_order, nchannels);
  
  for (int m = 0; m < nkernels; m++)
  {
    for (int c = 0; c < nchannels; c++)
    {
      for (int x = 0; x < kernel_order; x++)
      {
        for (int y = 0; y < kernel_order; y++)
        {
          kernelsRearranged[m][x][y][c] = kernels[m][c][x][y];
        }
      }
    }
  }
  
  //
  int mwhlimit = nkernels*width*height;
  //h = wh%height;
  //w = wh/width;
  #pragma omp parallel for
  for (int mwh = 0; mwh < mwhlimit; mwh++ ) {
    int m = mwh / (width * height);
    int w = (mwh/nkernels) % width;
    int h = mwh % height;
    __m128d vec_sum[2] = {_mm_setzero_pd(), _mm_setzero_pd()};
    
    for (int x = 0; x < kernel_order; x++) {
      for (int y = 0; y < kernel_order; y++ ) {
        for (int c = 0; c < nchannels; c+=4 ) {
          
          __m128 vec_image = _mm_loadu_ps(&image[w+x][h+y][c]);
          float kernelfloat[4] = {(float) kernelsRearranged[m][x][y][c], (float) kernelsRearranged[m][x][y][c+1], (float) kernelsRearranged[m][x][y][c+2], (float) kernelsRearranged[m][x][y][c+3]};
          __m128 vec_kernel = _mm_loadu_ps(&kernelfloat[0]);
          __m128 vec_mul = _mm_mul_ps(vec_image, vec_kernel);
          
          vec_sum[0] = _mm_add_pd(vec_sum[0], _mm_cvtps_pd(vec_mul));
          vec_mul = _mm_shuffle_ps(vec_mul, vec_mul, _MM_SHUFFLE(1, 0, 3, 2));
          vec_sum[1] = _mm_add_pd(vec_sum[1], _mm_cvtps_pd(vec_mul));
        }
      }
    }
    vec_sum[0] = _mm_hadd_pd(vec_sum[0], vec_sum[1]);
    vec_sum[0] = _mm_hadd_pd(vec_sum[0], vec_sum[0]);
    
    output[m][w][h] = (float) _mm_cvtsd_f64(vec_sum[0]);
  }
}

void maximumconstparallel(float *** image, int16_t **** kernels, float *** output,
               int width, int height, int nchannels, int nkernels,
               int kernel_order)
{
  int16_t ****kernelsRearranged = new_empty_4d_matrix_int16(nkernels, kernel_order, kernel_order, nchannels);
  
  
  for (int m = 0; m < nkernels; m++)
  {
    for (int c = 0; c < nchannels; c++)
    {
      for (int x = 0; x < kernel_order; x++)
      {
        for (int y = 0; y < kernel_order; y++)
        {
          kernelsRearranged[m][x][y][c] = kernels[m][c][x][y];
        }
      }
    }
  }
  
  int mwhlimit = nkernels*width*height;
  
  #pragma omp parallel for
  for (int mwh = 0; mwh < mwhlimit; mwh++ ) {
    const int m = mwh / (width * height);
    const int w = (mwh/nkernels) % width;
    const int h = mwh % height;
    __m128d vec_sum[2] = {_mm_setzero_pd(), _mm_setzero_pd()};
    
    for (int x = 0; x < kernel_order; x++) {
      for (int y = 0; y < kernel_order; y++ ) {
        for (int c = 0; c < nchannels; c+=4 ) {
          
          const __m128 vec_image = _mm_loadu_ps(&image[w+x][h+y][c]);
          const float kernelfloat[4] = {(float) kernelsRearranged[m][x][y][c], (float) kernelsRearranged[m][x][y][c+1], (float) kernelsRearranged[m][x][y][c+2], (float) kernelsRearranged[m][x][y][c+3]};
          const __m128 vec_kernel = _mm_loadu_ps(&kernelfloat[0]);
          __m128 vec_mul = _mm_mul_ps(vec_image, vec_kernel);
          
          vec_sum[0] = _mm_add_pd(vec_sum[0], _mm_cvtps_pd(vec_mul));
          vec_mul = _mm_shuffle_ps(vec_mul, vec_mul, _MM_SHUFFLE(1, 0, 3, 2));
          vec_sum[1] = _mm_add_pd(vec_sum[1], _mm_cvtps_pd(vec_mul));
        }
      }
    }
    vec_sum[0] = _mm_hadd_pd(vec_sum[0], vec_sum[1]);
    vec_sum[0] = _mm_hadd_pd(vec_sum[0], vec_sum[0]);
    
    output[m][w][h] = (float) _mm_cvtsd_f64(vec_sum[0]);
  }
}

void rearrangechannelparallel(float *** image, int16_t **** kernels, float *** output,
               int width, int height, int nchannels, int nkernels,
               int kernel_order)
{
  int16_t ****kernelsRearranged = new_empty_4d_matrix_int16(nkernels, kernel_order, kernel_order, nchannels);
  
  for (int m = 0; m < nkernels; m++)
  {
    #pragma omp parallel for
    for (int c = 0; c < nchannels; c++)
    {
      for (int x = 0; x < kernel_order; x++)
      {
        for (int y = 0; y < kernel_order; y++)
        {
          kernelsRearranged[m][x][y][c] = kernels[m][c][x][y];
        }
      }
    }
  }
  
  int mwhlimit = nkernels*width*height;
  
  #pragma omp parallel for
  for (int mwh = 0; mwh < mwhlimit; mwh++ ) {
    const int m = mwh / (width * height);
    const int w = (mwh/nkernels) % width;
    const int h = mwh % height;
    __m128d vec_sum[2] = {_mm_setzero_pd(), _mm_setzero_pd()};
    
    for (int x = 0; x < kernel_order; x++) {
      for (int y = 0; y < kernel_order; y++ ) {
        for (int c = 0; c < nchannels; c+=4 ) {
          
          const __m128 vec_image = _mm_loadu_ps(&image[w+x][h+y][c]);
          const float kernelfloat[4] = {(float) kernelsRearranged[m][x][y][c], (float) kernelsRearranged[m][x][y][c+1], (float) kernelsRearranged[m][x][y][c+2], (float) kernelsRearranged[m][x][y][c+3]};
          const __m128 vec_kernel = _mm_loadu_ps(&kernelfloat[0]);
          __m128 vec_mul = _mm_mul_ps(vec_image, vec_kernel);
          
          vec_sum[0] = _mm_add_pd(vec_sum[0], _mm_cvtps_pd(vec_mul));
          vec_mul = _mm_shuffle_ps(vec_mul, vec_mul, _MM_SHUFFLE(1, 0, 3, 2));
          vec_sum[1] = _mm_add_pd(vec_sum[1], _mm_cvtps_pd(vec_mul));
        }
      }
    }
    vec_sum[0] = _mm_hadd_pd(vec_sum[0], vec_sum[1]);
    vec_sum[0] = _mm_hadd_pd(vec_sum[0], vec_sum[0]);
    
    output[m][w][h] = (float) _mm_cvtsd_f64(vec_sum[0]);
  }
}

void rearrangeparallel(float *** image, int16_t **** kernels, float *** output,
               int width, int height, int nchannels, int nkernels,
               int kernel_order)
{
  int16_t ****kernelsRearranged = new_empty_4d_matrix_int16(nkernels, kernel_order, kernel_order, nchannels);
  
  #pragma omp parallel for
  for (int m = 0; m < nkernels; m++)
  {
    for (int c = 0; c < nchannels; c++)
    {
      for (int x = 0; x < kernel_order; x++)
      {
        for (int y = 0; y < kernel_order; y++)
        {
          kernelsRearranged[m][x][y][c] = kernels[m][c][x][y];
        }
      }
    }
  }
  
  int mwhlimit = nkernels*width*height;
  
  #pragma omp parallel for
  for (int mwh = 0; mwh < mwhlimit; mwh++ ) {
    const int m = mwh / (width * height);
    const int w = (mwh/nkernels) % width;
    const int h = mwh % height;
    __m128d vec_sum[2] = {_mm_setzero_pd(), _mm_setzero_pd()};
    
    for (int x = 0; x < kernel_order; x++) {
      for (int y = 0; y < kernel_order; y++ ) {
        for (int c = 0; c < nchannels; c+=4 ) {
          
          const __m128 vec_image = _mm_loadu_ps(&image[w+x][h+y][c]);
          const float kernelfloat[4] = {(float) kernelsRearranged[m][x][y][c], (float) kernelsRearranged[m][x][y][c+1], (float) kernelsRearranged[m][x][y][c+2], (float) kernelsRearranged[m][x][y][c+3]};
          const __m128 vec_kernel = _mm_loadu_ps(&kernelfloat[0]);
          __m128 vec_mul = _mm_mul_ps(vec_image, vec_kernel);
          
          vec_sum[0] = _mm_add_pd(vec_sum[0], _mm_cvtps_pd(vec_mul));
          vec_mul = _mm_shuffle_ps(vec_mul, vec_mul, _MM_SHUFFLE(1, 0, 3, 2));
          vec_sum[1] = _mm_add_pd(vec_sum[1], _mm_cvtps_pd(vec_mul));
        }
      }
    }
    vec_sum[0] = _mm_hadd_pd(vec_sum[0], vec_sum[1]);
    vec_sum[0] = _mm_hadd_pd(vec_sum[0], vec_sum[0]);
    
    output[m][w][h] = (float) _mm_cvtsd_f64(vec_sum[0]);
  }
}

void rearrange2dparallel(float *** image, int16_t **** kernels, float *** output,
               int width, int height, int nchannels, int nkernels,
               int kernel_order)
{
  int16_t ****kernelsRearranged = new_empty_4d_matrix_int16(nkernels, kernel_order, kernel_order, nchannels);
  
  const int mclimit = nkernels*nchannels;
  
  #pragma omp parallel for
  for (int mc = 0; mc < mclimit; mc++)
  {
    const int m = mc/nkernels;
    const int c = mc%nchannels;
    for (int x = 0; x < kernel_order; x++)
    {
      for (int y = 0; y < kernel_order; y++)
      {
        kernelsRearranged[m][x][y][c] = kernels[m][c][x][y];
      }
    }
  }
  
  int mwhlimit = nkernels*width*height;
  
  #pragma omp parallel for
  for (int mwh = 0; mwh < mwhlimit; mwh++ ) {
    const int m = mwh / (width * height);
    const int w = (mwh / nkernels) % width;
    const int h = mwh % height;
    __m128d vec_sum[2] = {_mm_setzero_pd(), _mm_setzero_pd()};
    
    for (int x = 0; x < kernel_order; x++) {
      for (int y = 0; y < kernel_order; y++ ) {
        for (int c = 0; c < nchannels; c+=4 ) {
          
          const __m128 vec_image = _mm_loadu_ps(&image[w+x][h+y][c]);
          const float kernelfloat[4] = {(float) kernelsRearranged[m][x][y][c], (float) kernelsRearranged[m][x][y][c+1], (float) kernelsRearranged[m][x][y][c+2], (float) kernelsRearranged[m][x][y][c+3]};
          const __m128 vec_kernel = _mm_loadu_ps(&kernelfloat[0]);
          __m128 vec_mul = _mm_mul_ps(vec_image, vec_kernel);
          
          vec_sum[0] = _mm_add_pd(vec_sum[0], _mm_cvtps_pd(vec_mul));
          vec_mul = _mm_shuffle_ps(vec_mul, vec_mul, _MM_SHUFFLE(1, 0, 3, 2));
          vec_sum[1] = _mm_add_pd(vec_sum[1], _mm_cvtps_pd(vec_mul));
        }
      }
    }
    vec_sum[0] = _mm_hadd_pd(vec_sum[0], vec_sum[1]);
    vec_sum[0] = _mm_hadd_pd(vec_sum[0], vec_sum[0]);
    
    output[m][w][h] = (float) _mm_cvtsd_f64(vec_sum[0]);
  }
}

void rearrange4dparallel(float *** image, int16_t **** kernels, float *** output,
               int width, int height, int nchannels, int nkernels,
               int kernel_order)
{
  int16_t ****kernelsRearranged = new_empty_4d_matrix_int16(nkernels, kernel_order, kernel_order, nchannels);
  
  const int mcxylimit = nkernels*nchannels*kernel_order*kernel_order;
  
  #pragma omp parallel for
  for (int mcxy = 0; mcxy < mcxylimit; mcxy++)
  {
    const int m = mcxy / (kernel_order*kernel_order*nchannels);
    const int c = (mcxy / (kernel_order*kernel_order)) % nchannels;
    const int x = (mcxy / kernel_order) % kernel_order;
    const int y = mcxy % kernel_order;
    kernelsRearranged[m][x][y][c] = kernels[m][c][x][y];
  }
  
  int mwhlimit = nkernels*width*height;
  
  #pragma omp parallel for
  for (int mwh = 0; mwh < mwhlimit; mwh++ ) {
    const int m = mwh / (width * height);
    const int w = (mwh/nkernels) % width;
    const int h = mwh % height;
    __m128d vec_sum[2] = {_mm_setzero_pd(), _mm_setzero_pd()};
    
    for (int x = 0; x < kernel_order; x++) {
      for (int y = 0; y < kernel_order; y++ ) {
        for (int c = 0; c < nchannels; c+=4 ) {
          
          const __m128 vec_image = _mm_loadu_ps(&image[w+x][h+y][c]);
          const float kernelfloat[4] = {(float) kernelsRearranged[m][x][y][c], (float) kernelsRearranged[m][x][y][c+1], (float) kernelsRearranged[m][x][y][c+2], (float) kernelsRearranged[m][x][y][c+3]};
          const __m128 vec_kernel = _mm_loadu_ps(&kernelfloat[0]);
          __m128 vec_mul = _mm_mul_ps(vec_image, vec_kernel);
          
          vec_sum[0] = _mm_add_pd(vec_sum[0], _mm_cvtps_pd(vec_mul));
          vec_mul = _mm_shuffle_ps(vec_mul, vec_mul, _MM_SHUFFLE(1, 0, 3, 2));
          vec_sum[1] = _mm_add_pd(vec_sum[1], _mm_cvtps_pd(vec_mul));
        }
      }
    }
    vec_sum[0] = _mm_hadd_pd(vec_sum[0], vec_sum[1]);
    vec_sum[0] = _mm_hadd_pd(vec_sum[0], vec_sum[0]);
    
    output[m][w][h] = (float) _mm_cvtsd_f64(vec_sum[0]);
  }
}