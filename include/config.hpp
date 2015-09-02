#ifndef CONFIG_HPP_
#define CONFIG_HPP_

#if defined DOUBLE_PREC
  typedef double real;
#else
  typedef float real;
#endif

static const int kBlockSizeCUDA = 128;

#endif
