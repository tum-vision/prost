#include "factory.hpp"

#include "prost/linop/block_dataterm_sublabel.hpp"

namespace mex
{

using namespace prost;

ProxRegistry custom_prox_reg[] = 
{
  // TODO: insert custom prox operators here

  // The end.
  { "END", nullptr },
};

BlockRegistry custom_block_reg[] = 
{
  // TODO: insert custom blocks here

  // The end.
  { "END", nullptr },
};

}
