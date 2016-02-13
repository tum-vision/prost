#include <map>
#include <string>

#include "factory.hpp"

namespace matlab
{

using namespace prost;

map<string, function<Prox<real>*(size_t, size_t, bool, const mxArray*)>> custom_prox_reg =
{
  // TODO: register your custom prox operators here
};
 
map<string, function<Block<real>*(size_t, size_t, const mxArray*)>> custom_block_reg =
{
  // TODO: register your custom blocks here
};

struct RegisterCustom {
  RegisterCustom() {
    get_prox_reg().insert(custom_prox_reg.begin(), custom_prox_reg.end());
    get_block_reg().insert(custom_block_reg.begin(), custom_block_reg.end());
  }
};

RegisterCustom registerCustom;

} // namespace matlab
