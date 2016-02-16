#ifndef PROST_CONFIG_HPP_
#define PROST_CONFIG_HPP_

namespace prost {

static const size_t kBlockSizeCUDA = 256;
	
#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // disable type-conversion loss of data warnings on windows
#endif

} // namespace prost

#endif // PROST_CONFIG_HPP_
