#include "IOUtil.hpp"

namespace sfz {

DynString convertToOSPath(const char *path) noexcept
{
	size_t pathLen = std::strlen(path);
	DynString outPath("", uint32_t(pathLen));

	auto& internalArray = outPath.internalDynArray();
	for (size_t i = 0; i < pathLen; i++) {
		char c = path[i];
#ifdef _WIN32
		if (c == '/') {
			c = '\\';
		}
#else
		if (c == '\\') {
			c = '/';
		}
#endif
		internalArray[i] = c;
	}
	internalArray[pathLen] = '\0';
	return outPath;
}

} // namespace sfz
