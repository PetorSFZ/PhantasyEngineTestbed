// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/containers/DynString.hpp>

namespace phe {

using sfz::DynString;

/// Converts a path containing mixed path separators ('/' or '\\'), return the path
/// using only the separator of the OS
DynString convertToOSPath(const char *path) noexcept;

} // namespace phe
