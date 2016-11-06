// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

#include <sfz/containers/DynArray.hpp>
#include <sfz/math/Matrix.hpp>

namespace phe {

using sfz::DynArray;
using sfz::mat4;
using sfz::vec3;
using std::uint32_t;

// SceneGraphNode
// ------------------------------------------------------------------------------------------------

struct SceneGraphNode final {
	uint32_t entityId = ~0u;
	mat4 transform = sfz::identityMatrix4<float>();
	uint32_t parent = ~0u;
	DynArray<uint32_t> children;
	
	// TODO:
	// Store something smarter than matrix, quaternion?
	// Bounding box?
	// Dirty flag?
	// Left/Right sibling + leftmost child indices instead of DynArray?
};

// SceneGraph
// ------------------------------------------------------------------------------------------------

class SceneGraph final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	SceneGraph() noexcept = default;
	SceneGraph(const SceneGraph&) = delete;
	SceneGraph& operator= (const SceneGraph&) = delete;

	SceneGraph(uint32_t maxNumNodes) noexcept;
	SceneGraph(SceneGraph&& other) noexcept;
	SceneGraph& operator= (SceneGraph&& other) noexcept;
	~SceneGraph() noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	void swap(SceneGraph& other) noexcept;
	void destroy() noexcept;
	bool isValid() const noexcept { return maxNumNodes() > 0u; }

	SceneGraphNode& getNode(uint32_t index) noexcept { return mNodes[index]; }
	SceneGraphNode& rootNode() noexcept { return getNode(0); }
	uint32_t maxNumNodes() const noexcept { return mNodes.size(); }

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	DynArray<SceneGraphNode> mNodes;
	DynArray<uint32_t> mFreeNodes;
};

} // namespace phe
