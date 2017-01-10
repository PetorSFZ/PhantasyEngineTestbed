// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/ecs/SceneGraph.hpp"

namespace phe {

// SceneGraph: Constructors & destructors
// ------------------------------------------------------------------------------------------------

SceneGraph::SceneGraph(uint32_t maxNumNodes) noexcept
{
	mNodes = DynArray<SceneGraphNode>(maxNumNodes);
	mFreeNodes = DynArray<uint32_t>(maxNumNodes);
	for (uint32_t i = 0; i < maxNumNodes; i++) {
		mFreeNodes[i] = maxNumNodes - i - 1u;
	}
}

SceneGraph::SceneGraph(SceneGraph&& other) noexcept
{
	this->swap(other);
}

SceneGraph& SceneGraph::operator= (SceneGraph&& other) noexcept
{
	this->swap(other);
	return *this;
}

SceneGraph::~SceneGraph() noexcept
{
	this->destroy();
}

// SceneGraph: Methods
// ------------------------------------------------------------------------------------------------

void SceneGraph::swap(SceneGraph& other) noexcept\
{
	this->mNodes.swap(other.mNodes);
	this->mFreeNodes.swap(other.mFreeNodes);
}

void SceneGraph::destroy() noexcept
{
	this->mNodes.destroy();
	this->mFreeNodes.destroy();
}

} // namespace phe
