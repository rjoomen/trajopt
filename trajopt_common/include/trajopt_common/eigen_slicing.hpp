#pragma once
#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <Eigen/Core>
#include <vector>
TRAJOPT_IGNORE_WARNINGS_POP

namespace trajopt_common
{
template <typename VectorT>
VectorT fancySlice(const VectorT& x, const std::vector<int>& inds)
{
  VectorT out(inds.size());
  for (std::size_t i = 0; i < inds.size(); ++i)
    out[i] = x[inds[i]];
  return out;
}

template <typename VectorT>
std::vector<int> flatnonzero(const VectorT& x)
{
  std::vector<int> out;
  for (int i = 0; i < x.size(); ++i)
    if (x[i] != 0)
      out.push_back(i);
  return out;
}
}  // namespace trajopt_common
