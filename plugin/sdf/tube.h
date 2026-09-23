#ifndef MUJOCO_PLUGIN_SDF_TUBE_H_
#define MUJOCO_PLUGIN_SDF_TUBE_H_

#include <optional>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
//#include <mujoco/mjtnum.h>
#include <mujoco/mjvisualize.h>

#include "sdf.h"

namespace mujoco::plugin::sdf {

struct TubeAttribute {
  static constexpr int nattribute = 3;

  static constexpr char const* names[nattribute] = {
      "inner_radius",
      "outer_radius",
      "half_height"
  };

  static constexpr mjtNum defaults[nattribute] = {
      0.02,
      0.05,
      0.10
  };
};

class Tube {
 public:
  static std::optional<Tube> Create(
      const mjModel* m,
      mjData* d,
      int instance);

  Tube(Tube&&) = default;
  ~Tube() = default;

  mjtNum Distance(const mjtNum point[3]) const;

  void Gradient(
      mjtNum grad[3],
      const mjtNum point[3]) const;

  static void RegisterPlugin();

  mjtNum attribute[TubeAttribute::nattribute];

 private:
  Tube(
      const mjModel* m,
      mjData* d,
      int instance);
};

}  // namespace mujoco::plugin::sdf

#endif