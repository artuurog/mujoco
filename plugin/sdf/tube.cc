#include "tube.h"

#include <cstdint>
#include <optional>
#include <utility>

#include <mujoco/mjplugin.h>
#include <mujoco/mjtype.h>
#include <mujoco/mujoco.h>

namespace mujoco::plugin::sdf {

namespace {

mjtNum distance(
    const mjtNum p[3],
    const mjtNum attr[3]) {

  const mjtNum Ri = attr[0];
  const mjtNum Ro = attr[1];
  const mjtNum h  = attr[2];

  const mjtNum rho =
      mju_sqrt(p[0]*p[0] + p[1]*p[1]);

  // La sezione del tubo nel piano (rho,z)
  // e' un rettangolo centrato in:
  const mjtNum Rmid = 0.5 * (Ri + Ro);

  // meta' dello spessore radiale
  const mjtNum radial_half_width =
      0.5 * (Ro - Ri);

  const mjtNum qx =
      mju_abs(rho - Rmid)
      - radial_half_width;

  const mjtNum qz =
      mju_abs(p[2]) - h;

  const mjtNum ox = mjMAX(qx, 0);
  const mjtNum oz = mjMAX(qz, 0);

  const mjtNum outside =
      mju_sqrt(ox*ox + oz*oz);

  const mjtNum inside =
      mjMIN(mjMAX(qx, qz), 0);

  return outside + inside;
}

}  // namespace


std::optional<Tube> Tube::Create(
    const mjModel* m,
    mjData* d,
    int instance) {

  if (CheckAttr("inner_radius", m, instance) &&
      CheckAttr("outer_radius", m, instance) &&
      CheckAttr("half_height", m, instance)) {

    return Tube(m, d, instance);
  }

  mju_warning("Invalid parameters in Tube SDF plugin");
  return std::nullopt;
}


Tube::Tube(
    const mjModel* m,
    mjData* d,
    int instance) {

  SdfDefault<TubeAttribute> defattribute;

  for (int i = 0;
       i < TubeAttribute::nattribute;
       i++) {

    attribute[i] =
        defattribute.GetDefault(
            TubeAttribute::names[i],
            mj_getPluginConfig(
                m,
                instance,
                TubeAttribute::names[i]));
  }
}


mjtNum Tube::Distance(
    const mjtNum point[3]) const {

  return distance(point, attribute);
}