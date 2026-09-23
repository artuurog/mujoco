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

void Tube::Gradient(
    mjtNum grad[3],
    const mjtNum point[3]) const {

  const mjtNum eps = 1e-8;

  const mjtNum d0 =
      distance(point, attribute);

  mjtNum px[3] = {
      point[0] + eps,
      point[1],
      point[2]
  };

  mjtNum py[3] = {
      point[0],
      point[1] + eps,
      point[2]
  };

  mjtNum pz[3] = {
      point[0],
      point[1],
      point[2] + eps
  };

  grad[0] =
      (distance(px, attribute) - d0) / eps;

  grad[1] =
      (distance(py, attribute) - d0) / eps;

  grad[2] =
      (distance(pz, attribute) - d0) / eps;
}

// plugin registration
void Tube::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.sdf.tube";
  plugin.capabilityflags |= mjPLUGIN_SDF;

  plugin.nattribute = TubeAttribute::nattribute;
  plugin.attributes = TubeAttribute::names;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    auto sdf_or_null = Tube::Create(m, d, instance);
    if (!sdf_or_null.has_value()) {
      return -1;
    }
    d->plugin_data[instance] =
        reinterpret_cast<uintptr_t>(new Tube(std::move(*sdf_or_null)));
    return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<Tube*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };
  plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data,
                     int instance) {
    // do nothing
  };
  plugin.compute =
      +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
        // do nothing;
      };
  plugin.sdf_distance =
      +[](const mjtNum point[3], const mjData* d, int instance) {
        auto* sdf = reinterpret_cast<Tube*>(d->plugin_data[instance]);
        return sdf->Distance(point);
      };
  plugin.sdf_gradient = +[](mjtNum gradient[3], const mjtNum point[3],
                            const mjData* d, int instance) {
    auto* sdf = reinterpret_cast<Tube*>(d->plugin_data[instance]);
    sdf->Gradient(gradient, point);
  };
  plugin.sdf_staticdistance =
      +[](const mjtNum point[3], const mjtNum* attributes) {
        return distance(point, attributes);
      };
  plugin.sdf_aabb = +[](mjtNum aabb[6], const mjtNum* attributes) {
    aabb[0] = aabb[1] = aabb[2] = 0;
    aabb[3] = aabb[4] = attributes[0] + attributes[1];
    aabb[5] = attributes[1];
  };
  plugin.sdf_attribute =
      +[](mjtNum attribute[], const char* name[], const char* value[]) {
        SdfDefault<TubeAttribute> defattribute;
        defattribute.GetDefaults(attribute, name, value);
      };

  mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::sdf
