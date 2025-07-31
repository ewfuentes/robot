#include "experimental/learn_descriptors/translation_prior_to_proto.hh"

#include "common/math/matrix_to_proto.hh"
#include "experimental/learn_descriptors/translation_prior.hh"

namespace robot::experimental::learn_descriptors::proto {
void pack_into(const learn_descriptors::TranslationPrior &in, TranslationPrior *out) {
    pack_into(in.translation, out->mutable_translation());
    pack_into(in.covariance, out->mutable_covariance());
}

learn_descriptors::TranslationPrior unpack_from(const TranslationPrior &in) {
    learn_descriptors::TranslationPrior out;
    out.translation = robot::math::proto::unpack_from<Eigen::Vector3d>(in.translation());
    out.covariance = robot::math::proto::unpack_from<Eigen::Matrix3d>(in.covariance());
    return out;
}
}  // namespace robot::experimental::learn_descriptors::proto