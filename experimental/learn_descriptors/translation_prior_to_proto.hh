#pragma once

#include "Eigen/Core"
#include "experimental/learn_descriptors/translation_prior.hh"
#include "experimental/learn_descriptors/translation_prior.pb.h"

namespace robot::experimental::learn_descriptors::proto {
void pack_into(const learn_descriptors::TranslationPrior &in, TranslationPrior *out);
learn_descriptors::TranslationPrior unpack_from(const TranslationPrior &in);
}  // namespace robot::experimental::learn_descriptors::proto