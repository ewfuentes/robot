
#include <random>

#include "common/argument_wrapper.hh"
#include "common/indexed_array.hh"
#include "domain/fog.hh"
#include "wise_enum.h"

namespace robot::domain {
WISE_ENUM_CLASS(KuhnAction, PASS, BET);
WISE_ENUM_CLASS(KuhnPlayer, PLAYER1, PLAYER2, CHANCE);

struct KuhnHistory {
    WISE_ENUM_CLASS_MEMBER(Card, Q, K, A)
    using FogCard = Fog<Card, KuhnPlayer>;

    IndexedArray<FogCard, KuhnPlayer> hands;
    std::vector<KuhnAction> actions;
};

struct KuhnPoker {
    using Players = KuhnPlayer;
    using Actions = KuhnAction;
    using History = KuhnPoker;
    using InfoSetId = std::string;
};

KuhnHistory play(const KuhnHistory &history, const KuhnAction &action);
KuhnHistory play(const KuhnHistory &history, InOut<std::mt19937> gen);
std::optional<KuhnPlayer> up_next(const KuhnHistory &history);
std::vector<KuhnAction> possible_actions(const KuhnHistory &history);
std::optional<int> terminal_value(const KuhnHistory &history, const KuhnPlayer player);
}  // namespace robot::domain
