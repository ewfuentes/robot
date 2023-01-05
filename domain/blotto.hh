
#include <vector>

#include "domain/fog.hh"
#include "wise_enum.h"

namespace robot::domain {

WISE_ENUM_CLASS(BlottoAction, (ASSIGN_500, 500), (ASSIGN_410, 410), (ASSIGN_401, 401),
                (ASSIGN_320, 320), (ASSIGN_311, 311), (ASSIGN_302, 302), (ASSIGN_230, 230),
                (ASSIGN_221, 221), (ASSIGN_212, 212), (ASSIGN_203, 203), (ASSIGN_140, 140),
                (ASSIGN_131, 131), (ASSIGN_122, 122), (ASSIGN_113, 113), (ASSIGN_104, 104),
                (ASSIGN_050, 050), (ASSIGN_041, 041), (ASSIGN_032, 032), (ASSIGN_023, 023),
                (ASSIGN_014, 014), (ASSIGN_005, 005));
WISE_ENUM_CLASS(BlottoPlayer, PLAYER1, PLAYER2);

struct BlottoHistory {
    using FogAction = Fog<BlottoAction, BlottoPlayer>;
    FogAction player_1_action;
    FogAction player_2_action;
};

struct Blotto {
    using Players = BlottoPlayer;
    using Actions = BlottoAction;
    using History = BlottoHistory;
    using InfoSetId = std::string;
};

BlottoHistory play(const BlottoHistory &history, const BlottoAction &action);
std::optional<BlottoPlayer> up_next(const BlottoHistory &history);
std::vector<BlottoAction> possible_actions(const BlottoHistory &history);
std::optional<int> terminal_value(const BlottoHistory &history, const BlottoPlayer player);
double compute_counterfactual_regret(const BlottoHistory &history, const BlottoPlayer player,
                                     const BlottoAction new_action);

}  // namespace robot::domain
