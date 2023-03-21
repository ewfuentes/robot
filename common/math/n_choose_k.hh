
namespace robot::math {
constexpr int n_choose_k(const int n, const int k) {
    int num = 1;
    int den = 1;
    for (int i = 1; i <= k; i++) {
        num *= n + 1 - i;
        den *= i;
    }
    return num / den;
}
}  // namespace robot::math
