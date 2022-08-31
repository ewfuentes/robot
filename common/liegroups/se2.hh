
namespace robot::liegroups {

class SE2 : public Sophus::SE2d {
   public:
    using Sophus::SE2d;

    double arclength const();
};
}  // namespace robot::liegroups
