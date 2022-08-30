
#pragma once

#include <memory>

namespace robot {
template <typename T>
class Out {
   public:
    explicit Out() : obj_(std::make_unique<T>()), obj_ptr_(obj_.get()) {}
    explicit Out(T &obj) : obj_(nullptr), obj_ptr_(&obj) {}

    T &operator*() { return *obj_ptr_; }
    T *operator->() { return obj_ptr_; }

   private:
    std::unique_ptr<T> obj_;
    T *obj_ptr_;
};

template <typename T>
class InOut {
   public:
    explicit InOut(T &obj) : obj_(obj) {}

    T &operator*() { return obj_; }
    T *operator->() { return &obj_; }

   private:
    T &obj_;
};

template <typename T>
Out<T> make_out(T &obj) {
    return Out(obj);
}

template <typename T>
Out<T> make_unused() {
    return Out<T>();
}

template <typename T>
InOut<T> make_in_out(T &obj) {
    return InOut(obj);
}

}  // namespace robot
