#pragma once

#include <array>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>

#include "GLFW/glfw3.h"

namespace robot::visualization::gl_window {

template <typename Callable>
class WorkQueue {
   public:
    void submit_work(Callable &&item) {
        std::lock_guard<std::mutex> guard(mutex_);
        items_.emplace(item);
    }

    bool empty() {
        std::lock_guard<std::mutex> guard(mutex_);
        return items_.empty();
    }

    template <typename... Args>
    void complete_work(Args... args) {
        items_.front()(std::forward<Args>(args)...);
        std::lock_guard<std::mutex> guard(mutex_);
        items_.pop();
    }

   private:
    std::mutex mutex_;
    std::queue<Callable> items_;
};

struct WindowData;

struct JoystickState {
    std::array<double, 6> axes;
    std::array<int, 15> buttons;
};

struct WindowDims {
    int width;
    int height;
};

class GlWindow {
   public:
    GlWindow(const int width, const int height, const std::string &title = "gl_window");
    ~GlWindow();

    using RenderCallback = std::function<void()>;
    using MousePosCallback = std::function<void(const double x, const double y)>;
    using MouseButtonCallback =
        std::function<void(const int button, const int action, const int mods)>;
    using KeyboardCallback =
        std::function<void(const int key, const int scancode, const int action, const int mods)>;
    using ResizeCallback = std::function<void(const int width, const int height)>;

    std::unordered_map<int, JoystickState> get_joystick_states();

    // Functions to register callbacks on window events. Note that these are run on the ui thread,
    // so ensure proper synchonization is used.
    void register_render_callback(RenderCallback f);
    void register_mouse_pos_callback(MousePosCallback f);
    void register_mouse_button_callback(MouseButtonCallback f);
    void register_keyboard_callback(KeyboardCallback f);
    void register_window_resize_callback(ResizeCallback f);

    // Request to close the window and blocks until the ui thread exits
    void close();

    bool is_window_open() const;

    WindowDims get_window_dims() const;

   private:
    WorkQueue<std::function<void(GLFWwindow *)>> queue_;
    std::thread ui_thread_;
    std::unique_ptr<WindowData> window_data_;
};
}  // namespace robot::visualization::gl_window
