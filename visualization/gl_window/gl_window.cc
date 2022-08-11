
#include "visualization/gl_window/gl_window.hh"

#include <iostream>

namespace visualization::gl_window {
struct Callbacks {
    GlWindow::RenderCallback render;
    GlWindow::MousePosCallback mouse_pos;
};
struct WindowData {
    std::mutex mutex;
    Callbacks callbacks;
};

namespace {
template <typename T, typename... Args>
void maybe_call(const T &f, Args... args) {
    if (f) {
        f(std::forward<Args>(args)...);
    }
}

void mouse_pos_callback(GLFWwindow *window, double x, double y) {
    WindowData *data = reinterpret_cast<WindowData *>(glfwGetWindowUserPointer(window));
    if (data) {
        std::lock_guard<std::mutex> guard(data->mutex);
        maybe_call(data->callbacks.mouse_pos, x, y);
    }
}

void register_callbacks(GLFWwindow *window) {
    glfwSetCursorPosCallback(window, mouse_pos_callback);
}

void window_loop(const int width, const int height, const std::string &title,
                 WorkQueue<std::function<void(GLFWwindow *)>> &queue) {
    if (glfwInit() != GLFW_TRUE) {
        glfwTerminate();
        return;
    }

    glfwWindowHint(GLFW_FLOATING, true);
    glfwWindowHint(GLFW_RESIZABLE, false);

    auto *window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    register_callbacks(window);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glEnable(GL_DEPTH_TEST | GL_LINE_SMOOTH);
    glClearColor(0.15, 0.15, 0.15, 1.0);

    while (!glfwWindowShouldClose(window)) {
        while (!queue.empty()) {
            queue.complete_work(window);
        }

        WindowData *data = reinterpret_cast<WindowData *>(glfwGetWindowUserPointer(window));
        if (data) {
          std::lock_guard<std::mutex> guard(data->mutex);
          maybe_call(data->callbacks.render);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
}
}  // namespace

GlWindow::GlWindow(const int width, const int height, const std::string &title)
    : queue_{},
      ui_thread_(window_loop, width, height, title, std::ref(queue_)),
      window_data_{std::make_unique<WindowData>()} {
    queue_.submit_work(
        [this](GLFWwindow *window) { glfwSetWindowUserPointer(window, window_data_.get()); });
}

GlWindow::~GlWindow() { close(); }

void GlWindow::register_mouse_pos_callback(GlWindow::MousePosCallback f) {
    std::lock_guard<std::mutex> guard(window_data_->mutex);
    window_data_->callbacks.mouse_pos = std::move(f);
}

void GlWindow::register_render_callback(GlWindow::RenderCallback f) {
    std::lock_guard<std::mutex> guard(window_data_->mutex);
    window_data_->callbacks.render = std::move(f);
}

void GlWindow::close() {
    queue_.submit_work([](GLFWwindow *window) { glfwSetWindowShouldClose(window, true); });
    ui_thread_.join();
}
}  // namespace visualization::gl_window
