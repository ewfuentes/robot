
#include "visualization/gl_window/gl_window.hh"

#include <iostream>
#include <semaphore>

namespace visualization::gl_window {
struct Callbacks {
    GlWindow::RenderCallback render;
    GlWindow::MousePosCallback mouse_pos;
    GlWindow::KeyboardCallback keyboard;
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

void mouse_pos_callback(GLFWwindow *window, const double x, const double y) {
    WindowData *data = reinterpret_cast<WindowData *>(glfwGetWindowUserPointer(window));
    if (data) {
        std::lock_guard<std::mutex> guard(data->mutex);
        maybe_call(data->callbacks.mouse_pos, x, y);
    }
}

void keyboard_callback(GLFWwindow *window, const int key, const int scancode, const int action,
                       const int mods) {
    WindowData *data = reinterpret_cast<WindowData *>(glfwGetWindowUserPointer(window));
    if (data) {
        std::lock_guard<std::mutex> guard(data->mutex);
        maybe_call(data->callbacks.keyboard, key, scancode, action, mods);
    }
}

void register_callbacks(GLFWwindow *window) {
    glfwSetCursorPosCallback(window, mouse_pos_callback);
    glfwSetKeyCallback(window, keyboard_callback);
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

void GlWindow::register_keyboard_callback(GlWindow::KeyboardCallback f) {
    std::lock_guard<std::mutex> guard(window_data_->mutex);
    window_data_->callbacks.keyboard = std::move(f);
}

void GlWindow::register_render_callback(GlWindow::RenderCallback f) {
    std::lock_guard<std::mutex> guard(window_data_->mutex);
    window_data_->callbacks.render = std::move(f);
}

std::unordered_map<int, JoystickState> GlWindow::get_joystick_states() {
  std::binary_semaphore sem(false);
  std::unordered_map<int, JoystickState> out;
  queue_.submit_work([&sem, &out](auto) mutable {
    GLFWgamepadstate state;
    for (int id = GLFW_JOYSTICK_1; id <= GLFW_JOYSTICK_LAST; id++) {
      if (glfwGetGamepadState(id, &state) == GLFW_FALSE) {
        continue;
      }

      std::copy(std::begin(state.buttons), std::end(state.buttons), out[id].buttons.begin());
      std::copy(std::begin(state.axes), std::end(state.axes), out[id].axes.begin());
    }
    sem.release();
  });
  sem.acquire();

  return out;
}

void GlWindow::close() {
    queue_.submit_work([](GLFWwindow *window) { glfwSetWindowShouldClose(window, true); });
    ui_thread_.join();
}
}  // namespace visualization::gl_window
