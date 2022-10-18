
#include "visualization/gl_window/gl_window.hh"

#include <condition_variable>
#include <iostream>
#include <optional>
#include <semaphore>

namespace robot::visualization::gl_window {
struct Callbacks {
    GlWindow::RenderCallback render;
    GlWindow::MousePosCallback mouse_pos;
    GlWindow::MouseButtonCallback mouse_button;
    GlWindow::KeyboardCallback keyboard;
    GlWindow::ResizeCallback resize;
};
struct WindowData {
    std::mutex mutex;
    Callbacks callbacks;
    std::condition_variable window_dims_cv;
    std::optional<WindowDims> window_dims;
    bool is_open;
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

void mouse_button_callback(GLFWwindow *window, const int button, const int action, const int mods) {
    WindowData *data = reinterpret_cast<WindowData *>(glfwGetWindowUserPointer(window));
    if (data) {
        std::lock_guard<std::mutex> guard(data->mutex);
        maybe_call(data->callbacks.mouse_button, button, action, mods);
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

void resize_callback(GLFWwindow *window, const int width, const int height) {
    WindowData *data = reinterpret_cast<WindowData *>(glfwGetWindowUserPointer(window));
    if (data) {
        std::lock_guard<std::mutex> guard(data->mutex);
        WindowDims dims{
            .width = width,
            .height = height,
        };
        data->window_dims = dims;
        data->window_dims_cv.notify_all();
        maybe_call(data->callbacks.resize, width, height);
    }
}

void register_callbacks(GLFWwindow *window) {
    glfwSetCursorPosCallback(window, mouse_pos_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetKeyCallback(window, keyboard_callback);
    glfwSetFramebufferSizeCallback(window, resize_callback);
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

    // Set the is open flag to false
    WindowData *data = reinterpret_cast<WindowData *>(glfwGetWindowUserPointer(window));
    if (data) {
        std::lock_guard<std::mutex> guard(data->mutex);
        data->is_open = false;
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
    queue_.submit_work([this](GLFWwindow *) {
        std::lock_guard<std::mutex> guard(window_data_->mutex);
        window_data_->is_open = true;
    });
    queue_.submit_work([this](GLFWwindow *window) {
        std::lock_guard<std::mutex> guard(window_data_->mutex);
        WindowDims dims;
        glfwGetFramebufferSize(window, &dims.width, &dims.height);
        window_data_->window_dims = dims;
        window_data_->window_dims_cv.notify_all();
    });
}

GlWindow::~GlWindow() { close(); }

void GlWindow::register_mouse_pos_callback(GlWindow::MousePosCallback f) {
    std::lock_guard<std::mutex> guard(window_data_->mutex);
    window_data_->callbacks.mouse_pos = std::move(f);
}

void GlWindow::register_mouse_button_callback(GlWindow::MouseButtonCallback f) {
    std::lock_guard<std::mutex> guard(window_data_->mutex);
    window_data_->callbacks.mouse_button = std::move(f);
}

void GlWindow::register_keyboard_callback(GlWindow::KeyboardCallback f) {
    std::lock_guard<std::mutex> guard(window_data_->mutex);
    window_data_->callbacks.keyboard = std::move(f);
}

void GlWindow::register_render_callback(GlWindow::RenderCallback f) {
    std::lock_guard<std::mutex> guard(window_data_->mutex);
    window_data_->callbacks.render = std::move(f);
}

void GlWindow::register_window_resize_callback(GlWindow::ResizeCallback f) {
    std::lock_guard<std::mutex> guard(window_data_->mutex);
    window_data_->callbacks.resize = std::move(f);
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

bool GlWindow::is_window_open() const {
    std::lock_guard<std::mutex> guard(window_data_->mutex);
    return window_data_->is_open;
}

void GlWindow::close() {
    queue_.submit_work([](GLFWwindow *window) { glfwSetWindowShouldClose(window, true); });
    ui_thread_.join();
}

WindowDims GlWindow::get_window_dims() const {
    std::unique_lock<std::mutex> lock(window_data_->mutex);
    window_data_->window_dims_cv.wait(lock,
                                      [&]() { return window_data_->window_dims.has_value(); });
    return window_data_->window_dims.value();
}

}  // namespace robot::visualization::gl_window
