#include "blur.h"

#include <libbase/runtime_assert.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace {

inline int clampi(int v, int lo, int hi) noexcept {
    return std::max(lo, std::min(hi, v));
}

struct Kernel1D {
    std::vector<float> w;
    int r = 0;
};

inline Kernel1D makeGaussianKernel(float sigma) {
    Kernel1D k;
    if (!(sigma > 0.0f)) return k;

    const float s = std::max(0.001f, sigma);
    k.r = std::max(0, static_cast<int>(std::ceil(3.0f * s)));
    const int R = k.r;
    if (R == 0) return k;

    k.w.assign(static_cast<size_t>(2 * R + 1), 0.0f);

    const float inv2s2 = 1.0f / (2.0f * s * s);

    float sum = 0.0f;
    for (int i = 0; i <= R; ++i) {
        const float v = std::exp(-(float)(i * i) * inv2s2);
        k.w[static_cast<size_t>(R + i)] = v;
        k.w[static_cast<size_t>(R - i)] = v;
        sum += (i == 0) ? v : (2.0f * v);
    }

    const float invSum = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
    for (float& v : k.w) v *= invSum;

    return k;
}

template <typename T>
inline float to_f(T v) noexcept {
    if constexpr (std::is_same_v<T, float>) return v;
    return static_cast<float>(v);
}

template <typename T>
inline T from_f(float v) noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
        v = std::clamp(v, 0.0f, 255.0f);
        return static_cast<std::uint8_t>(std::lround(v));
    } else if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(std::lround(v));
    } else {
        return static_cast<T>(v);
    }
}

// --------------------- Image blur: 1 channel ---------------------

template <typename T>
Image<T> blur_gray(const Image<T>& image, const Kernel1D& k) {
    const int W = image.width();
    const int H = image.height();
    const int R = k.r;
    const float* kw = k.w.data();

    std::vector<float> tmp(static_cast<size_t>(W) * static_cast<size_t>(H), 0.0f);

    #pragma omp parallel for
    for (int y = 0; y < H; ++y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(W);

        const int leftEnd = std::min(R, W);
        for (int x = 0; x < leftEnd; ++x) {
            float acc = 0.0f;
            for (int dx = -R; dx <= R; ++dx) {
                const int sx = clampi(x + dx, 0, W - 1);
                acc += kw[dx + R] * to_f(image(y, sx));
            }
            tmp[row + static_cast<size_t>(x)] = acc;
        }

        const int midBegin = R;
        const int midEnd = W - R;
        if (midBegin < midEnd) {
            for (int x = midBegin; x < midEnd; ++x) {
                float acc = 0.0f;
                for (int dx = -R; dx <= R; ++dx) {
                    acc += kw[dx + R] * to_f(image(y, x + dx));
                }
                tmp[row + static_cast<size_t>(x)] = acc;
            }
        }

        const int rightBegin = std::max(leftEnd, midEnd);
        for (int x = rightBegin; x < W; ++x) {
            float acc = 0.0f;
            for (int dx = -R; dx <= R; ++dx) {
                const int sx = clampi(x + dx, 0, W - 1);
                acc += kw[dx + R] * to_f(image(y, sx));
            }
            tmp[row + static_cast<size_t>(x)] = acc;
        }
    }

    Image<T> out(W, H, 1);

    #pragma omp parallel for
    for (int y = 0; y < H; ++y) {
        const bool midY = (y >= R && y < H - R);

        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;

            if (midY) {
                for (int dy = -R; dy <= R; ++dy) {
                    const int sy = y + dy;
                    acc += kw[dy + R] * tmp[static_cast<size_t>(sy) * static_cast<size_t>(W) + static_cast<size_t>(x)];
                }
            } else {
                for (int dy = -R; dy <= R; ++dy) {
                    const int sy = clampi(y + dy, 0, H - 1);
                    acc += kw[dy + R] * tmp[static_cast<size_t>(sy) * static_cast<size_t>(W) + static_cast<size_t>(x)];
                }
            }

            out(y, x) = from_f<T>(acc);
        }
    }

    return out;
}

// --------------------- Image blur: 3 channels ---------------------

template <typename T>
Image<T> blur_rgb(const Image<T>& image, const Kernel1D& k) {
    const int W = image.width();
    const int H = image.height();
    const int R = k.r;
    const float* kw = k.w.data();

    std::vector<float> tmp(static_cast<size_t>(W) * static_cast<size_t>(H) * 3u, 0.0f);

    auto idx3 = [&](int x, int y) -> size_t {
        return (static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x)) * 3u;
    };

    #pragma omp parallel for
    for (int y = 0; y < H; ++y) {
        const int leftEnd = std::min(R, W);
        for (int x = 0; x < leftEnd; ++x) {
            float a0 = 0, a1 = 0, a2 = 0;
            for (int dx = -R; dx <= R; ++dx) {
                const int sx = clampi(x + dx, 0, W - 1);
                const float w = kw[dx + R];
                a0 += w * to_f(image(y, sx, 0));
                a1 += w * to_f(image(y, sx, 1));
                a2 += w * to_f(image(y, sx, 2));
            }
            const size_t o = idx3(x, y);
            tmp[o + 0] = a0; tmp[o + 1] = a1; tmp[o + 2] = a2;
        }

        const int midBegin = R;
        const int midEnd = W - R;
        if (midBegin < midEnd) {
            for (int x = midBegin; x < midEnd; ++x) {
                float a0 = 0, a1 = 0, a2 = 0;
                for (int dx = -R; dx <= R; ++dx) {
                    const int sx = x + dx;
                    const float w = kw[dx + R];
                    a0 += w * to_f(image(y, sx, 0));
                    a1 += w * to_f(image(y, sx, 1));
                    a2 += w * to_f(image(y, sx, 2));
                }
                const size_t o = idx3(x, y);
                tmp[o + 0] = a0; tmp[o + 1] = a1; tmp[o + 2] = a2;
            }
        }

        const int rightBegin = std::max(leftEnd, midEnd);
        for (int x = rightBegin; x < W; ++x) {
            float a0 = 0, a1 = 0, a2 = 0;
            for (int dx = -R; dx <= R; ++dx) {
                const int sx = clampi(x + dx, 0, W - 1);
                const float w = kw[dx + R];
                a0 += w * to_f(image(y, sx, 0));
                a1 += w * to_f(image(y, sx, 1));
                a2 += w * to_f(image(y, sx, 2));
            }
            const size_t o = idx3(x, y);
            tmp[o + 0] = a0; tmp[o + 1] = a1; tmp[o + 2] = a2;
        }
    }

    Image<T> out(W, H, 3);

    #pragma omp parallel for
    for (int y = 0; y < H; ++y) {
        const bool midY = (y >= R && y < H - R);

        for (int x = 0; x < W; ++x) {
            float a0 = 0, a1 = 0, a2 = 0;

            if (midY) {
                for (int dy = -R; dy <= R; ++dy) {
                    const int sy = y + dy;
                    const float w = kw[dy + R];
                    const size_t o = idx3(x, sy);
                    a0 += w * tmp[o + 0];
                    a1 += w * tmp[o + 1];
                    a2 += w * tmp[o + 2];
                }
            } else {
                for (int dy = -R; dy <= R; ++dy) {
                    const int sy = clampi(y + dy, 0, H - 1);
                    const float w = kw[dy + R];
                    const size_t o = idx3(x, sy);
                    a0 += w * tmp[o + 0];
                    a1 += w * tmp[o + 1];
                    a2 += w * tmp[o + 2];
                }
            }

            out(y, x, 0) = from_f<T>(a0);
            out(y, x, 1) = from_f<T>(a1);
            out(y, x, 2) = from_f<T>(a2);
        }
    }

    return out;
}

} // namespace

template <typename T>
Image<T> blur(const Image<T> &image, float strength) {
    if (!(strength > 0.0f)) return image;

    const int W = image.width();
    const int H = image.height();
    const int C = image.channels();
    rassert(W > 0 && H > 0, 981234001);
    rassert(C == 1 || C == 3, 981234002, C);

    const Kernel1D k = makeGaussianKernel(strength);
    if (k.r == 0) return image;

    return (C == 1) ? blur_gray(image, k) : blur_rgb(image, k);
}

template <typename T>
std::vector<Color<T>> blur(const std::vector<Color<T>> &colors, float strength) {
    if (!(strength > 0.0f)) return colors;
    if (colors.empty()) return {};

    const Kernel1D k = makeGaussianKernel(strength);
    if (k.r == 0) return colors;

    const int R = k.r;
    const float* kw = k.w.data();
    const int n = static_cast<int>(colors.size());
    const int C = colors[0].channels();
    rassert(C == 1 || C == 3, 981234003, C);

    // flat tmp: [c*n + i]
    std::vector<float> tmp(static_cast<size_t>(n) * static_cast<size_t>(C), 0.0f);

    auto t = [&](int c, int i) -> float& {
        return tmp[static_cast<size_t>(c) * static_cast<size_t>(n) + static_cast<size_t>(i)];
    };

    for (int i = 0; i < n; ++i) {
        const bool mid = (i >= R && i < n - R);

        if (C == 1) {
            float acc = 0.0f;
            if (mid) {
                for (int d = -R; d <= R; ++d) {
                    const float w = kw[d + R];
                    const auto& col = colors[i + d];
                    acc += w * to_f(col(0));
                }
            } else {
                for (int d = -R; d <= R; ++d) {
                    const int si = clampi(i + d, 0, n - 1);
                    const float w = kw[d + R];
                    const auto& col = colors[si];
                    acc += w * to_f(col(0));
                }
            }
            t(0, i) = acc;
        } else {
            float a0 = 0, a1 = 0, a2 = 0;
            if (mid) {
                for (int d = -R; d <= R; ++d) {
                    const float w = kw[d + R];
                    const auto& col = colors[static_cast<size_t>(i + d)];
                    a0 += w * to_f(col(0));
                    a1 += w * to_f(col(1));
                    a2 += w * to_f(col(2));
                }
            } else {
                for (int d = -R; d <= R; ++d) {
                    const int si = clampi(i + d, 0, n - 1);
                    const float w = kw[d + R];
                    const auto& col = colors[static_cast<size_t>(si)];
                    a0 += w * to_f(col(0));
                    a1 += w * to_f(col(1));
                    a2 += w * to_f(col(2));
                }
            }
            t(0, i) = a0;
            t(1, i) = a1;
            t(2, i) = a2;
        }
    }

    std::vector<Color<T>> out;
    out.reserve(static_cast<size_t>(n));

    if (C == 1) {
        for (int i = 0; i < n; ++i) {
            out.emplace_back(from_f<T>(t(0, i)));
        }
    } else {
        for (int i = 0; i < n; ++i) {
            out.emplace_back(
                from_f<T>(t(0, i)),
                from_f<T>(t(1, i)),
                from_f<T>(t(2, i))
            );
        }
    }

    return out;
}

// explicit instantiations
template Image<std::uint8_t> blur(const Image<std::uint8_t>& image, float strength);
template Image<float>        blur(const Image<float>& image, float strength);

template std::vector<Color<std::uint8_t>> blur(const std::vector<Color<std::uint8_t>>& colors, float strength);
template std::vector<Color<float>>        blur(const std::vector<Color<float>>& colors, float strength);
