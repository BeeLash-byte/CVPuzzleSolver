#include "extract_contour.h"

#include <libbase/runtime_assert.h>

#include <algorithm>
#include <cstddef>
#include <limits>

namespace {

constexpr unsigned char kFg = 255;

inline bool inBounds(int x, int y, int w, int h) noexcept {
    return x >= 0 && x < w && y >= 0 && y < h;
}

inline bool isFg(const image8u& m, int x, int y) noexcept {
    if (!inBounds(x, y, m.width(), m.height())) return false;
    return m(y, x) == kFg;
}

// Clockwise neighbor order in image coordinates (y down):
// 0:E, 1:SE, 2:S, 3:SW, 4:W, 5:NW, 6:N, 7:NE
static constexpr int dx8[8] = {+1, +1,  0, -1, -1, -1,  0, +1};
static constexpr int dy8[8] = { 0, +1, +1, +1,  0, -1, -1, -1};

inline int dirFromDelta(int dx, int dy) {
    // Returns -1 if not an 8-neighbor delta.
    if (dx == +1 && dy ==  0) return 0;
    if (dx == +1 && dy == +1) return 1;
    if (dx ==  0 && dy == +1) return 2;
    if (dx == -1 && dy == +1) return 3;
    if (dx == -1 && dy ==  0) return 4;
    if (dx == -1 && dy == -1) return 5;
    if (dx ==  0 && dy == -1) return 6;
    if (dx == +1 && dy == -1) return 7;
    return -1;
}

inline long long signedArea2_imageCoords(const std::vector<point2i>& poly) {
    if (poly.size() < 3) return 0;
    long long a2 = 0;
    const int n = static_cast<int>(poly.size());
    for (int i = 0; i < n; ++i) {
        const auto& p = poly[i];
        const auto& q = poly[(i + 1) % n];
        a2 += static_cast<long long>(p.x) * static_cast<long long>(q.y)
            - static_cast<long long>(q.x) * static_cast<long long>(p.y);
    }
    // For image coords (y down): a2 > 0 corresponds to clockwise traversal.
    return a2;
}

inline void rotateToMinYX(std::vector<point2i>& pts) {
    if (pts.empty()) return;
    int best = 0;
    for (int i = 1; i < static_cast<int>(pts.size()); ++i) {
        if (pts[i].y < pts[best].y || (pts[i].y == pts[best].y && pts[i].x < pts[best].x)) {
            best = i;
        }
    }
    std::rotate(pts.begin(), pts.begin() + best, pts.end());
}

} // namespace

image8u buildContourMask(const image8u &objectMask) {
    rassert(objectMask.channels() == 1, 918273645);

    const int w = objectMask.width();
    const int h = objectMask.height();

    image8u contour(w, h, 1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (objectMask(y, x) != kFg) continue;

            bool isBoundary = false;
            for (int k = 0; k < 8; ++k) {
                const int nx = x + dx8[k];
                const int ny = y + dy8[k];
                if (!inBounds(nx, ny, w, h) || objectMask(ny, nx) != kFg) {
                    isBoundary = true;
                    break;
                }
            }
            if (isBoundary) contour(y, x) = kFg;
        }
    }

    return contour;
}

std::vector<point2i> extractContour(const image8u &objectContourMask) {
    rassert(objectContourMask.channels() == 1, 918273646);

    const int w = objectContourMask.width();
    const int h = objectContourMask.height();

    // Find start: top-most, then left-most contour pixel.
    point2i start{-1, -1};
    for (int y = 0; y < h && start.x < 0; ++y) {
        for (int x = 0; x < w; ++x) {
            if (objectContourMask(y, x) == kFg) {
                start = {x, y};
                break;
            }
        }
    }
    if (start.x < 0) return {};

    // Degenerate: single pixel contour.
    bool hasNeighbor = false;
    for (int k = 0; k < 8; ++k) {
        if (isFg(objectContourMask, start.x + dx8[k], start.y + dy8[k])) {
            hasNeighbor = true;
            break;
        }
    }
    if (!hasNeighbor) return {start};

    // Moore neighbor tracing (8-connected), using clockwise neighbor order.
    // backtrack starts at west of start (can be out-of-bounds; still treated as direction W).
    point2i p0 = start;
    point2i b = {p0.x - 1, p0.y}; // west
    int dir_b = 4;                // W by construction

    auto step = [&](const point2i& p, const point2i& back) -> std::pair<point2i, point2i> {
        const int db = dirFromDelta(back.x - p.x, back.y - p.y);
        const int dirBack = (db >= 0) ? db : dir_b;

        const int startDir = (dirBack + 1) & 7;
        for (int t = 0; t < 8; ++t) {
            const int d = (startDir + t) & 7;
            const int nx = p.x + dx8[d];
            const int ny = p.y + dy8[d];
            if (isFg(objectContourMask, nx, ny)) {
                // new backtrack is neighbor preceding d in clockwise order
                const int prevd = (d + 7) & 7;
                point2i newBack{p.x + dx8[prevd], p.y + dy8[prevd]};
                return {point2i{nx, ny}, newBack};
            }
        }
        // Should not happen for a valid single contour
        return {p, back};
    };

    std::vector<point2i> contour;
    contour.reserve(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) / 4);

    contour.push_back(p0);

    auto [p1, b1] = step(p0, b);
    if (p1 == p0) return contour;

    contour.push_back(p1);

    point2i cur = p1;
    b = b1;

    const std::size_t safetyLimit = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) + 8;

    while (contour.size() < safetyLimit) {
        auto [nxt, nb] = step(cur, b);

        // Closed the loop: do not append start again.
        if (nxt == p0) break;

        contour.push_back(nxt);
        b = nb;
        cur = nxt;
    }

    // Enforce clockwise orientation in image coords.
    if (signedArea2_imageCoords(contour) < 0) {
        std::reverse(contour.begin(), contour.end());
    }

    // Deterministic start: rotate to min (y, x).
    rotateToMinYX(contour);

    return contour;
}
