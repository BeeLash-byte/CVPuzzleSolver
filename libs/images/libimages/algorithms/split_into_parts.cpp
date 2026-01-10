#include "split_into_parts.h"

#include <libbase/disjoint_set.h>
#include <libbase/bbox2.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <tuple>
#include <vector>

#include <libbase/runtime_assert.h>

namespace {

constexpr unsigned char kObject = 255;

inline std::size_t linearIndex(int x, int y, int w) noexcept {
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x);
}

} // namespace

std::tuple<std::vector<point2i>, std::vector<image32f>, std::vector<image8u>> splitObjects(
    const image32f &image, const image8u &objectsMask)
{
    rassert(image.width() == objectsMask.width(), 980123741);
    rassert(image.height() == objectsMask.height(), 980123742);

    const int w = image.width();
    const int h = image.height();

    const std::size_t n = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    DisjointSetUnion dsu(n);

    // Build DSU for object pixels (8-connectivity).
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (objectsMask(y, x) != kObject) continue;

            const std::size_t id = linearIndex(x, y, w);

            // Left
            if (x > 0 && objectsMask(y, x - 1) == kObject) {
                dsu.unite(id, linearIndex(x - 1, y, w));
            }
            // Up
            if (y > 0 && objectsMask(y - 1, x) == kObject) {
                dsu.unite(id, linearIndex(x, y - 1, w));
            }
            // Up-left
            if (x > 0 && y > 0 && objectsMask(y - 1, x - 1) == kObject) {
                dsu.unite(id, linearIndex(x - 1, y - 1, w));
            }
            // Up-right
            if (x + 1 < w && y > 0 && objectsMask(y - 1, x + 1) == kObject) {
                dsu.unite(id, linearIndex(x + 1, y - 1, w));
            }
        }
    }

    // Compute bbox per component root and remember root for each object pixel.
    std::vector<bbox2i> boxes(n, bbox2i::make_empty());
    std::vector<std::size_t> rootOfPixel(n, static_cast<std::size_t>(-1));

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (objectsMask(y, x) != kObject) continue;

            const std::size_t id = linearIndex(x, y, w);
            const std::size_t r = dsu.find(id);
            rootOfPixel[id] = r;
            boxes[r].include_pixel(x, y);
        }
    }

    // Collect roots.
    std::vector<std::size_t> roots;
    roots.reserve(128);
    for (std::size_t r = 0; r < n; ++r) {
        if (!boxes[r].is_empty()) roots.push_back(r);
    }

    // Deterministic order: by bbox top-left (y, then x).
    std::sort(roots.begin(), roots.end(), [&](std::size_t a, std::size_t b) {
        const auto &A = boxes[a];
        const auto &B = boxes[b];
        if (A.min.y != B.min.y) return A.min.y < B.min.y;
        return A.min.x < B.min.x;
    });

    std::vector<point2i> offsets;
    std::vector<image32f> partsImages;
    std::vector<image8u> partsMasks;

    offsets.reserve(roots.size());
    partsImages.reserve(roots.size());
    partsMasks.reserve(roots.size());

    // Extract crops.
    for (std::size_t r : roots) {
        const bbox2i &bb = boxes[r];
        const int outW = bb.width();
        const int outH = bb.height();

        point2i offset = bb.min;
        offsets.push_back(offset);

        image32f partImage(outW, outH, 1);
        image8u partMask(outW, outH, 1);

        for (int yy = 0; yy < outH; ++yy) {
            const int srcY = offset.y + yy;
            for (int xx = 0; xx < outW; ++xx) {
                const int srcX = offset.x + xx;

                partImage(yy, xx) = image(srcY, srcX);

                const std::size_t sid = linearIndex(srcX, srcY, w);
                const bool belongs = (objectsMask(srcY, srcX) == kObject) && (rootOfPixel[sid] == r);
                partMask(yy, xx) = belongs ? kObject : 0;
            }
        }

        partsImages.push_back(std::move(partImage));
        partsMasks.push_back(std::move(partMask));
    }

    return {offsets, partsImages, partsMasks};
}
