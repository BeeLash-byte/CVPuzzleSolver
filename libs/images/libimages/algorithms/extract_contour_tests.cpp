#include "extract_contour.h"

#include <gtest/gtest.h>

#include <libbase/configure_working_directory.h>
#include <libimages/debug_io.h>
#include <libimages/tests_utils.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr unsigned char kFg = 255;

template <typename T>
static void fillRect(Image<T>& img, point2i from, point2i to, T v) {
    for (int y = from.y; y < to.y; ++y) {
        for (int x = from.x; x < to.x; ++x) {
            img(y, x) = v;
        }
    }
}

static int count255(const image8u& m) {
    int cnt = 0;
    for (int y = 0; y < m.height(); ++y) {
        for (int x = 0; x < m.width(); ++x) {
            if (m(y, x) == kFg) ++cnt;
        }
    }
    return cnt;
}

static long long signedArea2_imageCoords(const std::vector<point2i>& poly) {
    if (poly.size() < 3) return 0;
    long long a2 = 0;
    const int n = static_cast<int>(poly.size());
    for (int i = 0; i < n; ++i) {
        const auto& p = poly[i];
        const auto& q = poly[(i + 1) % n];
        a2 += static_cast<long long>(p.x) * static_cast<long long>(q.y)
            - static_cast<long long>(q.x) * static_cast<long long>(p.y);
    }
    // For image coords (y down): a2 > 0 => clockwise
    return a2;
}

static image8u visualizeContourTrace(int w, int h, const std::vector<point2i>& contour) {
    image8u vis(w, h, 1);
    vis.fill(0);

    const int n = static_cast<int>(contour.size());
    if (n == 0) return vis;

    // Draw gradient along traversal (helps see order / direction).
    for (int i = 0; i < n; ++i) {
        const auto& p = contour[i];
        if (p.x < 0 || p.x >= w || p.y < 0 || p.y >= h) continue;

        int v = 64;
        if (n > 1) {
            v = 64 + (i * (255 - 64)) / (n - 1);
        }
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        vis(p.y, p.x) = static_cast<unsigned char>(v);
    }

    // Mark start pixel bright.
    const auto& s = contour.front();
    if (s.x >= 0 && s.x < w && s.y >= 0 && s.y < h) {
        vis(s.y, s.x) = 255;
    }
    return vis;
}

} // namespace

TEST(extract_contour, buildContourMask_rectangle) {
    configureWorkingDirectory();

    image8u obj(10, 10, 1);
    obj.fill(0);

    // Filled rectangle: x in [2,7), y in [3,8) => w=5,h=5 => perimeter pixels = 2*(5+5)-4 = 16
    fillRect(obj, point2i{2, 3}, point2i{7, 8}, static_cast<unsigned char>(255));
    debug_io::dump_image(getUnitCaseDebugDir() + "00_object_mask.jpg", obj);

    image8u contour = buildContourMask(obj);
    debug_io::dump_image(getUnitCaseDebugDir() + "01_contour_mask.jpg", contour);

    EXPECT_EQ(count255(contour), 16);
    EXPECT_EQ(contour(5, 4), 0);          // interior
    EXPECT_EQ(contour(3, 2), 255);        // top-left corner
    EXPECT_EQ(contour(7, 6), 255);        // bottom edge
}

TEST(extract_contour, extractContour_rectangle_clockwise_and_adjacent) {
    configureWorkingDirectory();

    image8u obj(10, 10, 1);
    obj.fill(0);
    fillRect(obj, point2i{2, 3}, point2i{7, 8}, static_cast<unsigned char>(255));
    debug_io::dump_image(getUnitCaseDebugDir() + "00_object_mask.jpg", obj);

    image8u contourMask = buildContourMask(obj);
    debug_io::dump_image(getUnitCaseDebugDir() + "01_contour_mask.jpg", contourMask);

    auto contour = extractContour(contourMask);

    // Visualization of traversal order
    image8u trace = visualizeContourTrace(contourMask.width(), contourMask.height(), contour);
    debug_io::dump_image(getUnitCaseDebugDir() + "02_contour_trace.jpg", trace);

    ASSERT_EQ(static_cast<int>(contour.size()), 16);

    // Every point must be a contour pixel.
    for (const auto& p : contour) {
        ASSERT_GE(p.x, 0);
        ASSERT_LT(p.x, contourMask.width());
        ASSERT_GE(p.y, 0);
        ASSERT_LT(p.y, contourMask.height());
        EXPECT_EQ(contourMask(p.y, p.x), 255);
    }

    // Adjacency (8-neighborhood) for consecutive points, including wrap-around.
    const int n = static_cast<int>(contour.size());
    for (int i = 0; i < n; ++i) {
        const auto& a = contour[i];
        const auto& b = contour[(i + 1) % n];
        const int dx = std::abs(a.x - b.x);
        const int dy = std::abs(a.y - b.y);
        EXPECT_LE(dx, 1);
        EXPECT_LE(dy, 1);
        EXPECT_TRUE(dx != 0 || dy != 0);
    }

    // Clockwise in image coords (y down) => signed area > 0
    EXPECT_GT(signedArea2_imageCoords(contour), 0);
}

TEST(extract_contour, singlePixel) {
    configureWorkingDirectory();

    image8u obj(7, 7, 1);
    obj.fill(0);
    obj(3, 4) = 255;

    debug_io::dump_image(getUnitCaseDebugDir() + "00_object_mask.jpg", obj);

    image8u contourMask = buildContourMask(obj);
    debug_io::dump_image(getUnitCaseDebugDir() + "01_contour_mask.jpg", contourMask);

    auto contour = extractContour(contourMask);

    image8u trace = visualizeContourTrace(contourMask.width(), contourMask.height(), contour);
    debug_io::dump_image(getUnitCaseDebugDir() + "02_contour_trace.jpg", trace);

    EXPECT_EQ(count255(contourMask), 1);

    ASSERT_EQ(contour.size(), 1u);
    EXPECT_EQ(contour[0], (point2i{4, 3}));
}
