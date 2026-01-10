#include "split_into_parts.h"

#include <gtest/gtest.h>

#include <libbase/stats.h>
#include <libbase/point2.h>
#include <libbase/runtime_assert.h>
#include <libbase/configure_working_directory.h>
#include <libimages/debug_io.h>
#include <libimages/image.h>
#include <libimages/tests_utils.h>

template <typename T>
void drawCross(Image<T> &image, point2i from, point2i to, T value) {
    // from = top-left corner of bounding box (inclusive)
    // to   = bottom-right corner of bounding box (exclusive)
    rassert(from.x >= 0 && from.x < image.width(), 3728197319283);
    rassert(from.y >= 0 && from.y < image.height(), 3728197319284);
    rassert(to.x > 0 && to.x <= image.width(), 372819723413);
    rassert(to.y > 0 && to.y <= image.height(), 372819723414);
    rassert(from.x < to.x && from.y < to.y, 4782197429181);

    point2i center = (from + to) / 2;
    for (int j = from.y; j < to.y; ++j) {
        image(j, center.x) = value;
    }
    for (int i = from.x; i < to.x; ++i) {
        image(center.y, i) = value;
    }
}

TEST(split_into_parts, oneComponent) {
    configureWorkingDirectory();

    int w = 95;
    int h = 105;
    image32f image(w, h, 1);
    image8u objectsMask(w, h, 1);
    unsigned char objectMaskValue = 255;

    point2i aFrom = {10, 20};
    point2i aTo = {30, 60};
    point2i aSize = aTo - aFrom;
    float aColor = 239.0f;
    drawCross(image, aFrom, aTo, aColor);
    drawCross(objectsMask, aFrom, aTo, objectMaskValue);

    debug_io::dump_image(getUnitCaseDebugDir() + "00_input_image.jpg", image);
    debug_io::dump_image(getUnitCaseDebugDir() + "01_input_mask.jpg", objectsMask);

    auto [objectsOffsets, objectsImages, objectsMasks]
    = splitObjects(image, objectsMask);

    // check that we found exactly one object A
    size_t n = objectsImages.size();
    ASSERT_EQ(n, objectsMasks.size());
    ASSERT_EQ(n, objectsOffsets.size());
    ASSERT_EQ(n, 1);

    for (int i = 0; i < n; ++i) {
        debug_io::dump_image(getUnitCaseDebugDir() + "10_object" + std::to_string(i) + "_image.jpg", objectsImages[i]);
        debug_io::dump_image(getUnitCaseDebugDir() + "10_object" + std::to_string(i) + "_mask.png", objectsMasks[i]);
    }

    ASSERT_EQ(objectsOffsets[0], aFrom);

    ASSERT_EQ(objectsImages[0].width(), aSize.x);
    ASSERT_EQ(objectsImages[0].height(), aSize.y);

    ASSERT_EQ(objectsMasks[0].width(), aSize.x);
    ASSERT_EQ(objectsMasks[0].height(), aSize.y);

    ASSERT_EQ(stats::minValue(objectsImages[0].toVector()), 0);
    ASSERT_EQ(stats::maxValue(objectsImages[0].toVector()), aColor);

    ASSERT_EQ(stats::minValue(objectsMasks[0].toVector()), 0);
    ASSERT_EQ(stats::maxValue(objectsMasks[0].toVector()), 255);

    // Заполняем метки для найденных объектов
    image32i labels(w, h, 1);
    for (int i = 0; i < n; ++i) {
        const auto& offset = objectsOffsets[i];
        const auto& mask = objectsMasks[i];

        for (int y = 0; y < mask.height(); ++y) {
            for (int x = 0; x < mask.width(); ++x) {
                if (mask(y, x, 0) == 255) {
                    int origX = offset.x + x;
                    int origY = offset.y + y;
                    labels(origY, origX, 0) = i + 1;  // Компоненты с индексами 1 и 2 (0 - фон)
                }
            }
        }
    }

    debug_io::dump_image(getUnitCaseDebugDir() + "02_result_components.jpg", debug_io::colorize_labels(labels, 0));
}

TEST(split_into_parts, twoComponents) {
    configureWorkingDirectory();

    int w = 105;
    int h = 95;
    image32f image(w, h, 1);
    image8u objectsMask(w, h, 1);
    unsigned char objectMaskValue = 255;

    point2i aFrom = {10, 20};
    point2i aTo = {30, 60};
    point2i aSize = aTo - aFrom;
    float aColor = 239.0f;
    drawCross(image, aFrom, aTo, aColor);
    drawCross(objectsMask, aFrom, aTo, objectMaskValue);

    point2i bFrom = {60, 70};
    point2i bTo = {76, 85};
    point2i bSize = bTo - bFrom;
    float bColor = 123.0f;
    drawCross(image, bFrom, bTo, bColor);
    drawCross(objectsMask, bFrom, bTo, objectMaskValue);

    debug_io::dump_image(getUnitCaseDebugDir() + "00_input_image.jpg", image);
    debug_io::dump_image(getUnitCaseDebugDir() + "01_input_mask.jpg", objectsMask);

    auto [objectsOffsets, objectsImages, objectsMasks]
        = splitObjects(image, objectsMask);

    // check that we found exactly two objects: A and B
    size_t n = objectsImages.size();
    ASSERT_EQ(n, objectsMasks.size());
    ASSERT_EQ(n, objectsOffsets.size());
    ASSERT_EQ(n, 2);

    for (int i = 0; i < n; ++i) {
        debug_io::dump_image(getUnitCaseDebugDir() + "10_object" + std::to_string(i) + "_image.jpg", objectsImages[i]);
        debug_io::dump_image(getUnitCaseDebugDir() + "10_object" + std::to_string(i) + "_mask.png", objectsMasks[i]);
    }

    // if you have different order (B at index 0 and A at index 1) - that's OK
    // just update unit-test - update this two variables:
    int aIndex = 0;
    int bIndex = 1;

    ASSERT_EQ(objectsOffsets[aIndex], aFrom);
    ASSERT_EQ(objectsOffsets[bIndex], bFrom);

    ASSERT_EQ(objectsImages[aIndex].width(), aSize.x);
    ASSERT_EQ(objectsImages[aIndex].height(), aSize.y);
    ASSERT_EQ(objectsImages[bIndex].width(), bSize.x);
    ASSERT_EQ(objectsImages[bIndex].height(), bSize.y);

    ASSERT_EQ(objectsMasks[aIndex].width(), aSize.x);
    ASSERT_EQ(objectsMasks[aIndex].height(), aSize.y);
    ASSERT_EQ(objectsMasks[bIndex].width(), bSize.x);
    ASSERT_EQ(objectsMasks[bIndex].height(), bSize.y);

    ASSERT_EQ(stats::minValue(objectsImages[aIndex].toVector()), 0);
    ASSERT_EQ(stats::maxValue(objectsImages[aIndex].toVector()), aColor);
    ASSERT_EQ(stats::minValue(objectsImages[bIndex].toVector()), 0);
    ASSERT_EQ(stats::maxValue(objectsImages[bIndex].toVector()), bColor);

    ASSERT_EQ(stats::minValue(objectsMasks[aIndex].toVector()), 0);
    ASSERT_EQ(stats::maxValue(objectsMasks[aIndex].toVector()), 255);
    ASSERT_EQ(stats::minValue(objectsMasks[bIndex].toVector()), 0);
    ASSERT_EQ(stats::maxValue(objectsMasks[bIndex].toVector()), 255);

    // Заполняем метки для найденных объектов
    image32i labels(w, h, 1);
    for (int i = 0; i < n; ++i) {
        const auto& offset = objectsOffsets[i];
        const auto& mask = objectsMasks[i];

        for (int y = 0; y < mask.height(); ++y) {
            for (int x = 0; x < mask.width(); ++x) {
                if (mask(y, x, 0) == 255) {
                    int origX = offset.x + x;
                    int origY = offset.y + y;
                    labels(origY, origX, 0) = i + 1;  // Компоненты с индексами 1 и 2 (0 - фон)
                }
            }
        }
    }

    debug_io::dump_image(getUnitCaseDebugDir() + "02_result_components.jpg", debug_io::colorize_labels(labels, 0));
}

