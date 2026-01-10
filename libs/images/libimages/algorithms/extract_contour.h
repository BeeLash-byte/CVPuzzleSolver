#pragma once

#include <libimages/image.h>
#include <libbase/point2.h>

#include <vector>

// Input: object mask (0 = background, 255 = object).
// Output: contour mask (0 = not contour, 255 = contour pixel).
image8u buildContourMask(const image8u &objectMask);

// Input: contour mask (0 = background, 255 = contour pixel).
// Output: single closed loop of contour pixels in clockwise order (image coords: x right, y down).
std::vector<point2i> extractContour(const image8u &objectContourMask);
