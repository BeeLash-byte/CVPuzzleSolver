#pragma once

#include <libimages/image.h>
#include <libbase/point2.h>


std::tuple<std::vector<point2i>, std::vector<image32f>, std::vector<image8u>> splitObjects(
    const image32f &image, const image8u &objectsMask);
