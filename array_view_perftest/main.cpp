#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 4245)
#include <GSL/include/array_view.h>
#pragma warning(pop)
#include <ParallelSTL/include/experimental/array_view>

#pragma once

#include <cmath>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

struct Vec3f {
    float x, y, z;
};

auto operator==(const Vec3f& x, const Vec3f& y) {
    return std::tie(x.x, x.y, x.z) == std::tie(y.x, y.y, y.z);
}

constexpr auto operator*(const Vec3f& a, float s) {
    return Vec3f{a.x * s, a.y * s, a.z * s};
}

constexpr auto dot(const Vec3f& x, const Vec3f& y) {
    return x.x * y.x + x.y * y.y + x.z * y.z;
}

inline auto magnitude(const Vec3f& x) {
    return sqrt(dot(x, x));
}

inline auto normalize(const Vec3f& x) {
    return x * (1.0f / magnitude(x));
}

template <typename T, typename U, typename V>
constexpr auto clamp(T x, U a, V b) {
    return x < a ? a : x > b ? b : x;
}

struct Heightfield {
    Heightfield(int width_, int height_, float widthM_, float heightM_,
                std::vector<uint16_t> heights_)
        : width{width_},
          height{height_},
          widthM{widthM_},
          heightM{heightM_},
          heights{std::move(heights_)} {}
    int width = 0, height = 0;
    float widthM = 0, heightM = 0;
    std::vector<uint16_t> heights;

    float gridStepX() const { return widthM / width; }
    float gridStepY() const { return heightM / height; }
};

auto loadHeightfield() {
    auto data = std::ifstream{"heightfield.dat", std::ios::in | std::ios::binary};
    if (!data) std::exit(-1);
    auto readInt32 = [&data] {
        std::int32_t x{};
        data.read(reinterpret_cast<char*>(&x), sizeof(x));
        return x;
    };
    auto readFloat = [&data] {
        float x{};
        data.read(reinterpret_cast<char*>(&x), sizeof(x));
        return x;
    };
    auto readVector = [&data](int size) {
        std::vector<uint16_t> x(size);
        data.read(reinterpret_cast<char*>(x.data()), x.size() * sizeof(x[0]));
        return x;
    };
    const auto width = readInt32();
    const auto height = readInt32();
    const auto widthM = readFloat();
    const auto heightM = readFloat();
    return Heightfield{width, height, widthM, heightM, readVector(width * height)};
}

auto calculateHeightfieldNormalsGslArrayView(const Heightfield& heightfield) {
    auto heightsView = gsl::as_array_view(
        heightfield.heights.data(), gsl::dim<>(heightfield.height), gsl::dim<>(heightfield.width));
    auto getHeight = [heightsView,
                      w = heightsView.bounds().index_bounds()[1],
                      h = heightsView.bounds().index_bounds()[0]](auto idx, int xOff, int yOff) {
        idx[0] = clamp(int(idx[0]) + yOff, 0, int(h) - 1);
        idx[1] = clamp(int(idx[1]) + xOff, 0, int(w) - 1);
        return heightsView[idx];
    };

    const auto gridStepX = heightfield.gridStepX();
    const auto gridStepY = heightfield.gridStepY();
    std::vector<Vec3f> normals(heightfield.heights.size());
    auto normalsView = gsl::as_array_view(normals.data(), gsl::dim<>(heightfield.height),
                                          gsl::dim<>(heightfield.width));
    for (auto idx : normalsView.bounds()) {
        const auto yl = getHeight(idx, -1, 0);
        const auto yr = getHeight(idx, 1, 0);
        const auto yd = getHeight(idx, 0, -1);
        const auto yu = getHeight(idx, 0, 1);
        const auto normal =
            normalize(Vec3f{2.0f * gridStepY * (yl - yr), 4.0f * gridStepX * gridStepY,
                            2.0f * gridStepX * (yd - yu)});
        assert(normal.y > 0.0f);
        normalsView[idx] = normal;
    }
    return normals;
}

auto calculateHeightfieldNormalsParallelStlArrayView(const Heightfield& heightfield) {
    namespace pstl = std::experimental::D4087;
    const auto bounds = pstl::bounds<2>{heightfield.height, heightfield.width};
    auto heightsView = pstl::array_view<const uint16_t, 2>{bounds, heightfield.heights};
    auto getHeight = [ heightsView, w = heightsView.bounds()[1], h = heightsView.bounds()[0] ](
        auto idx, pstl::index<2> off) {
        idx += off;
        idx[0] = clamp(idx[0], 0, h - 1);
        idx[1] = clamp(idx[1], 0, w - 1);
        return heightsView[idx];
    };

    const auto gridStepX = heightfield.gridStepX();
    const auto gridStepY = heightfield.gridStepY();
    std::vector<Vec3f> normals(heightfield.heights.size());
    auto normalsView = pstl::array_view<Vec3f, 2>{bounds, normals};
    for (auto idx : normalsView.bounds()) {
        const auto yl = getHeight(idx, {0, -1});
        const auto yr = getHeight(idx, {0, 1});
        const auto yd = getHeight(idx, {-1, 0});
        const auto yu = getHeight(idx, {1, 0});
        const auto normal =
            normalize(Vec3f{2.0f * gridStepY * (yl - yr), 4.0f * gridStepX * gridStepY,
                            2.0f * gridStepX * (yd - yu)});
        assert(normal.y > 0.0f);
        normalsView[idx] = normal;
    }
    return normals;
}

auto calculateHeightfieldNormalsCArray(const Heightfield& heightfield) {
    auto getHeight = [heights = heightfield.heights.data(), 
                      w = heightfield.width, 
                      h = heightfield.height](int x, int y) {
        x = clamp(x, 0, w - 1);
        y = clamp(y, 0, h - 1);
        return heights[y * w + x];
    };

    const auto gridStepX = heightfield.gridStepX();
    const auto gridStepY = heightfield.gridStepY();
    std::vector<Vec3f> normalsVec(heightfield.heights.size());
    auto normals = normalsVec.data();
    for (auto y = 0; y < heightfield.height; ++y) {
        for (auto x = 0; x < heightfield.width; ++x) {
            const auto yl = getHeight(x - 1, y);
            const auto yr = getHeight(x + 1, y);
            const auto yd = getHeight(x, y - 1);
            const auto yu = getHeight(x, y + 1);
            const auto normal =
                normalize(Vec3f{2.0f * gridStepY * (yl - yr), 4.0f * gridStepX * gridStepY,
                                2.0f * gridStepX * (yd - yu)});
            assert(normal.y > 0.0f);
            normals[y * heightfield.width + x] = normal;
        }
    }
    return normalsVec;
}

auto calculateHeightfieldNormalsStdVector(const Heightfield& heightfield) {
    auto getHeight = [heights = heightfield.heights,
        w = heightfield.width,
        h = heightfield.height](int x, int y) {
        x = clamp(x, 0, w - 1);
        y = clamp(y, 0, h - 1);
        return heights[y * w + x];
    };

    const auto gridStepX = heightfield.gridStepX();
    const auto gridStepY = heightfield.gridStepY();
    std::vector<Vec3f> normals(heightfield.heights.size());
    for (auto y = 0; y < heightfield.height; ++y) {
        for (auto x = 0; x < heightfield.width; ++x) {
            const auto yl = getHeight(x - 1, y);
            const auto yr = getHeight(x + 1, y);
            const auto yd = getHeight(x, y - 1);
            const auto yu = getHeight(x, y + 1);
            const auto normal =
                normalize(Vec3f{ 2.0f * gridStepY * (yl - yr), 4.0f * gridStepX * gridStepY,
                    2.0f * gridStepX * (yd - yu) });
            assert(normal.y > 0.0f);
            normals[y * heightfield.width + x] = normal;
        }
    }
    return normals;
}

void printHeightfield(const Heightfield& heightfield) {
    std::cout << heightfield.width << ", " << heightfield.height << ", " << heightfield.widthM
              << ", " << heightfield.heightM << ", " << heightfield.heights[0] << std::endl;
}

auto timeInS = [](auto f) {
    const auto start = std::chrono::high_resolution_clock::now();
    f();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count() *
           1e-9f;
};

auto testCalculateNormalsFunc = [](auto func, const Heightfield& heightfield,
                                   std::string funcname) {
    auto normals = std::vector<Vec3f>{};
    const auto funcTime = timeInS([&] { normals = func(heightfield); });
    // We write results to a file to ensure the optimizer doesn't get too clever and eliminate
    // calculations for unused results.
    auto out = std::ofstream{funcname + ".dat", std::ios::out | std::ios::binary};
    out.write(reinterpret_cast<const char*>(normals.data()), normals.size() * sizeof(normals[0]));
    return make_tuple(normals, funcTime, funcname);
};

int main() {
    const auto heightfield = loadHeightfield();

#define TEST_CALCULATE_NORMALS_FUNC(f) testCalculateNormalsFunc(f, heightfield, #f)
    // Run the C array version once to warm the cache
    auto firstResults = TEST_CALCULATE_NORMALS_FUNC(calculateHeightfieldNormalsCArray);
    auto results = {TEST_CALCULATE_NORMALS_FUNC(calculateHeightfieldNormalsCArray),
                    TEST_CALCULATE_NORMALS_FUNC(calculateHeightfieldNormalsGslArrayView),
                    TEST_CALCULATE_NORMALS_FUNC(calculateHeightfieldNormalsParallelStlArrayView),
                    TEST_CALCULATE_NORMALS_FUNC(calculateHeightfieldNormalsStdVector)};
#undef TEST_CALCULATE_NORMALS_FUNC

    for (const auto& res : results) {
        (void)firstResults;
        assert(std::get<0>(firstResults) == std::get<0>(res));
        std::cout << std::get<2>(res) << ", " << std::get<1>(res) << "s\n";
    }
}
