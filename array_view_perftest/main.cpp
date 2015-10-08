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

template <typename T>
constexpr T clamp(T x, T a, T b) {
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

void printNormals(const std::vector<Vec3f>& normals) {
    std::cout << normals[0].x << ", " << normals[0].y << ", " << normals[0].z << std::endl;
}

auto timeInS = [](auto f) {
    const auto start = std::chrono::high_resolution_clock::now();
    f();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count() *
           1e-9f;
};

void writeNormals(const std::vector<Vec3f>& normals, const std::string& filename) {
    auto out = std::ofstream{filename, std::ios::out | std::ios::binary};
    out.write(reinterpret_cast<const char*>(normals.data()), normals.size() * sizeof(normals[0]));
}

auto testCalculateNormalsFunc = [](auto func, const Heightfield& heightfield,
                                   std::string funcname) {
    auto normals = std::vector<Vec3f>{};
    const auto funcTime = timeInS([&] { normals = func(heightfield); });
    writeNormals(normals, funcname + ".dat");
    return make_tuple(normals, funcTime);
};

#define TEST_CALCULATE_NORMALS_FUNC(f) testCalculateNormalsFunc(f, heightfield, #f)

int main() {
    const auto heightfield = loadHeightfield();

    auto gslResults = TEST_CALCULATE_NORMALS_FUNC(calculateHeightfieldNormalsGslArrayView);
    auto cArrayResults = TEST_CALCULATE_NORMALS_FUNC(calculateHeightfieldNormalsCArray);
    auto stdVectorResults = TEST_CALCULATE_NORMALS_FUNC(calculateHeightfieldNormalsStdVector);

    assert(get<0>(gslResults) == get<0>(cArrayResults));
    assert(get<0>(gslResults) == get<0>(stdVectorResults));

    printHeightfield(heightfield);
    printNormals(std::get<0>(gslResults));

    std::cout << "calculateHeightfieldNormalsGslArrayView took " << std::get<1>(gslResults) << "s\n"
              << "calculateHeightfieldNormalsCArray took " << std::get<1>(cArrayResults) << "s\n"
              << "calculateHeightfieldNormalsStdVector took " << std::get<1>(stdVectorResults)
              << "s\n" << std::endl;
}
