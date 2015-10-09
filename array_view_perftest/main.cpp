#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 4245)
#include <GSL/include/array_view.h>
#pragma warning(pop)
#include <ParallelSTL/include/experimental/array_view>

struct Vec3f {
    float x, y, z;
};

constexpr auto operator==(const Vec3f& x, const Vec3f& y) {
    return std::tie(x.x, x.y, x.z) == std::tie(y.x, y.y, y.z);
}

constexpr auto operator*(const Vec3f& a, float s) { return Vec3f{a.x * s, a.y * s, a.z * s}; }

constexpr auto dot(const Vec3f& x, const Vec3f& y) { return x.x * y.x + x.y * y.y + x.z * y.z; }

inline auto magnitude(const Vec3f& x) { return sqrt(dot(x, x)); }

inline auto normalize(const Vec3f& x) { return x * (1.0f / magnitude(x)); }

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
    float widthM = 0.0f, heightM = 0.0f;
    std::vector<uint16_t> heights;

    auto gridStepX() const { return widthM / width; }
    auto gridStepY() const { return heightM / height; }
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
        auto xView = gsl::as_array_view(x);
        data.read(reinterpret_cast<char*>(xView.as_writeable_bytes().data()),
                  xView.as_writeable_bytes().size());
        return x;
    };
    const auto width = readInt32(), height = readInt32();
    const auto widthM = readFloat(), heightM = readFloat();
    return Heightfield{width, height, widthM, heightM, readVector(width * height)};
}

auto calculateHeightfieldNormalsGslArrayView(const Heightfield& heightfield) {
    auto heightsView = gsl::as_array_view(
        heightfield.heights.data(), gsl::dim<>(heightfield.height), gsl::dim<>(heightfield.width));
    auto getHeight = [heightsView,
                      w = int(heightsView.bounds().index_bounds()[1]),
                      h = int(heightsView.bounds().index_bounds()[0])](auto idx, int xOff, int yOff) {
        idx[0] = clamp(int(idx[0]) + yOff, 0, h - 1);
        idx[1] = clamp(int(idx[1]) + xOff, 0, w - 1);
        return heightsView[idx];
    };

    const auto gridStepX = heightfield.gridStepX();
    const auto gridStepY = heightfield.gridStepY();
    std::vector<Vec3f> normals(heightfield.heights.size());
    auto normalsView = gsl::as_array_view(normals.data(), gsl::dim<>(heightfield.height),
                                          gsl::dim<>(heightfield.width));
    for (auto idx : normalsView.bounds()) {
        normalsView[idx] =
            normalize(Vec3f{2.0f * gridStepY * (getHeight(idx, -1, 0) - getHeight(idx, 1, 0)),
                            4.0f * gridStepX * gridStepY,
                            2.0f * gridStepX * (getHeight(idx, 0, -1) - getHeight(idx, 0, 1))});
    }
    return normals;
}

auto calculateHeightfieldNormalsParallelStlArrayView(const Heightfield& heightfield) {
    namespace pstl = std::experimental::D4087;
    auto heightsView = pstl::array_view<const uint16_t, 2>{{heightfield.height, heightfield.width},
                                                           heightfield.heights};
    auto getHeight = [heightsView, 
                      w = heightsView.bounds()[1],
                      h = heightsView.bounds()[0]](auto idx, pstl::index<2> off) {
        idx += off;
        idx[0] = clamp(idx[0], 0, h - 1);
        idx[1] = clamp(idx[1], 0, w - 1);
        return heightsView[idx];
    };

    const auto gridStepX = heightfield.gridStepX();
    const auto gridStepY = heightfield.gridStepY();
    std::vector<Vec3f> normals(heightfield.heights.size());
    auto normalsView = pstl::array_view<Vec3f, 2>{heightsView.bounds(), normals};
    for (auto idx : normalsView.bounds()) {
        normalsView[idx] =
            normalize(Vec3f{2.0f * gridStepY * (getHeight(idx, {0, -1}) - getHeight(idx, {0, 1})),
                            4.0f * gridStepX * gridStepY,
                            2.0f * gridStepX * (getHeight(idx, {-1, 0}) - getHeight(idx, {1, 0}))});
    }
    return normals;
}

auto calculateHeightfieldNormalsCArray(const Heightfield& heightfield) {
    auto getHeight = [heights = heightfield.heights.data(), 
                      w = heightfield.width, 
                      h = heightfield.height](int x, int y) {
        return heights[clamp(y, 0, h - 1) * w + clamp(x, 0, w - 1)];
    };

    const auto gridStepX = heightfield.gridStepX();
    const auto gridStepY = heightfield.gridStepY();
    std::vector<Vec3f> normalsVec(heightfield.heights.size());
    auto normals = normalsVec.data();
    for (auto y = 0; y < heightfield.height; ++y) {
        for (auto x = 0; x < heightfield.width; ++x) {
            normals[y * heightfield.width + x] =
                normalize(Vec3f{2.0f * gridStepY * (getHeight(x - 1, y) - getHeight(x + 1, y)),
                                4.0f * gridStepX * gridStepY,
                                2.0f * gridStepX * (getHeight(x, y - 1) - getHeight(x, y + 1))});
        }
    }
    return normalsVec;
}

auto calculateHeightfieldNormalsStdVector(const Heightfield& heightfield) {
    auto getHeight = [&heights = heightfield.heights, 
                      w = heightfield.width, 
                      h = heightfield.height ](int x, int y) {
        return heights[clamp(y, 0, h - 1) * w + clamp(x, 0, w - 1)];
    };

    const auto gridStepX = heightfield.gridStepX();
    const auto gridStepY = heightfield.gridStepY();
    std::vector<Vec3f> normals(heightfield.heights.size());
    for (auto y = 0; y < heightfield.height; ++y) {
        for (auto x = 0; x < heightfield.width; ++x) {
            normals[y * heightfield.width + x] =
                normalize(Vec3f{2.0f * gridStepY * (getHeight(x - 1, y) - getHeight(x + 1, y)),
                                4.0f * gridStepX * gridStepY,
                                2.0f * gridStepX * (getHeight(x, y - 1) - getHeight(x, y + 1))});
        }
    }
    return normals;
}

auto timeInS = [](auto f) {
    using namespace std::chrono;
    const auto start = high_resolution_clock::now();
    f();
    return duration_cast<nanoseconds>(high_resolution_clock::now() - start).count() * 1e-9f;
};

auto testCalculateNormalsFunc = [](auto func, const Heightfield& heightfield,
                                   std::string funcname) {
    auto normals = decltype(func(heightfield)){};
    const auto funcTime = timeInS([&] { normals = func(heightfield); });
    // Write results to a file to ensure the optimizer doesn't get too clever and eliminate
    // calculations for unused results.
    auto normalsView = gsl::as_array_view(normals);
    auto out = std::ofstream{funcname + ".dat", std::ios::out | std::ios::binary};
    out.write(reinterpret_cast<const char*>(normalsView.as_bytes().data()),
              normalsView.as_bytes().size());
    return make_tuple(normals, funcTime, funcname);
};

int main() {
    const auto heightfield = loadHeightfield();

#define TEST_CALCULATE_NORMALS_FUNC(f) testCalculateNormalsFunc(f, heightfield, #f)
    // Run the C array version once to warm the cache
    auto firstResults = TEST_CALCULATE_NORMALS_FUNC(calculateHeightfieldNormalsCArray);
    const auto& results = {
        TEST_CALCULATE_NORMALS_FUNC(calculateHeightfieldNormalsCArray),
        TEST_CALCULATE_NORMALS_FUNC(calculateHeightfieldNormalsGslArrayView),
        TEST_CALCULATE_NORMALS_FUNC(calculateHeightfieldNormalsParallelStlArrayView),
        TEST_CALCULATE_NORMALS_FUNC(calculateHeightfieldNormalsStdVector)};
#undef TEST_CALCULATE_NORMALS_FUNC

    auto resultsFile = std::ofstream{"results.csv", std::ios::app};
    for (const auto& res : results) {
        (void)firstResults;
        assert(std::get<0>(firstResults) == std::get<0>(res));
        resultsFile << CONFIG_NAME ", " << PLATFORM_NAME ", " << std::get<2>(res) << ", "
                    << std::get<1>(res) << "s\n";
    }
}
