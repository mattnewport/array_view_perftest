#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
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

// Vectors are treated as row vectors for the purposes of matrix multiplication (so to transform a
// Vector v by a Matrix M use v * M rather than M * v)

// Observations on efficient usage of Vector types.
//
// This code is very generic and uses multiple layers of function helpers, it compiles down to
// pretty efficient code in release builds but in debug builds without any inlining it will be
// pretty inefficient. Using /Ob1
// http://msdn.microsoft.com/en-us/library/47238hez.aspx for debug builds in Visual Studio will help
// debug performance a lot.
//
// You probably want to work with vec4fs as local variables and do all / most of your calculations
// with them, keeping vec3fs as mostly a storage format. This is because vec4fs are implemented as
// native SIMD vector types and most simple operations map to a single intrinsic. It is inefficient
// to work with vec3fs as native SIMD types since the compiler can not make intelligent decisions
// about when to keep values in registers. The exception to this is when doing bulk operations on
// arrays of vec3fs in which case for optimum efficiency you would write custom vector code in SoA
// rather than AoS style operating on multiple elements of the array at once.

template <typename... Ts>
inline void eval(Ts...) {}

template <typename T, size_t N>
class Vector {
public:
    static const size_t dimension = N;

    constexpr Vector() = default;

    // Standard constructor taking a sequence of 1 to N objects convertible to T. If you provide
    // less than N arguments, the remaining elements will be default initialized (to 0 for normal
    // numeric types).
    template <typename... Us>
    constexpr Vector(std::enable_if_t<(sizeof...(Us) <= N - 1), T> t, Us... us) : e_{t, us...} {}

    T& e(size_t i) { return e_[i]; }
    constexpr T e(size_t i) const { return e_[i]; }

    T& x() { return e_[0]; }
    constexpr T x() const { return e_[0]; }
    template <size_t M = N>
    std::enable_if_t<(M > 1), T&> y() {
        return e_[1];
    }
    template <size_t M = N>
    constexpr std::enable_if_t<(M > 1), T> y() const {
        return e_[1];
    }
    template <size_t M = N>
    std::enable_if_t<(M > 2), T&> z() {
        return e_[2];
    }
    template <size_t M = N>
    constexpr std::enable_if_t<(M > 2), T> z() const {
        return e_[2];
    }
    template <size_t M = N>
    std::enable_if_t<(M > 3), T&> w() {
        return e_[3];
    }
    template <size_t M = N>
    constexpr std::enable_if_t<(M > 3), T> w() const {
        return e_[3];
    }

    auto begin() { return std::begin(e_); }
    auto end() { return std::end(e_); }
    auto begin() const { return std::cbegin(e_); }
    auto end() const { return std::cend(e_); }
    auto cbegin() const { return std::cbegin(e_); }
    auto cend() const { return std::cend(e_); }

    template <typename U>
    Vector& operator+=(const Vector<U, N>& x) {
        return plusEquals(x, std::make_index_sequence<N>{});
    }

    template <typename U>
    Vector& operator-=(const Vector<U, N>& x) {
        return minusEquals(x, std::make_index_sequence<N>{});
    }

    template <typename U>
    Vector& operator*=(U x) {
        return multiplyEquals(x, std::make_index_sequence<N>{});
    }

    template <typename U, size_t... Is>
    static constexpr auto memberwiseMultiply(const Vector& x, const Vector<U, N>& y,
                                             std::index_sequence<Is...>) {
        return Vector<std::common_type_t<T, U>, N>{x.e_[Is] * y.e_[Is]...};
    }

    template <typename U, size_t... Is>
    static constexpr auto plus(const Vector& x, const Vector<U, N>& y, std::index_sequence<Is...>) {
        return Vector<std::common_type_t<T, U>, N>{x.e_[Is] + y.e_[Is]...};
    }

    template <typename U, size_t... Is>
    static constexpr auto minus(const Vector& x, const Vector<U, N>& y,
                                std::index_sequence<Is...>) {
        return Vector<std::common_type_t<T, U>, N>{x.e_[Is] - y.e_[Is]...};
    }

    template <typename U, size_t... Is>
    static constexpr auto multiply(const Vector& x, U y, std::index_sequence<Is...>) {
        return Vector<std::common_type_t<T, U>, N>{x.e_[Is] * y...};
    }

private:
    T e_[N];

    template <typename U, size_t M>
    friend class Vector;

    template <typename U, size_t... Is>
    Vector& plusEquals(const Vector<U, N>& x, std::index_sequence<Is...>) {
        eval(e_[Is] += x.e_[Is]...);
        return *this;
    }

    template <typename U, size_t... Is>
    Vector& minusEquals(const Vector<U, N>& x, std::index_sequence<Is...>) {
        eval(e_[Is] -= x.e_[Is]...);
        return *this;
    }

    template <typename U, size_t... Is>
    Vector& multiplyEquals(U x, std::index_sequence<Is...>) {
        eval(e_[Is] *= x...);
        return *this;
    }
};

using Vec2f = Vector<float, 2>;
using Vec3f = Vector<float, 3>;
using Vec4f = Vector<float, 4>;

// MNTODO: replace this memberwise / reduce machinery with C++17 fold expressions once available
template <typename F, size_t... Is, typename T, size_t N>
inline auto apply(F f, std::index_sequence<Is...>, const Vector<T, N>& x) {
    using resvec = Vector<decltype(f(x.e(0))), N>;
    return resvec{f(x.e(Is))...};
}

template <typename F, size_t... Is, typename T, typename U, size_t N>
inline auto apply(F f, std::index_sequence<Is...>, const Vector<T, N>& x, const Vector<U, N>& y) {
    using vec = Vector<std::common_type_t<T, U>, N>;
    using resvec = Vector<decltype(f(x.e(0), y.e(0))), N>;
    return resvec{f(x.e(Is), y.e(Is))...};
}

template <typename F, typename T, size_t N>
inline auto memberwise(F f, const Vector<T, N>& x) {
    return apply(f, std::make_index_sequence<N>{}, x);
}

template <typename F, typename T, typename U, size_t N>
inline auto memberwise(F f, const Vector<T, N>& x, const Vector<U, N>& y) {
    return apply(f, std::make_index_sequence<N>{}, x, y);
}

template <typename F, typename T>
inline auto reduce_impl(F, T t) {
    return t;
}

template <typename F, typename T>
inline auto reduce_impl(F f, T x, T y) {
    return f(x, y);
}

template <typename F, typename T, typename... Ts>
inline auto reduce_impl(F f, T x, T y, Ts... ts) {
    return f(x, reduce_impl(f, y, ts...));
}

template <typename F, typename T, size_t N, size_t... Is>
inline T reduce(F f, const Vector<T, N>& v, std::index_sequence<Is...>) {
    return reduce_impl(f, v.e(Is)...);
}

template <typename F, typename T, size_t N>
inline T reduce(F f, const Vector<T, N>& v) {
    return reduce(f, v, std::make_index_sequence<N>{});
}

// Manual specializations of reduce for commonly used + in dot product, remove when we have fold
template <typename T>
inline auto reduce(std::plus<>, const Vector<T, 3>& v) {
    auto vs = v.begin();
    return vs[0] + vs[1] + vs[2];
}

template <typename T, size_t N>
inline bool operator==(const Vector<T, N>& a, const Vector<T, N>& b) {
    return reduce(std::logical_and<>{}, memberwise(std::equal_to<>{}, a, b));
}

template <typename T, size_t N>
inline bool operator<(const Vector<T, N>& a, const Vector<T, N>& b) {
    return std::lexicographical_compare(begin(a), end(a), begin(b), end(b));
}

template <typename T, typename U, size_t N>
constexpr auto operator+(const Vector<T, N>& a, const Vector<U, N>& b) {
    return Vector<T, N>::plus(a, b, std::make_index_sequence<N>{});
}

template <typename T, typename U, size_t N>
constexpr auto operator-(const Vector<T, N>& a, const Vector<U, N>& b) {
    return Vector<T, N>::minus(a, b, std::make_index_sequence<N>{});
}

template <typename T, typename U, size_t N>
constexpr auto operator*(const Vector<T, N>& a, U s) {
    return Vector<T, N>::multiply(a, s, std::make_index_sequence<N>{});
}

template <typename T, typename U, size_t N>
constexpr auto operator*(T s, const Vector<U, N>& a) {
    return a * s;
}

template <typename T, typename U, size_t N>
constexpr auto operator/(const Vector<T, N>& a, U s) {
    return a * (T{1} / s);
}

template <typename T, typename U, size_t N>
constexpr auto dot(const Vector<T, N>& a, const Vector<U, N>& b) {
    return reduce(std::plus<>{},
                  Vector<T, N>::memberwiseMultiply(a, b, std::make_index_sequence<N>{}));
}

template <typename T, size_t N>
constexpr T magnitude(const Vector<T, N>& a) {
    return sqrt(dot(a, a));
}

template <typename T, size_t N>
constexpr auto normalize(const Vector<T, N>& a) {
    return a * (T{1} / magnitude(a));
}

template <typename T>
constexpr Vector<T, 3> cross(const Vector<T, 3>& a, const Vector<T, 3>& b) {
    return {a.y() * b.z() - a.z() * b.y(), a.z() * b.x() - a.x() * b.z(),
            a.x() * b.y() - a.y() * b.x()};
}

template <typename T>
constexpr T clamp(T x, T a, T b) {
    return x < a ? a : (x > b ? b : x);
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
        assert(normal.y() > 0.0f);
        normalsView[idx] = normal;
    }
    return normals;
}

void printHeightfield(const Heightfield& heightfield) {
    std::cout << heightfield.width << ", " << heightfield.height << ", " << heightfield.widthM
              << ", " << heightfield.heightM << ", " << heightfield.heights[0] << std::endl;
}

void printNormals(const std::vector<Vec3f>& normals) {
    std::cout << ", " << normals[0].x() << ", " << normals[0].y() << ", " << normals[0].z()
              << std::endl;
}

int main() {
    const auto heightfield = loadHeightfield();

    const auto normals1 = calculateHeightfieldNormalsGslArrayView(heightfield);

    printHeightfield(heightfield);
    printNormals(normals1);
}
