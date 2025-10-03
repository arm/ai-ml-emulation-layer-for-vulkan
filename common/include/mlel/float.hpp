/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include <cstddef>
#include <cstdint>
#include <limits>

/*******************************************************************************
 * uint helper classes
 *******************************************************************************/

template <std::size_t SIZE> struct u_int_t {};
template <> struct u_int_t<8> { using type = uint8_t; };
template <> struct u_int_t<16> { using type = uint16_t; };
template <> struct u_int_t<32> { using type = uint32_t; };
template <> struct u_int_t<64> { using type = uint64_t; };

/*******************************************************************************
 * FloatingPoint
 *******************************************************************************/

template <std::size_t EXPONENT, std::size_t MANTISSA> class FloatingPoint;

using float8_e4m3 = FloatingPoint<4, 3>;
using float8_e5m2 = FloatingPoint<5, 2>;
using float8 = float8_e4m3;
using float16 = FloatingPoint<5, 10>;
using float32 = FloatingPoint<8, 23>;
using float64 = FloatingPoint<11, 52>;

/**
 * Floating point conversion between double and the most common floating point types.
 *
 * A normalized floating point number is calculated as:
 *   normalized   = (-1)^sign * 2^(exponent - bias) * (1 + mantissa / 2^mantissa_bits)
 *   denormalized = (-1)^sign * 2^(1 - bias) * mantissa / 2^mantissa_bits
 *
 * The exponent bias is half the exponent range:
 *   bias = 2^(exponent_bits - 1) - 1
 *
 * Table below defines number of bits for each floating point variant.
 *  +------+------+----------+----------|
 *  | Name | Sign | Exponent | Mantissa |
 *  +------+------+----------+----------|
 *  | FP8  |   1  |     4    |    3     |
 *  | FP16 |   1  |     5    |   10     |
 *  | FP32 |   1  |     8    |   23     |
 *  | FP64 |   1  |    11    |   52     |
 *  +------+------+----------+----------|
 *
 * The exponent value defines how the value is calculated:
 * +------+----------+----------+--------------+
 * | Sign | Exponent | Mantissa | Type         |
 * +------+----------+----------+--------------+
 * |   0  |  All '1' |  All '0' | +Infinity    |
 * |   1  |  All '1' |  All '0' | -Infinity    |
 * |   -  |  All '1' |  Any '1' | Not a Number |
 * |   -  |  All '0' |     -    | Denormalized |
 * |   -  |  Any '1' |     -    | Normalized   |
 * +------+----------+----------+--------------+
 *
 * Exponent can be for example be converted between 8- and 32-bit float like below.
 * The smaller exponent can over- or underflow causing the number to become zero or
 * infinity.
 *   2^(E8 - E8_BIAS) = 2^(E32 - E32_BIAS)
 *       E8 - E8_BIAS = E32 - E32_BIAS
 *                 E8 = E32 - E32_BIAS + E8_BIAS
 *
 * The mantissa can be converted like below. This is a potentially lossy operation that
 * does not take rounding into account.
 *   (1 + M8 / 2^M8_BITS) = (1 + M32 / 2^M32_BITS)
 *                     M8 = M32 * 2^(M8_BITS - M32_BITS)
 *                     M8 = M32 >> (M8_BITS - M32_BITS)
 */
template <std::size_t EXPONENT, std::size_t MANTISSA> class FloatingPoint {
  public:
    using dtype = typename u_int_t<1 + EXPONENT + MANTISSA>::type;
    template <std::size_t, std::size_t> friend class FloatingPoint;

    explicit FloatingPoint(const double v = 0) {
        auto fp = reinterpret_cast<const float64 *>(&v);
        const auto exponent64 = fp->exponent();

        // Set sign bit
        f._sign = fp->sign();

        // Mantissa for normalized and denormalized numbers
        f._mantissa = fp->mantissa() >> (float64::MANTISSA_BITS - MANTISSA_BITS);

        // Infinity or NaN
        if (exponent64 == float64::EXPONENT_INF) {
            f._exponent = EXPONENT_INF;
            f._mantissa = fp->mantissa() ? MANTISSA_NAN : 0;
        } else {
            // Normalized number
            if (exponent64 != 0) {
                auto exp = exponent64 - float64::EXPONENT_BIAS + EXPONENT_BIAS;

                // In range
                if (exp < (1 << EXPONENT_BITS)) {
                    f._exponent = exp;
                }
                // Underflow, treat as zero value
                else if (exponent64 < float64::EXPONENT_BIAS) {
                    f._exponent = 0;
                    f._mantissa = 0;
                }
                // Overflow, treat as infinite value
                else {
                    f._exponent = EXPONENT_INF;
                    f._mantissa = 0;
                }
            }
            // Denormalized number
            else {
                f._exponent = 0;
            }
        }
    }

    // cppcheck-suppress noExplicitConstructor
    template <typename T> FloatingPoint(const T v) : FloatingPoint(double(v)) {}

    operator double() const {
        float64 fp;

        // Infinity or NaN
        if (f._exponent == EXPONENT_INF) {
            fp.f._exponent = float64::EXPONENT_INF;
            fp.f._mantissa = f._mantissa ? MANTISSA_NAN : 0;
        } else {
            // Normalized number
            if (f._exponent != 0) {
                // Convert exponent to equivalent float64 exponent
                fp.f._exponent = float64::dtype(f._exponent) - EXPONENT_BIAS + float64::EXPONENT_BIAS;
            }
            // Denormalized number
            else {
                fp.f._exponent = 0;
            }

            fp.f._mantissa = float64::dtype(f._mantissa) << (float64::MANTISSA_BITS - MANTISSA_BITS);
        }

        // Set sign bit
        fp.f._sign = f._sign;

        return *reinterpret_cast<double *>(&fp);
    }

    static constexpr size_t EXPONENT_BITS = EXPONENT;
    static constexpr size_t EXPONENT_BIAS = (1 << (EXPONENT - 1)) - 1;
    static constexpr size_t EXPONENT_INF = (1 << EXPONENT) - 1;
    static constexpr size_t MANTISSA_BITS = MANTISSA;
    static constexpr size_t MANTISSA_DIVISOR = 1ULL << MANTISSA;
    static constexpr size_t MANTISSA_NAN = (1ULL << MANTISSA) - 1;

    dtype sign() const { return f._sign; }
    dtype exponent() const { return f._exponent; }
    dtype mantissa() const { return f._mantissa; }

    bool isinf() const { return f._exponent == EXPONENT_INF && f._mantissa == 0; }
    bool isnan() const { return f._exponent == EXPONENT_INF && f._mantissa != 0; }

    static constexpr double lowest() { return -max(); }
    static constexpr double max() {
        // Select maximum exponent, 2^(max exponent)
        double _max = 1 << EXPONENT_BIAS;

        // Multiply with largest possible mantissa
        _max *= 1 + double(MANTISSA_DIVISOR - 1) / MANTISSA_DIVISOR;

        return _max;
    }

    template <typename T> void operator+=(const T &other) { *this = double(*this) + other; }

    template <typename T> void operator-=(const T &other) { *this = double(*this) - other; }

    template <typename T> void operator*=(const T &other) { *this = double(*this) * other; }

    template <typename T> void operator/=(const T &other) { *this = double(*this) / other; }

  private:
    struct f {
        dtype _mantissa : MANTISSA;
        dtype _exponent : EXPONENT;
        dtype _sign : 1;
    } f;
};

template <std::size_t EXPONENT, std::size_t MANTISSA>
struct std::numeric_limits<FloatingPoint<EXPONENT, MANTISSA>> : public std::numeric_limits<float> {
    static constexpr float lowest() noexcept { return static_cast<float>(FloatingPoint<EXPONENT, MANTISSA>::lowest()); }
    static constexpr float max() noexcept { return static_cast<float>(FloatingPoint<EXPONENT, MANTISSA>::max()); }
};
