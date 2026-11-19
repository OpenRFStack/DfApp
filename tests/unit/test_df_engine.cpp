/*
========================================================================
Project: OpenRFStack
Author:  Brendan Michaud
Year:    2026
Part of OpenRFStack (https://github.com/OpenRFStack)

Licensed under the Personal Use License.
Do not use for commercial, organizational, or military purposes.
Contact author for permission: https://github.com/OpenRFStack
========================================================================
*/
#include "Config.hpp"
#include "DfEngine.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include <au/units/hertz.hh>

// ── Synthetic IQ generator ────────────────────────────────────────────────────
//
// Generates a narrowband baseband signal at a small frequency offset (1 kHz)
// so the covariance matrix has proper rank across samples. The inter-antenna
// phase difference encodes the source bearing, which MUSIC should recover.

static std::vector<float> makeIq(double        ant_x,
                                 double        ant_y,  // antenna position (m, ENU)
                                 double        bearing_deg,
                                 double        freq_hz,  // source bearing and carrier
                                 int           n_samples,
                                 double        snr_db,
                                 std::mt19937& rng)
{
    constexpr double C           = 2.998e8;
    constexpr double SAMPLE_RATE = 20e6;
    constexpr double TONE_HZ     = 1000.0;  // 1 kHz offset from DC

    double bearing_rad = bearing_deg * M_PI / 180.0;
    double k           = 2.0 * M_PI * freq_hz / C;
    // Phase advance at this antenna relative to the origin
    double spatial_phase = k * (ant_x * std::sin(bearing_rad) + ant_y * std::cos(bearing_rad));

    double                          noise_sigma = std::pow(10.0, -snr_db / 20.0);
    std::normal_distribution<float> noise(0.0f, (float)noise_sigma);

    std::vector<float> iq(2 * n_samples);
    for (int n = 0; n < n_samples; ++n)
    {
        double t      = n / SAMPLE_RATE;
        double phase  = spatial_phase + 2.0 * M_PI * TONE_HZ * t;
        iq[2 * n]     = (float)std::cos(phase) + noise(rng);
        iq[2 * n + 1] = (float)std::sin(phase) + noise(rng);
    }
    return iq;
}

// 5-element UCA (uniform circular array), 2 m radius, matching config/df.xml
static std::vector<df::AntennaElement> makeUca()
{
    return {
        {"scanner-0", 0.00f, 0.00f, 0.0f},
        {"scanner-1", 1.90f, 0.62f, 0.0f},
        {"scanner-2", 1.18f, 1.62f, 0.0f},
        {"scanner-3", -1.18f, 1.62f, 0.0f},
        {"scanner-4", -1.90f, 0.62f, 0.0f},
    };
}

// ── Helpers ───────────────────────────────────────────────────────────────────

static double angleDiff(double a, double b)
{
    double d = std::fmod(std::abs(a - b), 360.0);
    return d > 180.0 ? 360.0 - d : d;
}

static std::vector<std::vector<float>>
makeSyntheticSnapshot(const std::vector<df::AntennaElement>& antennas,
                      double                                 bearing_deg,
                      double                                 freq_hz,
                      int                                    n_samples,
                      double                                 snr_db,
                      std::mt19937&                          rng)
{
    std::vector<std::vector<float>> snapshots;
    for (const auto& a : antennas)
        snapshots.push_back(makeIq(a.x, a.y, bearing_deg, freq_hz, n_samples, snr_db, rng));
    return snapshots;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

class DfEngineTest : public ::testing::Test
{
protected:
    df::DfEngine                    engine_{0.5};
    std::vector<df::AntennaElement> uca_ = makeUca();
    std::mt19937                    rng_{42};
    static constexpr double         FREQ_HZ       = 99.5e6;
    static constexpr int            N_SAMP        = 1024;
    static constexpr double         HIGH_SNR      = 25.0;
    static constexpr double         TOLERANCE_DEG = 2.0;
};

TEST_F(DfEngineTest, KnownBearing_North_0deg)
{
    auto snap = makeSyntheticSnapshot(uca_, 0.0, FREQ_HZ, N_SAMP, HIGH_SNR, rng_);
    auto r    = engine_.compute(snap, uca_, au::hertz(FREQ_HZ));
    EXPECT_TRUE(r.valid);
    EXPECT_LT(angleDiff(r.azimuth_deg, 0.0), TOLERANCE_DEG) << "got " << r.azimuth_deg << "°";
    EXPECT_GT(r.confidence, 0.3f);
}

TEST_F(DfEngineTest, KnownBearing_East_90deg)
{
    auto snap = makeSyntheticSnapshot(uca_, 90.0, FREQ_HZ, N_SAMP, HIGH_SNR, rng_);
    auto r    = engine_.compute(snap, uca_, au::hertz(FREQ_HZ));
    EXPECT_TRUE(r.valid);
    EXPECT_LT(angleDiff(r.azimuth_deg, 90.0), TOLERANCE_DEG) << "got " << r.azimuth_deg << "°";
}

TEST_F(DfEngineTest, KnownBearing_South_180deg)
{
    auto snap = makeSyntheticSnapshot(uca_, 180.0, FREQ_HZ, N_SAMP, HIGH_SNR, rng_);
    auto r    = engine_.compute(snap, uca_, au::hertz(FREQ_HZ));
    EXPECT_TRUE(r.valid);
    EXPECT_LT(angleDiff(r.azimuth_deg, 180.0), TOLERANCE_DEG) << "got " << r.azimuth_deg << "°";
}

TEST_F(DfEngineTest, KnownBearing_West_270deg)
{
    auto snap = makeSyntheticSnapshot(uca_, 270.0, FREQ_HZ, N_SAMP, HIGH_SNR, rng_);
    auto r    = engine_.compute(snap, uca_, au::hertz(FREQ_HZ));
    EXPECT_TRUE(r.valid);
    EXPECT_LT(angleDiff(r.azimuth_deg, 270.0), TOLERANCE_DEG) << "got " << r.azimuth_deg << "°";
}

TEST_F(DfEngineTest, KnownBearing_Diagonal_135deg)
{
    auto snap = makeSyntheticSnapshot(uca_, 135.0, FREQ_HZ, N_SAMP, HIGH_SNR, rng_);
    auto r    = engine_.compute(snap, uca_, au::hertz(FREQ_HZ));
    EXPECT_TRUE(r.valid);
    EXPECT_LT(angleDiff(r.azimuth_deg, 135.0), TOLERANCE_DEG) << "got " << r.azimuth_deg << "°";
}

// Low SNR: bearing less precise but should still be valid
TEST_F(DfEngineTest, LowSnr_6dB_StillConverges)
{
    constexpr double LOW_SNR = 6.0;
    constexpr double RELAXED = 5.0;
    auto             snap    = makeSyntheticSnapshot(uca_, 45.0, FREQ_HZ, N_SAMP, LOW_SNR, rng_);
    auto             r       = engine_.compute(snap, uca_, au::hertz(FREQ_HZ));
    EXPECT_TRUE(r.valid);
    EXPECT_LT(angleDiff(r.azimuth_deg, 45.0), RELAXED) << "got " << r.azimuth_deg << "°";
}

// Only 2 elements — minimum for computation, but worst-case accuracy
TEST_F(DfEngineTest, TwoElements_ReturnsResult)
{
    auto two_ant = std::vector<df::AntennaElement>{uca_[0], uca_[1]};
    auto snap    = makeSyntheticSnapshot(two_ant, 60.0, FREQ_HZ, N_SAMP, HIGH_SNR, rng_);
    auto r       = engine_.compute(snap, two_ant, au::hertz(FREQ_HZ));
    EXPECT_TRUE(r.valid);
    EXPECT_EQ(r.num_elements, 2);
}

// Fewer than 2 elements → invalid
TEST_F(DfEngineTest, OneElement_ReturnsInvalid)
{
    auto one_ant = std::vector<df::AntennaElement>{uca_[0]};
    auto snap = std::vector<std::vector<float>>{makeIq(0, 0, 0, FREQ_HZ, N_SAMP, HIGH_SNR, rng_)};
    auto r    = engine_.compute(snap, one_ant, au::hertz(FREQ_HZ));
    EXPECT_FALSE(r.valid);
}

// Empty snapshot → invalid
TEST_F(DfEngineTest, EmptySnapshots_ReturnsInvalid)
{
    auto r = engine_.compute({}, {}, au::hertz(FREQ_HZ));
    EXPECT_FALSE(r.valid);
}

// Different frequency: λ changes so the phase delays change — MUSIC should
// still recover the correct bearing for a signal at 433 MHz (ISM band).
TEST_F(DfEngineTest, UhfFrequency_433MHz)
{
    constexpr double UHF  = 433e6;
    auto             snap = makeSyntheticSnapshot(uca_, 220.0, UHF, N_SAMP, HIGH_SNR, rng_);
    auto             r    = engine_.compute(snap, uca_, au::hertz(UHF));
    EXPECT_TRUE(r.valid);
    EXPECT_LT(angleDiff(r.azimuth_deg, 220.0), TOLERANCE_DEG) << "got " << r.azimuth_deg << "°";
}

// 5 elements should give better accuracy than 2 at moderate SNR (10 dB).
// At high SNR both saturate to confidence=1; at moderate SNR the larger
// noise subspace (dimension 4 vs 1) gives MUSIC a sharper null for 5 elements.
TEST_F(DfEngineTest, MoreElements_BetterAccuracyAtModerateSnr)
{
    constexpr double MOD_SNR = 10.0;
    constexpr double BEARING = 30.0;

    auto snap5 = makeSyntheticSnapshot(uca_, BEARING, FREQ_HZ, N_SAMP, MOD_SNR, rng_);
    auto r5    = engine_.compute(snap5, uca_, au::hertz(FREQ_HZ));

    auto two_ant = std::vector<df::AntennaElement>{uca_[0], uca_[1]};
    auto snap2   = makeSyntheticSnapshot(two_ant, BEARING, FREQ_HZ, N_SAMP, MOD_SNR, rng_);
    auto r2      = engine_.compute(snap2, two_ant, au::hertz(FREQ_HZ));

    // 5 elements must recover the bearing; 2 elements may be less accurate.
    EXPECT_TRUE(r5.valid);
    EXPECT_LT(angleDiff(r5.azimuth_deg, BEARING), 3.0) << "5-el got " << r5.azimuth_deg << "°";
    // Confidence with more elements should be at least as high.
    EXPECT_GE(r5.confidence, r2.confidence)
        << "5-el=" << r5.confidence << " 2-el=" << r2.confidence;
}

// ── Regression: steering-vector sign (audit finding DF-01) ────────────────────
//
// Commit ce688f1 negated the steering vector, rotating every bearing by 180°.
// This generator does not assume a phase sign: it models the physical arrival
// delay τ_i of a plane wave at each element and an ideal quadrature
// down-converter (RF at f_c + f_tone mixed with e^{-j2πf_c t}), which is what
// any SoapySDR CF32 stream delivers. Swept over all quadrants on the
// asymmetric L-shaped array from config/df.xml so a 180° rotation or an E↔W /
// N↔S mirror cannot pass.
static std::vector<std::vector<float>>
makeDelayBasedSnapshot(const std::vector<df::AntennaElement>& antennas,
                       double                                 bearing_deg,
                       double                                 fc_hz,
                       int                                    n_samples,
                       std::mt19937&                          rng)
{
    constexpr double                 C       = 2.998e8;
    constexpr double                 FS      = 2e6;
    constexpr double                 TONE_HZ = 12.5e3;
    const double                     th      = bearing_deg * M_PI / 180.0;
    const double                     ux = std::sin(th), uy = std::cos(th);  // towards the source
    std::normal_distribution<double> noise(0.0, 0.02);
    std::vector<std::vector<float>>  out;
    for (const auto& a : antennas)
    {
        // Element nearer the source (positive r·u) is reached EARLIER.
        const double       tau = -(a.x * ux + a.y * uy) / C;
        std::vector<float> iq(2 * (size_t)n_samples);
        for (int n = 0; n < n_samples; ++n)
        {
            const double t = n / FS;
            // Received RF: cos(2π(fc+ftone)(t − τ)). Ideal I/Q down-conversion
            // by fc keeps exp(j[2π(fc+ftone)(t − τ) − 2π·fc·t]).
            const double ph = 2.0 * M_PI * (fc_hz + TONE_HZ) * (t - tau) - 2.0 * M_PI * fc_hz * t;
            iq[2 * n]       = (float)(std::cos(ph) + noise(rng));
            iq[2 * n + 1]   = (float)(std::sin(ph) + noise(rng));
        }
        out.push_back(std::move(iq));
    }
    return out;
}

TEST_F(DfEngineTest, DelayModel_AllQuadrants_LShapedArray)
{
    // config/df.xml example array (λ/2 at 300 MHz).
    const std::vector<df::AntennaElement> l_array = {
        {"scanner-0", 0.00f, 0.00f, 0.0f},
        {"scanner-1", 0.00f, 0.50f, 0.0f},
        {"scanner-2", 0.50f, 0.00f, 0.0f},
        {"scanner-3", 1.00f, 0.00f, 0.0f},
    };
    constexpr double FC = 300e6;
    for (int b = 0; b < 360; b += 15)
    {
        auto snap = makeDelayBasedSnapshot(l_array, b, FC, N_SAMP, rng_);
        auto r    = engine_.compute(snap, l_array, au::hertz(FC));
        EXPECT_TRUE(r.valid) << "bearing " << b;
        EXPECT_LT(angleDiff(r.azimuth_deg, b), TOLERANCE_DEG)
            << "true " << b << "° → estimated " << r.azimuth_deg << "°";
    }
}

// Documents the most likely real-world cause of "mirrored" bearings: entering
// antenna x (East) coordinates with the wrong sign mirrors bearings E↔W
// (θ → −θ). The engine is right; the array description is wrong.
TEST_F(DfEngineTest, MirroredArrayConfig_MirrorsBearing)
{
    const std::vector<df::AntennaElement> truth = {
        {"a0", 0.00f, 0.00f, 0.0f},
        {"a1", 0.00f, 0.50f, 0.0f},
        {"a2", 0.50f, 0.00f, 0.0f},
        {"a3", 1.00f, 0.00f, 0.0f},
    };
    auto mirrored = truth;
    for (auto& a : mirrored)
        a.x = -a.x;  // operator typed West-positive
    constexpr double FC   = 300e6;
    auto             snap = makeDelayBasedSnapshot(truth, 40.0, FC, N_SAMP, rng_);
    auto             good = engine_.compute(snap, truth, au::hertz(FC));
    auto             bad  = engine_.compute(snap, mirrored, au::hertz(FC));
    EXPECT_LT(angleDiff(good.azimuth_deg, 40.0), TOLERANCE_DEG);
    EXPECT_LT(angleDiff(bad.azimuth_deg, 320.0), TOLERANCE_DEG)
        << "mirrored config should report −40° (got " << bad.azimuth_deg << "°)";
}

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
