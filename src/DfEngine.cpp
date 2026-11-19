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
#include "DfEngine.hpp"

#include <spdlog/spdlog.h>

#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <au/units/hertz.hh>

namespace df
{

static constexpr double SPEED_OF_LIGHT = 2.998e8;
static constexpr double PI             = M_PI;

DfEngine::DfEngine(double angle_step_deg)
    : angle_step_deg_(std::max(angle_step_deg, 0.1))
{
    if (angle_step_deg < 0.1)
        spdlog::warn("DfEngine: angle_step_deg={} clamped to 0.1 to prevent infinite sweep",
                     angle_step_deg);
}

DfResult DfEngine::compute(const std::vector<std::vector<float>>& iq_snapshots,
                           const std::vector<AntennaElement>&     antennas,
                           au::QuantityD<au::Hertz>               freq) const
{
    DfResult result;
    result.center_freq_hz = freq;
    result.algorithm      = "MUSIC";

    const double freq_hz = freq.in(au::hertz);

    int M = static_cast<int>(iq_snapshots.size());
    if (M < 2 || M != static_cast<int>(antennas.size()))
    {
        spdlog::warn("DfEngine: need ≥2 antennas, got {}", M);
        return result;
    }

    int N = static_cast<int>(iq_snapshots[0].size()) / 2;  // samples per element
    // Clamp N to the smallest snapshot so the data matrix loop never OOBs on
    // shorter snapshots from other scanner nodes.
    for (int i = 1; i < M; ++i)
        N = std::min(N, static_cast<int>(iq_snapshots[i].size()) / 2);
    if (N < 16)
    {
        spdlog::warn("DfEngine: snapshot too short ({} samples)", N);
        return result;
    }

    // ── Build array data matrix X (M×N) ──────────────────────────────────────
    using Cx = std::complex<double>;
    Eigen::MatrixXcd X(M, N);
    for (int i = 0; i < M; ++i)
    {
        for (int n = 0; n < N; ++n)
        {
            X(i, n) = Cx(iq_snapshots[i][2 * n], iq_snapshots[i][2 * n + 1]);
        }
    }

    // ── Sample covariance R = (1/N) X X^H ────────────────────────────────────
    Eigen::MatrixXcd R = (X * X.adjoint()) / static_cast<double>(N);

    // ── Eigendecomposition (ascending eigenvalue order) ───────────────────────
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(R);
    if (solver.info() != Eigen::Success)
    {
        spdlog::warn("DfEngine: eigendecomposition failed");
        return result;
    }

    // Assume D=1 source; noise subspace = eigenvectors for M-1 smallest values.
    // For multi-source scenarios extend with MDL/AIC criterion.
    int              D  = 1;
    Eigen::MatrixXcd En = solver.eigenvectors().leftCols(M - D);  // (M × M-D)

    // ── MUSIC pseudo-spectrum sweep ───────────────────────────────────────────
    // Bearing convention: θ=0° North, θ=90° East (clockwise), ENU frame.
    // Steering vector component for antenna i at bearing θ:
    //   a_i(θ) = exp(+j·k·(x_i·sin(θ) + y_i·cos(θ)))
    // where k = 2π/λ and u(θ) = (sinθ, cosθ) points from the array TOWARDS the
    // source. An element with positive projection r_i·u is closer to the source,
    // so the wavefront reaches it earlier by τ = (r_i·u)/c. After standard SDR
    // down-conversion (RF above the LO → positive baseband frequency) an earlier
    // arrival is a phase ADVANCE e^{+j·2π·f_c·τ} = e^{+j·k·(r_i·u)}. This is the
    // textbook array manifold (Van Trees: v = e^{-j·kᵀp} with wavenumber
    // k = -(2π/λ)·u). Negating the phase evaluates u → -u, i.e. every bearing
    // is reported rotated by exactly 180° (the regression introduced by
    // ce688f1 — tests/unit/test_df_engine.cpp caught it: 0°→180°, 90°→270°).
    // If field bearings look mirrored, check the antenna x/y convention
    // (x = East, y = North), the scanner_id→element mapping, and whether a
    // receiver delivers spectrally inverted (conjugated) IQ — not this sign.
    double k = 2.0 * PI * freq_hz / SPEED_OF_LIGHT;

    double best_power   = -1.0;
    double best_bearing = 0.0;
    double sum_power    = 0.0;
    int    n_steps      = 0;

    for (double bearing = 0.0; bearing < 360.0; bearing += angle_step_deg_)
    {
        double bearing_rad = bearing * PI / 180.0;

        Eigen::VectorXcd a(M);
        for (int i = 0; i < M; ++i)
        {
            // Phase advance for elements nearer the source (see comment above).
            double phase =
                k * (antennas[i].x * std::sin(bearing_rad) + antennas[i].y * std::cos(bearing_rad));
            a(i) = std::exp(Cx(0.0, phase));
        }

        // P(θ) = 1 / ‖Eₙᴴ a‖²
        double denom = (En.adjoint() * a).squaredNorm();
        double power = 1.0 / (denom + 1e-20);

        sum_power += power;
        ++n_steps;

        if (power > best_power)
        {
            best_power   = power;
            best_bearing = bearing;
        }
    }

    // ── Confidence: log peak-to-mean ratio, normalised to 0–1 ─────────────────
    // A sharp null in the noise subspace projection gives a very high peak.
    // 20 dB peak-to-mean → confidence ≈ 1.0 (empirical scale).
    double mean_power    = sum_power / n_steps;
    double prominence_db = 10.0 * std::log10(best_power / (mean_power + 1e-20));
    float  confidence    = std::clamp(static_cast<float>(prominence_db / 20.0), 0.0f, 1.0f);

    result.azimuth_deg  = best_bearing;
    result.confidence   = confidence;
    result.num_elements = M;
    result.valid        = (confidence > 0.05f);

    spdlog::debug("DfEngine: {:.3f} MHz → {:.1f}° confidence={:.2f} (peak/mean={:.1f} dB) M={}",
                  freq_hz / 1e6,
                  best_bearing,
                  confidence,
                  prominence_db,
                  M);
    return result;
}

}  // namespace df

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
