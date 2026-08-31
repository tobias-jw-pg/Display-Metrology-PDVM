# Copyright (c) 2026 Tobias Jianwei. All rights reserved.
# DOI: https://doi.org/10.5281/zenodo.22211087
# This code is licensed under the GNU General Public License v3.0 (GPLv3).
# Foundational Methodology: 10.5281/zenodo.22211087

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# --- Configuration ---
FILE_PATH = "# insert your flickerlab .csv link here #"
CONVERSION_FACTOR = 900   
SAMPLING_RATE = 300000         

def calculate_jeita_flicker(signal: np.ndarray, fs: float, freq_min: float, freq_max: float) -> dict:
    N = len(signal)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    
    p_r0 = np.abs(fft_vals[0]) / N
    ac_amps = 2.0 * np.abs(fft_vals[1:]) / N
    ac_freqs = freqs[1:]
    
    jeita_freq_pts = np.array([0.0, 15.0, 30.0, 45.0, 65.0])
    jeita_weight_pts = np.array([1.0, 1.0, 0.708, 0.376, 0.0])
    weights = np.interp(ac_freqs, jeita_freq_pts, jeita_weight_pts, left=0.0, right=0.0)
    
    flicker_mask = (ac_freqs >= freq_min) & (ac_freqs <= freq_max)
    if not np.any(flicker_mask):
        raise ValueError(f"No frequencies found in the {freq_min} Hz - {freq_max} Hz range.")
        
    weighted_ac_amps = ac_amps * weights
    flicker_spectrum = weighted_ac_amps[flicker_mask]
    flicker_frequencies = ac_freqs[flicker_mask]
    
    max_idx = np.argmax(flicker_spectrum)
    p_r1 = flicker_spectrum[max_idx]
    peak_freq = flicker_frequencies[max_idx]
    
    if p_r0 == 0:
        raise ValueError("DC component (P_r0) is zero.")
        
    if p_r1 <= 0:
        flicker_jeita_db = -np.inf
    else:
        flicker_jeita_db = 20.0 * np.log10(p_r1 / p_r0)
        
    flicker_vesa_db = flicker_jeita_db - 3.0103
    
    return {
        "jeita_db": flicker_jeita_db,
        "vesa_db": flicker_vesa_db,
        "P_r0": p_r0,
        "P_r1": p_r1,
        "peak_freq_hz": peak_freq
    }

def get_wtlm_weight(freq):
    freqs = [0, 20, 30, 40, 50, 60, 240, 800, 2400, 4800, 10000, 60000]
    weights = [1.00, 1.00, 0.708, 0.501, 0.251, 0.200, 0.350, 0.708, 0.450, 0.250, 0.100, 0.010]
    return np.interp(freq, freqs, weights)

def calculate_custom_sensitivity(f, target_visual_angle=24.0):
    if f <= 0:
        return 0.0
    
    numerator = 5679.0
    exponent_term = ((np.log(f) - 7.987) ** 2) / 2.885
    denominator = f * np.exp(exponent_term)
    base_sm = numerator / denominator

    effective_angle = min(target_visual_angle, 24.0)
    baseline_angle = 0.06 
    
    m_piper = effective_angle / baseline_angle
    return base_sm * m_piper

def calculate_pavm(xf, ac_magnitudes, mean_val, fund_freq, target_visual_angle=24.0):
    if mean_val <= 0 or fund_freq <= 0:
        return 0.0
        
    pavm_sum = 0.0
    max_m = int(xf[-1] / fund_freq)
    minkowski_exponent = 2.1
    
    for m in range(1, max_m + 1):
        harmonic_freq = m * fund_freq
        idx = np.argmin(np.abs(xf - harmonic_freq))
        c_m = ac_magnitudes[idx] / mean_val
        s_m = calculate_custom_sensitivity(harmonic_freq, target_visual_angle)
        
        if s_m > 0:
            pavm_sum += (c_m * s_m) ** minkowski_exponent
                
    return pavm_sum ** (1.0 / minkowski_exponent)

def plot_waveform():
    print(f"Loading data from {FILE_PATH}...")
    
    valid_data = []
    time_data = [] 
    
    with open(FILE_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('[FFT]'):
                break
            try:
                parts = line.split(',')
                if len(parts) >= 2:
                    time_data.append(float(parts[0]))
                    valid_data.append(float(parts[1]))
                elif len(parts) == 1:
                    valid_data.append(float(parts[0]))
            except ValueError:
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        time_data.append(float(parts[0]))
                        valid_data.append(float(parts[1]))
                    elif len(parts) == 1:
                        valid_data.append(float(parts[0]))
                except ValueError:
                    continue
    
    data_array = np.array(valid_data) * CONVERSION_FACTOR
    N = len(data_array)
    
    if N == 0:
        print("Error: No valid numerical data found.")
        return

    if len(time_data) == N and N > 1:
        time_array = np.array(time_data)
        total_time_ms = time_data[-1] - time_data[0]
        actual_fs = N / (total_time_ms / 1000.0)
    else:
        actual_fs = SAMPLING_RATE 
        time_array = np.arange(N) * (1000.0 / actual_fs) 

    mean_val = np.mean(data_array)
    ac_signal = data_array - mean_val
    window = np.hanning(N)
    window_correction = 2.0  
    yf = np.fft.rfft(ac_signal * window)
    xf = np.fft.rfftfreq(N, 1 / actual_fs) 
    ac_magnitudes = (np.abs(yf) / N) * 2.0 * window_correction
    
    refresh_zone_mask = (xf > 12.0) & (xf < 125.0)
    if np.any(refresh_zone_mask):
        peak_idx_refresh = np.argmax(ac_magnitudes[refresh_zone_mask])
        refresh_peak_amp = ac_magnitudes[refresh_zone_mask][peak_idx_refresh]
        refresh_freq = xf[refresh_zone_mask][peak_idx_refresh]
    else:
        refresh_peak_amp = refresh_freq = 0
        
    valid_global_mask = xf > 11.0
    if np.any(valid_global_mask):
        peak_idx_global = np.argmax(ac_magnitudes[valid_global_mask])
        global_max_freq = xf[valid_global_mask][peak_idx_global]
        global_max_amp = ac_magnitudes[valid_global_mask][peak_idx_global]
    else:
        global_max_freq = global_max_amp = 0
    
    if global_max_freq > 125.0 and refresh_peak_amp > (global_max_amp * 0.15):
        dominant_freq = refresh_freq
    else:
        dominant_freq = global_max_freq
        
    if dominant_freq > 60000.0:
        mid_zone_mask = (xf >= 400.0) & (xf <= 60000.0)
        if np.any(mid_zone_mask):
            dominant_freq = xf[mid_zone_mask][np.argmax(ac_magnitudes[mid_zone_mask])]
        else:
            dominant_freq = refresh_freq

    split_freq = dominant_freq * 1.98
    
    # ==========================================
    # --- ENVELOPE DEMODULATION FOR LOOP OSCILLATION ---
    # ==========================================
    expected_peak_dist = int(actual_fs / global_max_freq) if global_max_freq > 0 else 1
    safe_distance = max(1,int(expected_peak_dist * 0.2)) 

    robust_max = np.percentile(data_array, 97)
    signal_ptp = np.max(data_array) - np.min(data_array)
    upper_max_height = robust_max * 2.5
    
    upper_prominence = signal_ptp * 0.21
    peaks, _ = find_peaks(data_array, distance=safe_distance, prominence=upper_prominence, height=(None, upper_max_height))

    inv_min_height = -mean_val
    inv_prominence = signal_ptp * 0.21
    inv_distance = max(1, int(expected_peak_dist * 0.2))

    valleys, _ = find_peaks(-data_array, distance=inv_distance, prominence=inv_prominence, height=(inv_min_height, None))
    
    envelope_times = time_array[peaks]
    envelope_vals = data_array[peaks]

    if (len(peaks) + len(valleys)) >= 70:
        env_mean = np.mean(envelope_vals)
        env_ac = envelope_vals - env_mean
        
        if len(valleys) > 0:
            valley_vals = data_array[valleys]
            env_upper_avg = np.mean(envelope_vals)
            env_lower_avg = np.mean(valley_vals)
            true_beating_depth = ((env_upper_avg - env_lower_avg) / (env_upper_avg + env_lower_avg)) * 100.0 if (env_upper_avg + env_lower_avg) > 0 else 0.0
        else:
            true_beating_depth = 0.0
            valley_vals = []
        
        avg_time_step_ms = np.mean(np.diff(envelope_times))
        env_fs = 1000.0 / avg_time_step_ms  
        env_N = len(env_ac)
        env_yf = np.fft.rfft(env_ac * np.hanning(env_N))
        env_xf = np.fft.rfftfreq(env_N, 1 / env_fs)
        env_mags = (np.abs(env_yf) / env_N) * 2.0 * window_correction
        env_mask = (env_xf >= 12) & (env_xf <= 150.0)
        
        if np.any(env_mask):
            loop_idx = np.argmax(env_mags[env_mask])
            loop_osc_freq = env_xf[env_mask][loop_idx]
        else:
            loop_osc_freq = true_beating_depth = 0.0
    else:
        loop_osc_freq = 0.0
        true_beating_depth = 0.0
        valley_vals = data_array[valleys] if len(valleys) > 0 else []

    # ==========================================
    # --- MACRO-ENVELOPE SAWTOOTH ANALYSIS ---
    # ==========================================
    filter_window = max(2, int(actual_fs / 2000.0)) 
    macro_signal = np.convolve(data_array, np.ones(filter_window)/filter_window, mode='same')
    
    macro_distance = max(1, int(actual_fs / 240.0)) 
    
    min_width_samples = int(actual_fs * 0.00025) 
    
    macro_peaks, _ = find_peaks(macro_signal, 
                                distance=macro_distance, 
                                prominence=signal_ptp*0.05, 
                                width=min_width_samples)
                                
    macro_valleys, _ = find_peaks(-macro_signal, 
                                  distance=macro_distance, 
                                  prominence=signal_ptp*0.05, 
                                  width=min_width_samples)
    
    asymmetry_ratios = []
    for i in range(min(len(macro_peaks), len(macro_valleys)) - 1):
        p1 = macro_peaks[i]
        v1 = macro_valleys[i]
        
        if p1 < v1:
            fall_time = v1 - p1
            rise_time = macro_peaks[i+1] - v1
        else:
            rise_time = p1 - v1
            fall_time = macro_valleys[i+1] - p1
            
        if fall_time > 0 and rise_time > 0:
            ratio = (max(rise_time, fall_time) / min(rise_time, fall_time)) - 1
            asymmetry_ratios.append(ratio)
            
    avg_asymmetry = np.mean(asymmetry_ratios) if asymmetry_ratios else 0.0
    sawtooth_flag = "Detected" if avg_asymmetry > 0 else "N.D."
    
    # ==========================================
        
    full_spectrum_weights = get_wtlm_weight(xf)
    wtd_ac_spectrum = ac_magnitudes * full_spectrum_weights
    
    try:
        pri_jeita_results = calculate_jeita_flicker(data_array, actual_fs, freq_min=0.5, freq_max=split_freq)
        pri_jeita_str = f"{pri_jeita_results['jeita_db']:.2f} dB"
        pri_vesa_str = f"{pri_jeita_results['vesa_db']:.2f} dB"
    except ValueError:
        pri_jeita_str = pri_vesa_str = "N/A"

    try:
        aux_jeita_results = calculate_jeita_flicker(data_array, actual_fs, freq_min=split_freq, freq_max=(actual_fs/2.0))
        aux_jeita_str = f"{aux_jeita_results['jeita_db']:.2f} dB"
        aux_vesa_str = f"{aux_jeita_results['vesa_db']:.2f} dB"
    except ValueError:
        aux_jeita_str = aux_vesa_str = "N/A"

    aux_search_floor = max(split_freq, 400.0)
    high_freq_mask = xf > aux_search_floor
    low_freq_mask = xf <= split_freq
    freq_resolution = actual_fs / N
    
    primary_band_wtd_mags = wtd_ac_spectrum[low_freq_mask]
    primary_pavm = calculate_pavm(xf, ac_magnitudes, mean_val, dominant_freq, target_visual_angle=24.0)
    
    pri_rolling_window_hz = 50.0  
    pri_window_bins = max(1, int(pri_rolling_window_hz / freq_resolution))
    
    if len(primary_band_wtd_mags) < pri_window_bins:
        pri_window_bins = len(primary_band_wtd_mags)
        
    pri_rolling_sq_sum = np.convolve(primary_band_wtd_mags**2, np.ones(pri_window_bins), mode='valid')
    peak_pri_rolling_rss_amp = np.sqrt(np.max(pri_rolling_sq_sum)) if len(pri_rolling_sq_sum) > 0 else 0.0
    primary_wtd_pct = (peak_pri_rolling_rss_amp / mean_val) * 100.0 if mean_val > 0 else 0.0

    if np.any(high_freq_mask):
        aux_mags = ac_magnitudes[high_freq_mask]
        aux_peak_idx = np.argmax(aux_mags)
        aux_freq = xf[high_freq_mask][aux_peak_idx]
        aux_band_wtd_mags = wtd_ac_spectrum[high_freq_mask]
        
        aux_rolling_window_hz = 50.0 
        aux_window_bins = max(1, int(aux_rolling_window_hz / freq_resolution))
        
        if len(aux_band_wtd_mags) < aux_window_bins:
            aux_window_bins = len(aux_band_wtd_mags)
            
        aux_rolling_sq_sum = np.convolve(aux_band_wtd_mags**2, np.ones(aux_window_bins), mode='valid')
        peak_aux_rolling_rss_amp = np.sqrt(np.max(aux_rolling_sq_sum)) if len(aux_rolling_sq_sum) > 0 else 0.0
        aux_wtd_pct = (peak_aux_rolling_rss_amp / mean_val) * 100.0 if mean_val > 0 else 0.0
        aux_pavm = calculate_pavm(xf, ac_magnitudes, mean_val, aux_freq, target_visual_angle=24.0)
    else:
        aux_freq = aux_wtd_pct = aux_pavm = 0.0

    # ==========================================
    # --- FLAGGING LOGIC FOR BEATING FREQUENCY ---
    # ==========================================
    beat_freq_str = f"{loop_osc_freq:.1f} Hz"
    beat_mod_str = f"{true_beating_depth:.2f}%"

    if loop_osc_freq > 0:
        if loop_osc_freq < 16:
            beat_freq_str += " [[!]]"
        elif loop_osc_freq <= 30:
            beat_freq_str += " [!]"
        elif loop_osc_freq <= 47:
            pass  # Leave exactly as it is
        elif mean_val < 1000 and loop_osc_freq > 48:
            beat_freq_str = "0.0 Hz"
            beat_mod_str = "0.00%"
        elif mean_val >= 1000 and loop_osc_freq > 70:
            beat_freq_str = "0.0 Hz"
            beat_mod_str = "0.00%"
        else:
            beat_freq_str += " [!]"
            
    # ==========================================
    # standardise pdvm results to be between 0.001 to 100
    primary_pavm = primary_pavm/10
    aux_pavm = aux_pavm/10
    # ==========================================


    
    # --- Plotting ---
    plt.figure(figsize=(10, 5)) 
    plt.plot(time_array, data_array, color='red', linewidth=1.2)

    if (len(peaks) + len(valleys)) > 70:
        plt.plot(envelope_times, envelope_vals, "o", color='blue', markersize=4, label="Detected Peaks")
        plt.plot(time_array[valleys], valley_vals, "x", color='blue', markersize=4, label="Detected Valleys")
    
    plt.title('Sensor Waveform (DC & PWM)', fontsize=13, loc="left")
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Luminance (cd/m2)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    max_idx = min(250000, N - 1)
    plt.xlim(0, time_array[max_idx])

        
    metrics_text = (
        f"nits : {mean_val:.1f}\n"
        f"Dominant (PWM/DC/VCOM) Freq: {dominant_freq:.1f} Hz\n"
        f"Weighted TLM flicker %: {primary_wtd_pct:.2f}%\n"
        f"PDVM : {primary_pavm:.3f}\n"
        f"--------------------------------------------\n"
        f"Sub-harmonic Beating Freq: {beat_freq_str}\n"
        f"Beating modulation : {beat_mod_str}\n"
        f"--------------------------------------------\n"
        f"FRC | DAC Sawtooth Artifacts: {sawtooth_flag}\n"
        f"Sawtooth Asymmetry Deviation(Δ): {avg_asymmetry:.2f}\n"
        f"--------------------------------------------\n"
        f"Auxiliary (Harmonic) Freq: >{aux_freq:.1f} Hz\n"
        f"Weighted TLM flicker %: {aux_wtd_pct:.2f}%\n"
        f"PDVM : {aux_pavm:.3f}\n"
        )
    
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='gray')
    
    plt.gca().text(0.98, 0.95, metrics_text, transform=plt.gca().transAxes, 
                   fontsize=9, verticalalignment='bottom', horizontalalignment='right', 
                   bbox=props, weight='bold', color='black', family='monospace')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_waveform()
