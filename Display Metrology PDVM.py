import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# --- Configuration ---
FILE_PATH = "/Users/kamjianwei/Downloads/S20 FE 55 percent fft.csv"        # Replace with your actual text file name
CONVERSION_FACTOR = 900   # Your AD to Nits calibration multiplier
SAMPLING_RATE = 300000         # Ensure this matches your actual hardware speed


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
        raise ValueError("DC component (P_r0) is zero. Check signal baseline.")
        
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
    """
    Calculates the human visual sensitivity (Sm) for a given frequency in Hz,
    incorporating Piper's Law for spatial summation up to a specific visual angle.
    """
    if f <= 0:
        return 0.0
    
    # 1. Calculate the baseline temporal sensitivity (Sm) using the log-normal function[cite: 1]
    numerator = 5679.0
    exponent_term = ((np.log(f) - 7.987) ** 2) / 2.885
    denominator = f * np.exp(exponent_term)
    base_sm = numerator / denominator

    
    # 2. Calculate the spatial modifier based on Piper's Law
    effective_angle = min(target_visual_angle, 24.0)
    baseline_angle = 0.06 # Based on the Miller et al. experimental setup[cite: 1]
    
    # M_piper = target_angle / baseline_angle (simplified from square root of areas)
    m_piper = effective_angle / baseline_angle
    
    # 3. Return the tightened sensitivity
    return base_sm * m_piper


def calculate_pavm(xf, ac_magnitudes, mean_val, fund_freq, target_visual_angle=24.0):
    """
    Calculates the Phantom Array Visibility Measure (PAVM) over the harmonic series.
    """
    if mean_val <= 0 or fund_freq <= 0:
        return 0.0
        
    pavm_sum = 0.0
    max_m = int(xf[-1] / fund_freq)
    
    # The Minkowski exponent for PAVM is 2.1[cite: 1]
    minkowski_exponent = 2.1
    
    for m in range(1, max_m + 1):
        harmonic_freq = m * fund_freq
        idx = np.argmin(np.abs(xf - harmonic_freq))
        
        # Cm is the amplitude divided by the direct current (DC) value[cite: 1]
        c_m = ac_magnitudes[idx] / mean_val
        
        # Sm is the sensitivity value of visibility for the frequency[cite: 1]
        s_m = calculate_custom_sensitivity(harmonic_freq, target_visual_angle)
        
        if s_m > 0:
            # Add to summation: (Cm * Sm)^2.1
            pavm_sum += (c_m * s_m) ** minkowski_exponent
                
    return pavm_sum ** (1.0 / minkowski_exponent)


def plot_waveform():
    print(f"Loading data from {FILE_PATH}...")
    
    valid_data = []
    time_data = [] 
    
    with open(FILE_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            
            # CRITICAL: You must stop reading if you hit the FFT section of a standard FlickerLab export.
            # Otherwise, the try block below will successfully parse the FFT numbers and mix 
            # them into your time/luminance arrays, completely corrupting your math.
            if line.startswith('[FFT]'):
                break
                
            # The Nested Try-Except Fallback Method
            try:
                # ATTEMPT 1: Try treating it as a comma-separated CSV
                parts = line.split(',')
                
                # If the file is actually space-separated, parts[0] will equal a string like "0.001 0.432".
                # Attempting to convert that string using float() will instantly throw a ValueError,
                # which safely triggers the except block below.
                if len(parts) >= 2:
                    time_data.append(float(parts[0]))
                    valid_data.append(float(parts[1]))
                elif len(parts) == 1:
                    valid_data.append(float(parts[0]))
                    
            except ValueError:
                # ATTEMPT 2: The comma split failed. Fallback to space-separated text.
                try:
                    parts = line.split()
                    
                    if len(parts) >= 2:
                        time_data.append(float(parts[0]))
                        valid_data.append(float(parts[1]))
                    elif len(parts) == 1:
                        valid_data.append(float(parts[0]))
                        
                except ValueError:
                    # ATTEMPT 3: Both failed. This means the line is pure text 
                    # (like the "x0000,y0000" header in your CSV file). 
                    # We catch the final ValueError and silently skip to the next line.
                    continue
    
    data_array = np.array(valid_data) * CONVERSION_FACTOR
    N = len(data_array)
    
    if N == 0:
        print("Error: No valid numerical data found in the file.")
        return

    # Handle Time Array and Sampling Rate
    if len(time_data) == N and N > 1:
        time_array = np.array(time_data)
        total_time_ms = time_data[-1] - time_data[0]
        actual_fs = N / (total_time_ms / 1000.0)
        print(f"Successfully loaded {N} points.")
        print(f"Dynamically Calculated Sampling Rate: {actual_fs:.2f} Hz")
    else:
        actual_fs = SAMPLING_RATE 
        time_array = np.arange(N) * (1000.0 / actual_fs) 
        print(f"Successfully loaded {N} points.")
        print(f"Using Hardcoded Sampling Rate: {actual_fs:.2f} Hz")

    # --- Global Metrics ---
    mean_val = np.mean(data_array)
    ac_signal = data_array - mean_val
    window = np.hanning(N)
    window_correction = 2.0  
    yf = np.fft.rfft(ac_signal * window)
    xf = np.fft.rfftfreq(N, 1 / actual_fs) 
    ac_magnitudes = (np.abs(yf) / N) * 2.0 * window_correction
    
    # 1. Search the standard display refresh rate zone and vcom flicker (< 60 Hz)
    refresh_zone_mask = (xf > 12.0) & (xf < 125.0)
    if np.any(refresh_zone_mask):
        peak_idx_refresh = np.argmax(ac_magnitudes[refresh_zone_mask])
        refresh_peak_amp = ac_magnitudes[refresh_zone_mask][peak_idx_refresh]
        refresh_freq = xf[refresh_zone_mask][peak_idx_refresh]
    else:
        refresh_peak_amp = refresh_freq = 0
        
    # 2. Find the absolute highest peak in the ENTIRE spectrum

    valid_global_mask = xf > 11.0

    if np.any(valid_global_mask):
        peak_idx_global = np.argmax(ac_magnitudes[valid_global_mask])
        global_max_freq = xf[valid_global_mask][peak_idx_global]
        global_max_amp = ac_magnitudes[valid_global_mask][peak_idx_global]
    else:
        global_max_freq = global_max_amp = 0
    
    # 3. SMART SELECTION: If the global max is a high-frequency carrier (like 1538 Hz), 
    # but there is a strong macro-pulse (like 120 Hz), prioritize the macro-pulse.
    if global_max_freq > 125.0 and refresh_peak_amp > (global_max_amp * 0.15):
        dominant_freq = refresh_freq
    else:
        dominant_freq = global_max_freq
        

    # If the dominant frequency exceeds 60 kHz, it pushes the split_freq beyond the sensor's
    # Nyquist limit, causing the Auxiliary channel to go blind. We force the dominant freq 
    # back to the human-relevant range, which pushes the ultra-high noise to the Auxiliary channel.
    if dominant_freq > 60000.0:
        mid_zone_mask = (xf >= 400.0) & (xf <= 60000.0)
        if np.any(mid_zone_mask):
            # Find the true peak in the normal visual/PWM range
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

    #Define minimum and maximum bounds for the blue dots
    upper_max_height = robust_max * 2
    
    # 1. Extract UPPER peaks (Blue Dots)
    upper_prominence = signal_ptp * 0.4
    peaks, _ = find_peaks(data_array, distance=safe_distance, prominence=upper_prominence, height=(None, upper_max_height))

    
    # 2. Extract LOWER valleys (Green X's) using Highest Peak Concentration Floor
    # Filter out absolute zeroes to find the ON-state plateau concentration
    on_state_data = data_array[data_array > mean_val]
    hist, bin_edges = np.histogram(on_state_data, bins=50)
    plateau_nits = bin_edges[np.argmax(hist)]
    
    # Define parameters for INVERTED peak detection
    #inverted_floor = -plateau_nits

    #valley_threshold_nits = mean_val * 0.9
    
    # Because we are searching an inverted array (-data_array), we invert the threshold
    inv_min_height = -mean_val

    #inv_min_height = -valley_threshold_nits
    #inv_max_height = inverted_floor * 0.1 
    
    
    inv_prominence = signal_ptp * 0.5
    inv_distance = max(1, int(expected_peak_dist * 0.2))

    valleys, _ = find_peaks(-data_array, 
                            distance=inv_distance, 
                            prominence=inv_prominence, 
                            height=(inv_min_height, None))
    
    
    envelope_times = time_array[peaks]
    envelope_vals = data_array[peaks]

    if (len(peaks) + len(valleys)) >= 70:
        env_mean = np.mean(envelope_vals)
        env_ac = envelope_vals - env_mean
        
        # Calculate True Envelope Beating Depth (Michelson Contrast)
        if len(valleys) > 0:
            valley_vals = data_array[valleys]
            env_upper_avg = np.mean(envelope_vals)
            env_lower_avg = np.mean(valley_vals)
            true_beating_depth = ((env_upper_avg - env_lower_avg) / (env_upper_avg + env_lower_avg)) * 100.0 if (env_upper_avg + env_lower_avg) > 0 else 0.0
        else:
            true_beating_depth = 0.0
            valley_vals = []
        
        # FFT on Envelope
        avg_time_step_ms = np.mean(np.diff(envelope_times))
        env_fs = 1000.0 / avg_time_step_ms  
        env_N = len(env_ac)
        env_yf = np.fft.rfft(env_ac * np.hanning(env_N))
        env_xf = np.fft.rfftfreq(env_N, 1 / env_fs)
        env_mags = (np.abs(env_yf) / env_N) * 2.0 * window_correction
        env_mask = (env_xf >= 12) & (env_xf <= 70.0)
        
        if np.any(env_mask):
            loop_idx = np.argmax(env_mags[env_mask])
            loop_osc_freq = env_xf[env_mask][loop_idx]
    
        else:
            loop_osc_freq = loop_mod_depth = weighted_loop_severity = true_beating_depth = 0.0
    else:
        loop_osc_freq = 0.0
        true_beating_depth = 0.0
        
        # Preserve valley_vals so the plot function can still draw any sparse 'x's found
        valley_vals = data_array[valleys] if len(valleys) > 0 else []



    # ==========================================

        
    full_spectrum_weights = get_wtlm_weight(xf)
    wtd_ac_spectrum = ac_magnitudes * full_spectrum_weights
    
    # Calculate Primary/Aux JEITA/VESA
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

    # --- Channel Splitting & PDVM ---
    
    # Enforce a safe floor (400 Hz) to clear the fundamental harmonic splash zone
    aux_search_floor = max(split_freq, 400.0)
    
    high_freq_mask = xf > aux_search_floor
    low_freq_mask = xf <= split_freq
    freq_resolution = actual_fs / N
    
    primary_band_wtd_mags = wtd_ac_spectrum[low_freq_mask]
    
    # Generate Primary PAVM
    primary_pavm = calculate_pavm(xf, ac_magnitudes, mean_val, dominant_freq, target_visual_angle=24.0)
    
    # --- ROLLING RSS OVER PRIMARY SPECTRUM ---
    pri_rolling_window_hz = 50.0  # Adjust this bandwidth for the primary channel if needed
    pri_window_bins = max(1, int(pri_rolling_window_hz / freq_resolution))
    
    if len(primary_band_wtd_mags) < pri_window_bins:
        pri_window_bins = len(primary_band_wtd_mags)
        
    pri_rolling_sq_sum = np.convolve(primary_band_wtd_mags**2, np.ones(pri_window_bins), mode='valid')
    if len(pri_rolling_sq_sum) > 0:
        peak_pri_rolling_rss_amp = np.sqrt(np.max(pri_rolling_sq_sum))
    else:
        peak_pri_rolling_rss_amp = 0.0
        
    primary_wtd_pct = (peak_pri_rolling_rss_amp / mean_val) * 100.0 if mean_val > 0 else 0.0

    
    if np.any(high_freq_mask):
        aux_mags = ac_magnitudes[high_freq_mask]
        aux_peak_idx = np.argmax(aux_mags)
        aux_freq = xf[high_freq_mask][aux_peak_idx]
        aux_band_wtd_mags = wtd_ac_spectrum[high_freq_mask]
        
        # --- ROLLING RSS OVER AUXILIARY SPECTRUM ---
        aux_rolling_window_hz = 50.0  # Adjust this bandwidth if needed
        aux_window_bins = max(1, int(aux_rolling_window_hz / freq_resolution))
        
        # Protect against edge case where the high frequency array is smaller than the window
        if len(aux_band_wtd_mags) < aux_window_bins:
            aux_window_bins = len(aux_band_wtd_mags)
            
        # Convolve (sum) the squares across the sliding window, then find the max cluster
        aux_rolling_sq_sum = np.convolve(aux_band_wtd_mags**2, np.ones(aux_window_bins), mode='valid')
        if len(aux_rolling_sq_sum) > 0:
            peak_aux_rolling_rss_amp = np.sqrt(np.max(aux_rolling_sq_sum))
        else:
            peak_aux_rolling_rss_amp = 0.0
        
        aux_wtd_pct = (peak_aux_rolling_rss_amp / mean_val) * 100.0 if mean_val > 0 else 0.0
        
        # Generate Auxiliary PAVM
        aux_pavm = calculate_pavm(xf, ac_magnitudes, mean_val, aux_freq, target_visual_angle=24.0)
    else:
        aux_freq = aux_wtd_pct = aux_pavm = 0.0

    # --- Plotting ---
    plt.figure(figsize=(10, 5)) 
    
    plt.plot(time_array, data_array, color='red', linewidth=1.2)

    if (len(peaks) + len(valleys)) > 70:
        plt.plot(envelope_times, envelope_vals, "o", color='blue', markersize=4, label="Detected Peaks")
        plt.plot(time_array[valleys], valley_vals, "x", color='blue', markersize=4, label="Detected Valleys")
    
        
    
    plt.title('                                                   Sensor Waveform (DC & PWM)', fontsize=13, loc="left")
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Luminance (cd/m2)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    max_idx = min(250000, N - 1)
    plt.xlim(0, time_array[max_idx]) 
    
    metrics_text = (
        f"nits : {mean_val:.1f}\n"
        f"Dominant (PWM/DC/VCOM) Freq: {dominant_freq:.1f} Hz\n"
        f"Weighted TLM ratio flicker: {primary_wtd_pct:.2f}%\n"
        f"PDVM : {primary_pavm:.3f}\n"
        f"JEITA | VESA: {pri_jeita_str} | {pri_vesa_str}\n"
        f"--------------------------------------------\n"
        f"Sub-Harmonic Beating Freq: {loop_osc_freq:.1f} Hz\n"
        f"Beating modulation depth: {true_beating_depth:.2f}%\n"
        f"--------------------------------------------\n"
        f"Auxiliary (Harmonic) Freq: >{aux_freq:.1f} Hz\n"
        f"Weighted TLM Ratio flicker: {aux_wtd_pct:.2f}%\n"
        f"PDVM : {aux_pavm:.3f}\n"
        f"JEITA | VESA: {aux_jeita_str} | {aux_vesa_str}\n"
        )
    
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='gray')
    
    plt.gca().text(0.95, 0.95, metrics_text, transform=plt.gca().transAxes, 
                   fontsize=9, verticalalignment='bottom', horizontalalignment='right', 
                   bbox=props, weight='bold', color='black', family='monospace')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_waveform()
