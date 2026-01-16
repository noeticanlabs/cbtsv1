import json
import sys

def load_receipts(filename='receipts.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def condense_receipts(data):
    summary = {
        'total_steps': len(data),
        'final_t': data[-1]['t'] if data else 0,
        'constraint_residuals': {
            'eps_H_over_time': [],
            'eps_M_over_time': []
        },
        'loom_activity': {
            'D_band_over_time': [],
            'dt_loom_over_time': []
        },
        'tight_threads': {
            'over_time': []
        },
        'indicators': {
            'constraint_damping_success': [],
            'loom_reactivity_success': []
        }
    }

    for entry in data:
        t = entry['t']
        constraints = entry['constraints']
        eps_H = constraints['eps_H']
        eps_M = constraints['eps_M']

        summary['constraint_residuals']['eps_H_over_time'].append((t, eps_H))
        summary['constraint_residuals']['eps_M_over_time'].append((t, eps_M))

        loom = entry['loom']
        D_band = loom['D_band']
        dt_loom = loom['dt_loom']

        summary['loom_activity']['D_band_over_time'].append((t, D_band))
        summary['loom_activity']['dt_loom_over_time'].append((t, dt_loom))

        tight_threads = entry['tight_threads']
        summary['tight_threads']['over_time'].append((t, tight_threads))

    # Analyze trends
    trends = {
        'eps_H_decays': analyze_decay(summary['constraint_residuals']['eps_H_over_time']),
        'eps_M_decays': analyze_decay(summary['constraint_residuals']['eps_M_over_time']),
        'D_band_activity': describe_D_band(summary['loom_activity']['D_band_over_time']),
        'dt_loom_stability': analyze_stability(summary['loom_activity']['dt_loom_over_time']),
        'tight_threads_margins': analyze_margins(summary['tight_threads']['over_time'])
    }

    summary['trends'] = trends

    return summary

def analyze_decay(data):
    if not data:
        return "No data"
    initial = abs(data[0][1])
    final = abs(data[-1][1])
    if initial == 0:
        return "Constant at 0"
    decay_factor = final / initial
    return f"Decayed from {initial:.2e} to {final:.2e}, factor: {decay_factor:.2e}"

def describe_D_band(data):
    if not data:
        return "No data"
    # Check if D_band remains all zeros
    all_zero = all(all(val == 0 for val in entry[1]) for entry in data)
    if all_zero:
        return "All D_band values are 0 throughout the simulation, indicating no Loom activity."
    else:
        return "D_band values show non-zero activity in some bands."

def analyze_stability(data):
    if not data:
        return "No data"
    values = [entry[1] for entry in data]
    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        return f"Stable at {min_val}"
    else:
        return f"Ranges from {min_val} to {max_val}"

def analyze_margins(data):
    if not data:
        return "No data"
    # Extract margins for tight threads
    margins_over_time = {}
    for t, threads in data:
        for thread in threads:
            name, ratio, margin = thread
            if name not in margins_over_time:
                margins_over_time[name] = []
            margins_over_time[name].append((t, margin))
    description = {}
    for name, mdata in margins_over_time.items():
        initial_margin = mdata[0][1]
        final_margin = mdata[-1][1]
        description[name] = f"Margin from {initial_margin:.2e} to {final_margin:.2e}"
    return description

def extract_excerpt(data):
    excerpt = {
        "header": data[0],  # step 0
        "first_5_decay_onset": data[0:5],  # steps 0-4
        "around_10_15_tightening": data[10:15]  # steps 10-14
    }
    return excerpt

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'receipts.json'
    data = load_receipts(filename)
    excerpt = extract_excerpt(data)
    print(json.dumps(excerpt, indent=2))