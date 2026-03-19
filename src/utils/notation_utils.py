import re

from IPython.core.display import Math
from IPython.core.display_functions import display


def feature_to_physics_notation(feature_name, units = False, long=False):
    if "diff" in feature_name:
        base, lag = feature_name.replace('_diff', ''), None
        symbol = feature_to_physics_notation(base)
        return fr"\Delta {symbol}"
    if '_' in feature_name:
        base, lag = feature_name.rsplit('_', 1)
    else:
        base, lag = feature_name, None

    mapping = {
        '|avg B|': r'|B| (nT)' if units else r'|B|',
        'Flow Speed (km/s': r'v',
        'Flow pressure': r'P',
        'Temperature (K)': r'T',
        'Temperature': r'T',
        'AsyH': r'\mathrm{AsyH}',
        'Vx Velocity': r'V_x',
        'Vy Velocity': r'V_y',
        'Vz Velocity': r'V_z',
        'F10.7 (LASP)': r'F_{10.7}',
        'F30 (LASP)': r'F_{30}',
        'ap (LASP)': r'a_p',
        'Kp (LASP)': r'K_p',
        'SymD (Omni)': r'\mathrm{SymD}',
        'SymH (Omni)': r'\mathrm{SymH}',
        'AsyD (Omni)': r'\mathrm{AsyD}',
        'By GSE': r'B_y',
        'Bx GSE': r'B_x',
        'Bz GSE': r'B_z',
        'By GSM': r'B_y',
        'Bz GSM': r'B_z',
        'Proton density (n/cc)': r'\rho_p',
        'Magnetosonic mach number': r'M_{ms}',
        'Alfven mach number': r'M_A',
        'Plasma beta': r'\beta',
        'Electric Field (Mv/m)': r'E',
        'Percent Interpolated': r'\mathrm{pInterp}',
        '# fine scale Plasma points': r'N_\mathrm{plasma}',
        '# fine scale IMF points' : r'N_\mathrm{IMF}',
        'RMS SD B vector (nT)' : r'\mathrm{RMS}_B',
        'RMS SD B scalar (nT)' : r'\mathrm{RMS}_{|B|}',
        "Timeshift (seconds)": r'\Delta_\mathrm{bow} t',
        "Time between observations (seconds)": r'\Delta_\mathrm{obs} t',
        "RMS Timeshift (seconds)": r'\mathrm{RMS}_{\Delta_\mathrm{bow} t}',
    }
    symbol = None
    for key in mapping:
        if base.startswith(key):
            symbol = mapping[key]
            break

    if symbol is None:
        symbol = base

    if lag is not None:
        # lag = str(int(lag)) # convert to minutes given that lag=1 means 30 seconds
        if '_' in symbol:
            symbol = re.sub(r'_(\w+)$', r'_{\1,t-' + lag + '}', symbol)
        else:
            symbol = fr"{symbol}_{{t-{lag}}}"

    return symbol

def format_lasso_equation(coef_df, intercept, terms_per_line=3):
    lines = []
    current_line = []

    sort_by_column = 'coef_normalized'

    coef_df = coef_df.sort_values(by=sort_by_column, ascending=False).reset_index(drop=True)
    feature_names = coef_df['notation'].values
    coefficients = coef_df['coef_signed'].values

    for coef, name in zip(coefficients, feature_names):
        if abs(coef) < 1e-6:
            continue

        sign = '+' if coef > 0 else '-'
        coef_abs = abs(coef)

        term = f"{sign} {coef_abs:.3g} {name}"

        current_line.append(term)
        if len(current_line) >= terms_per_line:
            lines.append(" ".join(current_line))
            current_line = []

    if current_line:
        lines.append(" ".join(current_line))

    latex_lines = [f"\\hat{{y}}_t = {intercept:.2f} "] +[f"& {line}" for line in lines]
    equation = r"$$\begin{aligned}" + "\n" + "\\\\\n".join(latex_lines) + "\n" + r"\end{aligned}$$"

    return equation

def pretty_print_lasso_equation(coef_df, intercept, terms_per_line=3):
    equation = format_lasso_equation(coef_df, intercept, terms_per_line)
    display(Math(equation))
    return equation
