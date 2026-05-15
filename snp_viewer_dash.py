"""
SNP Viewer - Dash 版本
完整移植自 Streamlit 版，功能對等：
- 多檔案上傳與疊圖
- S-parameter 選擇 (S11/S21/S12/S22...)
- 資料類型選擇 (Magnitude / Phase / Smith Chart)
- 頻率範圍篩選
- Amplitude 範圍控制
- Marker (垂直/水平/十字) + 自訂樣式
- Threshold Marker + 交叉點計算
- 檔案顯示/隱藏勾選
- 統計表格 / Marker 數值表 / 原始資料表
- 線條寬度控制
"""

import base64
import io
import json
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import skrf as rf
from dash import (
    ALL,
    Dash,
    Input,
    Output,
    State,
    callback,
    ctx,
    dash_table,
    dcc,
    html,
    no_update,
)
from dash.exceptions import PreventUpdate

warnings.filterwarnings("ignore", message=".*keyword arguments have been deprecated.*")

# ============================================================
# Version
# ============================================================
APP_VERSION = "1.1.0"

# v1.0.4  2026-05-07
#   - feat: Smith marker click-to-set — click on Smith curve to move a marker
#   - feat: 🎯 toggle button per Smith marker (click-to-set target); default moves last marker
#   - ui: 🎯 button enlarged (22px), hover scale + blue glow effect
#   - ui: del buttons (all 3 marker types) hover effect — red darken + scale
#   - fix: 🎯 button focus outline removed (black border on deactivate)
#
# v1.0.3  2026-05-07
#   - feat: Smith marker table redesigned — columns: Marker / File / Freq / Z₀(Ω) / Γ / Z(Ω) / Y(mS)
#   - feat: Γ, Z, Y displayed as complex strings "a ± jb" (2 decimal places)
#   - feat: Z₀ read from SNP file header (network.z0); fallback to 50 Ω
#   - feat: hover tooltip updated to show Z₀, Γ, Z, Y in complex format
#
# v1.0.2  2026-05-06
#   - feat: threshold Peak / -3 dB / Dip preset buttons
#   - fix: param type switch now preserves data type selection
#   - fix: threshold preset value uses exact precision (no rounding loss)
#   - fix: marker/threshold input step="any" allows arbitrary decimals
#   - fix: pd.read_json suppress date-parse UserWarning (convert_dates=False)
#   - ui: preset buttons hover effect, equal-width fill sidebar
#
# v1.0.1  2026-05-06
#   - fix: threshold crossings now respect frequency range filter
#   - ui: update empty-state message
#
# v1.0.0  2026-05-05
#   - migrated from Streamlit to Dash
#   - freq / amplitude range filter
#   - vertical markers + threshold markers with drag support
#   - Smith Chart support
#   - deployed on Render

# ============================================================
# App 初始化
# ============================================================
app = Dash(
    __name__,
    title="SNP Viewer",
    suppress_callback_exceptions=True,  # 動態元件必須設定
)
server = app.server  # for deployment (gunicorn)

# ============================================================
# 常數
# ============================================================
DEFAULT_MARKER_COLORS = [
    "#FF0000", "#00AA00", "#0000FF", "#FF00FF", "#CC8800",
    "#00CCCC", "#FF6600", "#800080", "#CC0066", "#A52A2A",
]
COLOR_PALETTE = px.colors.qualitative.Plotly + px.colors.qualitative.Bold

_COLOR_OPTIONS_MAP = {
    "#FF0000": "🔴 Red",       "#00AA00": "🟢 Green",
    "#0000FF": "🔵 Blue",      "#FF00FF": "🟣 Magenta",
    "#CC8800": "🟠 Orange",    "#00CCCC": "🩵 Cyan",
    "#FF6600": "🟧 Dk.Orange", "#800080": "🟤 Purple",
    "#CC0066": "🌸 Pink",      "#A52A2A": "🟫 Brown",
    "#000000": "⚫ Black",     "#888888": "⬜ Gray",
}
COLOR_DROPDOWN_OPTIONS = [{"label": v, "value": k} for k, v in _COLOR_OPTIONS_MAP.items()]

def color_dropdown(component_id, default_index: int = 0):
    return dcc.Dropdown(
        id=component_id,
        options=COLOR_DROPDOWN_OPTIONS,
        value=DEFAULT_MARKER_COLORS[default_index % len(DEFAULT_MARKER_COLORS)],
        clearable=False,
        style={"fontSize": "12px"},
    )


# ============================================================
# 純計算函數（與 Streamlit 版完全相同，直接搬移）
# ============================================================

def fmt_complex(c: complex, decimals: int = 2) -> str:
    sign = "+" if c.imag >= 0 else "-"
    return f"{c.real:.{decimals}f} {sign} j{abs(c.imag):.{decimals}f}"


def parse_sparameter_file(filename: str, file_content: str) -> tuple[pd.DataFrame, float]:
    """解析 Touchstone 檔案，回傳 (DataFrame, z0)。z0 從檔案讀取，讀不到預設 50.0。"""
    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name
    try:
        network = rf.Network(tmp_path)
    finally:
        os.remove(tmp_path)

    try:
        z0 = float(np.real(network.z0[0, 0]))
    except Exception:
        z0 = 50.0

    freq = network.frequency.f
    n_ports = network.number_of_ports
    data: dict = {"Frequency": freq}
    for i in range(n_ports):
        for j in range(n_ports):
            s = network.s[:, i, j]
            name = f"S{i+1}{j+1}"
            data[f"{name}_mag"]   = 20 * np.log10(np.abs(s) + 1e-12)
            data[f"{name}_phase"] = np.angle(s, deg=True)
            data[f"{name}_real"]  = np.real(s)
            data[f"{name}_imag"]  = np.imag(s)
    return pd.DataFrame(data), z0


def get_available_parameters(df: pd.DataFrame) -> list[str]:
    param_types: set[str] = set()
    for col in df.columns:
        if col != "Frequency" and "_" in col:
            param_types.add(col.split("_")[0])
    return sorted(param_types)


def get_data_types(df: pd.DataFrame, param_type: str) -> list[str]:
    data_types: set[str] = set()
    for col in df.columns:
        if col.startswith(param_type + "_"):
            dt = col.split("_")[1]
            if dt not in ("real", "imag"):
                data_types.add(dt)
    if f"{param_type}_real" in df.columns and f"{param_type}_imag" in df.columns:
        data_types.add("smith")
    return sorted(data_types)


def auto_convert_frequency(df: pd.DataFrame, return_unit_factor=False):
    freq = df["Frequency"].copy()
    max_f = freq.max()
    if max_f >= 1e9:
        factor, unit = 1e9, "GHz"
    elif max_f >= 1e6:
        factor, unit = 1e6, "MHz"
    elif max_f >= 1e3:
        factor, unit = 1e3, "kHz"
    else:
        factor, unit = 1, "Hz"
    df2 = df.copy()
    df2["Frequency"] = freq / factor
    if return_unit_factor:
        return df2, unit, factor
    return df2, unit


def get_display_name(filename: str) -> str:
    return os.path.splitext(filename)[0]


def get_y_axis_unit(data_type: str) -> str:
    if "mag" in data_type.lower():
        return "dB"
    if "phase" in data_type.lower():
        return "deg"
    return ""


def filter_by_frequency_range(df: pd.DataFrame, fmin: float, fmax: float) -> pd.DataFrame:
    mask = (df["Frequency"] >= fmin) & (df["Frequency"] <= fmax)
    return df[mask].copy()


def find_nearest_value(df: pd.DataFrame, freq_target: float, col: str):
    idx = (df["Frequency"] - freq_target).abs().idxmin()
    return df.loc[idx, "Frequency"], df.loc[idx, col]


def find_threshold_crossings(df: pd.DataFrame, col: str, threshold: float) -> list[float]:
    freqs  = df["Frequency"].values
    values = df[col].values
    crossings = []
    for i in range(len(values) - 1):
        v0 = values[i] - threshold
        v1 = values[i + 1] - threshold
        if v0 * v1 <= 0 and v0 != v1:
            t = -v0 / (v1 - v0)
            crossings.append(float(freqs[i] + t * (freqs[i + 1] - freqs[i])))
    return crossings


# ============================================================
# 圖表建構（邏輯與 Streamlit 版相同）
# ============================================================

def build_base_figure(
    files_data: dict,
    visible_filenames: list[str],
    selected_param: str,
    selected_data_type: str,
    freq_range: tuple,
    amp_range: tuple,
    threshold_markers_list: list,
    line_width: int,
    num_markers: int = 0,
) -> tuple[go.Figure, dict, int, dict]:
    """
    只畫主曲線 + threshold，不畫 marker overlay。
    回傳 (fig, threshold_crossings, base_trace_count, threshold_shape_map)
    threshold_shape_map = { str(shape_index): threshold_id_str }
    """
    selected_param_full = f"{selected_param}_{selected_data_type}"
    y_axis_unit = get_y_axis_unit(selected_data_type)
    is_smith = selected_data_type == "smith"

    visible_files = {k: files_data[k] for k in visible_filenames if k in files_data}

    if visible_files:
        first_df = list(visible_files.values())[0]["df"]
        _, freq_unit, _ = auto_convert_frequency(first_df, return_unit_factor=True)
    else:
        freq_unit = "GHz"

    fig = go.Figure()
    threshold_crossings: dict = {}

    # ── 主線 ──────────────────────────────────────────────
    for idx, (filename, file_info) in enumerate(visible_files.items()):
        df = file_info["df"]
        df_conv, _ = auto_convert_frequency(df)
        df_filt = filter_by_frequency_range(df_conv, freq_range[0], freq_range[1])
        display_name = get_display_name(filename)
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]

        if is_smith:
            if f"{selected_param}_real" not in df_filt.columns:
                continue
            gamma = (df_filt[f"{selected_param}_real"].values
                     + 1j * df_filt[f"{selected_param}_imag"].values)
            z_norm = (1 + gamma) / (1 - gamma + 1e-30)
            fig.add_trace(go.Scattersmith(
                real=np.real(z_norm).tolist(),
                imag=np.imag(z_norm).tolist(),
                customdata=df_filt["Frequency"].tolist(),
                mode="lines",
                name=display_name,
                line=dict(color=color, width=line_width),
                hovertemplate=(
                    f"<b>{display_name}</b><br>"
                    f"Freq: %{{customdata:.3f}} {freq_unit}<br>"
                    "<extra></extra>"
                ),
            ))
        else:
            if selected_param_full not in df_filt.columns:
                continue
            show_line_legend = (num_markers == 0)
            fig.add_trace(go.Scatter(
                x=df_filt["Frequency"],
                y=df_filt[selected_param_full],
                mode="lines",
                name=display_name,
                line=dict(color=color, width=line_width),
                marker=dict(size=4),
                showlegend=show_line_legend,
                hovertemplate=(
                    f"<b>{display_name}</b><br>"
                    f"Frequency: %{{x:.3f}} {freq_unit}<br>"
                    f"{selected_param_full}: %{{y:.3f}} {y_axis_unit}<br>"
                    "<extra></extra>"
                ),
            ))

    # ── Threshold Markers ─────────────────────────────────
    threshold_shape_map: dict = {}   # shape_index → threshold id str
    if not is_smith:
        for t_idx, marker in enumerate(threshold_markers_list):
            t_val   = marker["value"]
            t_label = marker["label"]
            t_color = marker["color"]
            t_show  = marker.get("show_in_legend", True)
            t_id_str = marker.get("id_str", str(t_idx + 1))
            threshold_crossings[t_label] = {}

            # 記錄此 threshold 對應的 shape index（fig.layout.shapes 目前長度）
            shape_idx = len(fig.layout.shapes)
            threshold_shape_map[str(shape_idx)] = t_id_str

            # 水平線 shape，加上 editable=True 讓使用者可拖動
            fig.add_shape(
                type="line",
                x0=0, x1=1,
                y0=t_val, y1=t_val,
                xref="paper", yref="y",
                line=dict(color=t_color, dash="dash", width=2),
                opacity=0.8,
                editable=True,
            )
            # 標籤固定在圖框右端，水平顯示
            fig.add_annotation(
                x=0,
                y=t_val,
                xref="paper",
                yref="y",
                text=t_label,
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                xshift=6,
                yshift=4,
                font=dict(size=10, color=t_color),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=t_color,
                borderwidth=1,
                borderpad=3,
            )

            for idx, (filename, file_info) in enumerate(visible_files.items()):
                df_conv, _ = auto_convert_frequency(file_info["df"])
                if selected_param_full not in df_conv.columns:
                    continue
                df_filt_thresh = filter_by_frequency_range(df_conv, freq_range[0], freq_range[1])
                crossings = find_threshold_crossings(df_filt_thresh, selected_param_full, t_val)
                display_name = get_display_name(filename)
                threshold_crossings[t_label][display_name] = crossings
                line_color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
                for c_freq in crossings:
                    legend_text = f"{display_name} ✕ {t_label}: {c_freq:.3f} {freq_unit}"
                    fig.add_trace(go.Scatter(
                        x=[c_freq], y=[t_val],
                        mode="markers",
                        marker=dict(
                            size=12, color=line_color, symbol="x",
                            line=dict(width=2, color=t_color),
                        ),
                        name=legend_text,
                        legendgroup=f"threshold_{t_label}",
                        showlegend=t_show,
                        hovertemplate=(
                            f"<b>{t_label}</b><br>{display_name}<br>"
                            f"Frequency: {c_freq:.3f} {freq_unit}<br>"
                            f"Threshold: {t_val:.1f} {y_axis_unit}<br>"
                            "<extra></extra>"
                        ),
                    ))

    base_trace_count = len(fig.data)

    # ── Layout ────────────────────────────────────────────
    legend_style = dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(200,200,200,0.5)",
        borderwidth=1,
        orientation="v",
        yanchor="top", y=1,
        xanchor="left", x=1.02,
        itemsizing="constant",
        tracegroupgap=0,
    )

    if is_smith:
        fig.update_layout(
            title=f"{selected_param} Smith Chart",
            height=600,
            showlegend=True,
            legend=legend_style,
            margin=dict(t=40, b=40, l=80, r=150),
        )
    else:
        y_title = f"{selected_param_full} ({y_axis_unit})" if y_axis_unit else selected_param_full
        y_min = amp_range[0]
        y_max = amp_range[1]
        if threshold_markers_list:
            y_min = min(y_min, min(m["value"] for m in threshold_markers_list))
            y_max = max(y_max, max(m["value"] for m in threshold_markers_list))

        fig.update_layout(
            title=selected_param_full,
            xaxis_title=f"Freq ({freq_unit})",
            yaxis_title=y_title,
            plot_bgcolor="rgba(240,240,240,0.5)",
            height=600,
            showlegend=True,
            legend=legend_style,
            margin=dict(t=40, b=40, l=80, r=150),
            xaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.3)", gridwidth=1),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(200,200,200,0.3)",
                gridwidth=1,
                range=[y_min, y_max],
            ),
        )

    return fig, threshold_crossings, base_trace_count, threshold_shape_map


def compute_marker_overlay(
    files_data: dict,
    visible_filenames: list[str],
    selected_param: str,
    selected_data_type: str,
    markers_list: list,
) -> tuple[list, list, list, dict, dict]:
    """
    計算 marker 讀值 trace、shape 和 annotation，不需要重畫曲線。
    回傳 (overlay_traces, shapes, annotations, marker_values, shape_index_map)
    """
    selected_param_full = f"{selected_param}_{selected_data_type}"
    y_axis_unit = get_y_axis_unit(selected_data_type)
    is_smith = selected_data_type == "smith"

    visible_files = {k: files_data[k] for k in visible_filenames if k in files_data}

    if visible_files:
        first_df = list(visible_files.values())[0]["df"]
        _, freq_unit, _ = auto_convert_frequency(first_df, return_unit_factor=True)
    else:
        freq_unit = "GHz"

    overlay_traces = []
    shapes         = []
    annotations    = []
    marker_values: dict     = {}
    shape_index_map: dict   = {}

    if is_smith or not markers_list:
        return overlay_traces, shapes, annotations, marker_values, shape_index_map

    for marker_idx, marker in enumerate(markers_list):
        m_freq  = marker["freq"]
        m_label = marker["label"]
        m_color = marker["color"]
        m_show  = marker.get("show_in_legend", True)

        marker_values[m_label] = []

        shape_index_map[len(shapes)] = marker_idx
        shapes.append(dict(
            type="line",
            x0=m_freq, x1=m_freq,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color=m_color, dash="dash", width=2),
            opacity=0.8,
            editable=True,
        ))

        # 標籤固定在圖框頂端，水平顯示，不隨線旋轉
        annotations.append(dict(
            x=m_freq,
            y=1,
            xref="x",
            yref="paper",
            text=m_label,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            yshift=4,
            font=dict(size=10, color=m_color),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=m_color,
            borderwidth=1,
            borderpad=3,
        ))

        for idx, (filename, file_info) in enumerate(visible_files.items()):
            df_conv, _ = auto_convert_frequency(file_info["df"])
            if selected_param_full not in df_conv.columns:
                continue
            actual_freq, actual_val = find_nearest_value(df_conv, m_freq, selected_param_full)
            display_name = get_display_name(filename)
            marker_values[m_label].append((display_name, actual_freq, actual_val))
            line_color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
            legend_text = f"{display_name} @ {m_label} {freq_unit}: {actual_val:.3f} {y_axis_unit}"
            x_coords = [actual_freq - 0.001, actual_freq, actual_freq + 0.001]
            y_coords = [actual_val] * 3
            overlay_traces.append(go.Scatter(
                x=x_coords, y=y_coords,
                mode="lines+markers",
                line=dict(color=m_color, width=2, dash="dash"),
                marker=dict(
                    size=[0, 12, 0],
                    color=line_color,
                    symbol="diamond",
                    line=dict(width=2, color="white"),
                ),
                name=legend_text,
                legendgroup=f"marker_{m_label}",
                showlegend=m_show,
                hovertemplate=(
                    f"<b>{m_label}</b><br>{display_name}<br>"
                    f"Frequency: {actual_freq:.3f} {freq_unit}<br>"
                    f"Value: {actual_val:.3f} {y_axis_unit}<br>"
                    "<extra></extra>"
                ),
            ))

    return overlay_traces, shapes, annotations, marker_values, shape_index_map


def build_smith_marker_overlays(
    files_data: dict,
    visible_filenames: list[str],
    selected_param: str,
    smith_markers_list: list,
) -> tuple[list, dict]:
    """
    為每個 smith marker × visible file 建立 Scattersmith dot trace。
    回傳 (overlay_traces, marker_values)
    marker_values: {label: [(display_name, actual_freq, gamma_r, gamma_i, z_r, z_i), ...]}
    """
    if not smith_markers_list:
        return [], {}

    visible_files = {k: files_data[k] for k in visible_filenames if k in files_data}
    if not visible_files:
        return [], {}

    first_df = list(visible_files.values())[0]["df"]
    _, freq_unit, _ = auto_convert_frequency(first_df, return_unit_factor=True)

    overlay_traces = []
    marker_values: dict = {}

    for marker in smith_markers_list:
        m_freq  = marker["freq"]
        m_label = marker["label"]
        m_color = marker["color"]
        short_label = m_label.split(":")[0]   # e.g. "M1"
        marker_values[m_label] = []

        for idx, (filename, file_info) in enumerate(visible_files.items()):
            df_conv, _ = auto_convert_frequency(file_info["df"])
            real_col = f"{selected_param}_real"
            imag_col = f"{selected_param}_imag"
            if real_col not in df_conv.columns or imag_col not in df_conv.columns:
                continue

            z0 = float(file_info.get("z0", 50.0))

            row_idx = (df_conv["Frequency"] - m_freq).abs().idxmin()
            actual_freq = float(df_conv.loc[row_idx, "Frequency"])
            actual_real = float(df_conv.loc[row_idx, real_col])
            actual_imag = float(df_conv.loc[row_idx, imag_col])

            gamma = actual_real + 1j * actual_imag
            z     = z0 * (1 + gamma) / (1 - gamma)
            y     = (1 / z) * 1000  # mS

            display_name = get_display_name(filename)
            marker_values[m_label].append(
                (display_name, actual_freq, z0, gamma, z, y)
            )

            overlay_traces.append(go.Scattersmith(
                real=[float(np.real(z / z0))],
                imag=[float(np.imag(z / z0))],
                customdata=[[
                    actual_freq,
                    float(np.real(gamma)), float(np.imag(gamma)),
                    float(np.real(z)),     float(np.imag(z)),
                    float(np.real(y)),     float(np.imag(y)),
                    z0,
                ]],
                mode="markers+text",
                text=[short_label],
                textposition="top center",
                textfont=dict(color=m_color, size=12),
                marker=dict(
                    color=m_color,
                    size=10,
                    symbol="circle",
                    line=dict(color="white", width=1.5),
                ),
                name=f"{display_name} @ {m_label}",
                showlegend=False,
                hovertemplate=(
                    f"<b>{m_label}</b><br>"
                    f"<b>{display_name}</b><br>"
                    f"Freq: %{{customdata[0]:.3f}} {freq_unit}<br>"
                    f"Z₀: %{{customdata[7]:.0f}} Ω<br>"
                    f"Γ: %{{customdata[1]:.2f}} %{{customdata[2]:+.2f}}j<br>"
                    f"Z: %{{customdata[3]:.2f}} %{{customdata[4]:+.2f}}j Ω<br>"
                    f"Y: %{{customdata[5]:.2f}} %{{customdata[6]:+.2f}}j mS<br>"
                    "<extra></extra>"
                ),
            ))

    return overlay_traces, marker_values


def build_smith_marker_table(marker_values: dict, visible_filenames: list, freq_unit: str) -> pd.DataFrame:
    """marker_values 轉成讀值 DataFrame（欄位：Marker, File, Freq, Z₀, Γ, Z, Y）"""
    rows = []
    for label, entries in marker_values.items():
        for (display_name, actual_freq, z0, gamma, z, y) in entries:
            rows.append({
                "Marker":            label,
                "File":              display_name,
                f"Freq ({freq_unit})": round(actual_freq, 4),
                "Z₀ (Ω)":           int(round(z0)),
                "Γ":                 fmt_complex(gamma),
                "Z (Ω)":             fmt_complex(z),
                "Y (mS)":            fmt_complex(y),
            })
    return pd.DataFrame(rows)


# 保留舊名稱作為 wrapper，方便測試相容
def build_figure(
    files_data, visible_filenames, selected_param, selected_data_type,
    freq_range, amp_range, markers_list, threshold_markers_list, line_width,
):
    fig, threshold_crossings, base_trace_count, threshold_shape_map = build_base_figure(
        files_data, visible_filenames, selected_param, selected_data_type,
        freq_range, amp_range, threshold_markers_list, line_width,
        num_markers=len(markers_list),
    )
    overlay_traces, shapes, annotations, marker_values, shape_index_map = compute_marker_overlay(
        files_data, visible_filenames, selected_param, selected_data_type, markers_list,
    )
    for trace in overlay_traces:
        fig.add_trace(trace)
    for shape in shapes:
        fig.add_shape(**shape)
    for ann in annotations:
        fig.add_annotation(**ann)
    return fig, marker_values, threshold_crossings, shape_index_map


# ============================================================
# 統計 / Marker / Threshold 表格
# ============================================================

def build_stats_table(files_data, visible_filenames, selected_param_full, freq_range):
    rows = []
    for fname in visible_filenames:
        if fname not in files_data:
            continue
        df_conv, _ = auto_convert_frequency(files_data[fname]["df"])
        df_filt = filter_by_frequency_range(df_conv, freq_range[0], freq_range[1])
        if selected_param_full not in df_filt.columns:
            continue
        data = df_filt[selected_param_full]
        rows.append({
            "File Name": get_display_name(fname),
            "Max":        f"{data.max():.3f}",
            "Min":        f"{data.min():.3f}",
            "Mean":       f"{data.mean():.3f}",
            "Range":      f"{(data.max() - data.min()):.3f}",
            "Data Points": len(data),
        })
    return pd.DataFrame(rows)


def build_marker_table(marker_values, visible_filenames, freq_unit, y_axis_unit):
    table: dict = {}
    for fname in visible_filenames:
        dn = get_display_name(fname)
        table[dn] = {"File": dn}
    for m_label, values in marker_values.items():
        col = f"Value @ {m_label} {freq_unit} ({y_axis_unit})"
        for dn, actual_freq, actual_val in values:
            if dn in table:
                table[dn][col] = f"{actual_val:.3f}"
    return pd.DataFrame(list(table.values())).set_index("File").reset_index()


def build_threshold_table(threshold_crossings, visible_filenames, freq_unit):
    table: dict = {}
    for fname in visible_filenames:
        dn = get_display_name(fname)
        table[dn] = {"File": dn}
    for t_label, file_crossings in threshold_crossings.items():
        col = f"Crossing Freq @ {t_label} ({freq_unit})"
        for dn, crossings in file_crossings.items():
            if dn in table:
                table[dn][col] = (", ".join(f"{f:.3f}" for f in crossings) if crossings else "—")
    return pd.DataFrame(list(table.values())).set_index("File").reset_index()


# ============================================================
# Dash Table helper
# ============================================================

def df_to_dash_table(df: pd.DataFrame, table_id: str):
    return dash_table.DataTable(
        id=table_id,
        columns=[{"name": c, "id": c} for c in df.columns],
        data=df.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px 10px", "fontSize": "13px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"}
        ],
    )


# ============================================================
# Callback shared helpers
# ============================================================

def _parse_files(files_data: dict) -> dict:
    return {
        fn: {"df": pd.read_json(io.StringIO(fdata["df_json"]), orient="split", convert_dates=False)}
        for fn, fdata in files_data.items()
    }


def _build_markers_list(m_freqs, m_colors, m_shows, store, unit: str) -> list:
    store_ids = list((store or {}).keys())
    result = []
    for i, freq_val in enumerate(m_freqs or []):
        if freq_val is None:
            continue
        m_id_str = store_ids[i] if i < len(store_ids) else str(i + 1)
        result.append({
            "freq":           freq_val,
            "label":          f"M{m_id_str}: {freq_val:.3f} {unit}",
            "color":          m_colors[i] if i < len(m_colors) and m_colors[i] else DEFAULT_MARKER_COLORS[i % 10],
            "style":          "vertical",
            "show_in_legend": bool(m_shows[i]) if m_shows and i < len(m_shows) else True,
        })
    return result


def _build_threshold_markers_list(t_values, t_colors, t_shows, store, y_unit: str) -> list:
    store_ids = list((store or {}).keys())
    result = []
    for i, t_val in enumerate(t_values or []):
        if t_val is None:
            continue
        t_id_str = store_ids[i] if i < len(store_ids) else str(i + 1)
        result.append({
            "value":          t_val,
            "label":          f"T{t_id_str}: {t_val:.1f} {y_unit}",
            "id_str":         t_id_str,
            "color":          t_colors[i] if i < len(t_colors) and t_colors[i] else DEFAULT_MARKER_COLORS[i % 10],
            "show_in_legend": bool(t_shows[i]) if t_shows and i < len(t_shows) else True,
        })
    return result


def _build_smith_markers_list(sm_freqs, sm_colors, store, unit: str) -> list:
    store_ids = list((store or {}).keys())
    result = []
    for i, freq_val in enumerate(sm_freqs or []):
        if freq_val is None:
            continue
        sm_id_str = store_ids[i] if i < len(store_ids) else str(i + 1)
        result.append({
            "freq":  freq_val,
            "label": f"M{sm_id_str}: {freq_val:.3f} {unit}",
            "color": sm_colors[i] if i < len(sm_colors) and sm_colors[i] else DEFAULT_MARKER_COLORS[i % 10],
        })
    return result


def _compute_threshold_crossings(
    threshold_markers_list: list,
    parsed_files: dict,
    selected_param_full: str,
    fmin: float,
    fmax: float,
) -> dict:
    crossings: dict = {}
    for t_marker in threshold_markers_list:
        t_label = t_marker["label"]
        crossings[t_label] = {}
        for fname, finfo in parsed_files.items():
            df_conv, _ = auto_convert_frequency(finfo["df"])
            if selected_param_full not in df_conv.columns:
                continue
            df_filt = filter_by_frequency_range(df_conv, fmin, fmax)
            crossings[t_label][get_display_name(fname)] = find_threshold_crossings(
                df_filt, selected_param_full, t_marker["value"]
            )
    return crossings


def _get_threshold_table(threshold_markers_list, threshold_crossings, visible_filenames, unit, cell_idx):
    if threshold_markers_list and threshold_crossings:
        thresh_df = build_threshold_table(threshold_crossings, visible_filenames, unit)
        return df_to_dash_table(thresh_df, f"table-thresholds-{cell_idx}")
    return None


def _get_stats_table(parsed_files, visible_filenames, selected_param_full, fmin, fmax, cell_idx):
    stats_df = build_stats_table(parsed_files, visible_filenames, selected_param_full, (fmin, fmax))
    if not stats_df.empty:
        return df_to_dash_table(stats_df, f"table-stats-{cell_idx}")
    return None


# ============================================================
# Layout
# ============================================================

SIDEBAR_STYLE = {
    "width": "300px",
    "minWidth": "300px",
    "padding": "16px",
    "backgroundColor": "#f8f9fa",
    "borderRight": "1px solid #dee2e6",
    "overflowY": "auto",
    "height": "100vh",
    "position": "sticky",
    "top": 0,
    "transition": "width 0.2s ease, min-width 0.2s ease, padding 0.2s ease",
}

SIDEBAR_COLLAPSED_STYLE = {
    "width": "0",
    "minWidth": "0",
    "padding": "0",
    "backgroundColor": "#f8f9fa",
    "borderRight": "none",
    "overflow": "hidden",
    "height": "100vh",
    "position": "sticky",
    "top": 0,
    "transition": "width 0.2s ease, min-width 0.2s ease, padding 0.2s ease",
}

CONTENT_STYLE = {
    "flex": "1",
    "padding": "20px",
    "overflowY": "auto",
    "minWidth": 0,
}

SECTION_STYLE = {
    "marginBottom": "8px",
    "fontWeight": "600",
    "fontSize": "13px",
    "color": "#495057",
    "borderBottom": "1px solid #dee2e6",
    "paddingBottom": "4px",
    "marginTop": "16px",
}

INPUT_STYLE = {
    "width": "100%",
    "padding": "4px 8px",
    "border": "1px solid #ced4da",
    "borderRadius": "4px",
    "fontSize": "13px",
}

BTN_LAYOUT_ACTIVE = {
    "padding": "4px 10px", "border": "1px solid #495057",
    "borderRadius": "4px", "backgroundColor": "#495057", "color": "white",
    "cursor": "pointer", "fontSize": "12px", "fontWeight": "600",
}
BTN_LAYOUT_INACTIVE = {
    "padding": "4px 10px", "border": "1px solid #ced4da",
    "borderRadius": "4px", "backgroundColor": "#fff", "color": "#495057",
    "cursor": "pointer", "fontSize": "12px",
}

LAYOUT_GRID = {
    "1x1": {"cols": 1, "rows": 1, "cells": 1, "graph_height": 580},
    "1x2": {"cols": 2, "rows": 1, "cells": 2, "graph_height": 540},
    "2x1": {"cols": 1, "rows": 2, "cells": 2, "graph_height": 340},
    "2x2": {"cols": 2, "rows": 2, "cells": 4, "graph_height": 340},
}

def label(text):
    return html.Label(text, style={"fontSize": "12px", "color": "#6c757d", "marginBottom": "2px", "display": "block"})

def section(text):
    return html.Div(text, style=SECTION_STYLE)


app.layout = html.Div([
    # ── dcc.Store：所有狀態存在 browser session ──────────────
    dcc.Store(id="store-files-data",      storage_type="memory", data={}),
    dcc.Store(id="store-freq-meta",       storage_type="memory", data={}),
    dcc.Store(id="store-markers",         storage_type="memory", data=[]),
    dcc.Store(id="store-thresholds",      storage_type="memory", data=[]),
    dcc.Store(id="store-resize-init",     data=1),
    dcc.Store(id="store-layout",             storage_type="memory", data="1x1"),
    dcc.Store(id="store-graph-height",       storage_type="memory", data=None),
    dcc.Store(id="store-cell-configs",       storage_type="memory", data=[
        {"param": None, "data_type": None},
        {"param": None, "data_type": None},
        {"param": None, "data_type": None},
    ]),
    dcc.Store(id="store-shape-index-map",           storage_type="memory", data={}),
    dcc.Store(id="store-threshold-shape-map",       storage_type="memory", data={}),
    dcc.Store(id="store-visible-files",   storage_type="memory", data=[]),
    dcc.Store(id="store-base-trace-count",       storage_type="memory", data=[0, 0, 0, 0]),
    dcc.Store(id="store-threshold-annotations",  storage_type="memory", data=[]),
    dcc.Store(id="store-upload-queue",           storage_type="memory", data=None),
    dcc.Store(id="store-setup-done",             data=0),
    dcc.Store(id="store-active-smith-marker",    storage_type="memory", data=None),
    dcc.Interval(id="interval-upload-check", interval=300, n_intervals=0),

    # ── 頁面主體 ──────────────────────────────────────────────
    html.Div([

        # ════ 側邊欄 ════
        html.Div([
            html.H2("⚙️ Settings", style={"fontSize": "16px", "marginBottom": "12px", "marginTop": 0}),

            # 上傳
            section("📎 Upload Files"),
            html.Div([
                html.Div([
                    "Drag & Drop or ",
                    html.Span("Browse .snp files", style={"color": "#0d6efd"}),
                ]),
            ], id="upload-drop-zone", style={
                "width": "100%", "boxSizing": "border-box",
                "border": "2px dashed #ced4da",
                "borderRadius": "8px", "textAlign": "center",
                "padding": "16px 8px", "cursor": "pointer",
                "fontSize": "13px", "backgroundColor": "#fff",
            }),
            html.Div(id="uploaded-files-list", style={
                "marginTop": "8px",
                "maxHeight": "220px",
                "overflowY": "auto",
            }),
            html.Div(id="upload-error", style={"color": "red", "fontSize": "12px", "marginTop": "4px"}),

            # 選擇 Parameter
            section("📈 Select Parameters"),
            label("Parameter Type"),
            dcc.Dropdown(id="dd-param", clearable=False, style={"fontSize": "13px", "marginBottom": "6px"}),
            label("Data Type"),
            dcc.Dropdown(id="dd-datatype", clearable=False, style={"fontSize": "13px"}),

            # 頻率範圍
            section("🔍 Frequency Range"),
            html.Div([
                html.Div([
                    label("Min"),
                    dcc.Input(id="freq-min", type="number", debounce=True, style=INPUT_STYLE),
                ], style={"flex": 1, "marginRight": "6px"}),
                html.Div([
                    label("Max"),
                    dcc.Input(id="freq-max", type="number", debounce=True, style=INPUT_STYLE),
                ], style={"flex": 1}),
            ], style={"display": "flex", "marginBottom": "4px"}),
            html.Div(id="freq-range-caption", style={"fontSize": "11px", "color": "#6c757d"}),

            # Amplitude 範圍
            html.Div(id="amp-range-section", children=[
                section("📏 Amplitude Range"),
                html.Div([
                    html.Div([
                        label("Min (dB)"),
                        dcc.Input(id="amp-min", type="number", debounce=True, style=INPUT_STYLE),
                    ], style={"flex": 1, "marginRight": "6px"}),
                    html.Div([
                        label("Max (dB)"),
                        dcc.Input(id="amp-max", type="number", debounce=True, style=INPUT_STYLE),
                    ], style={"flex": 1}),
                ], style={"display": "flex", "marginBottom": "4px"}),
                html.Div(id="amp-range-caption", style={"fontSize": "11px", "color": "#6c757d"}),
            ]),

            # Frequency-domain marker sections (hidden when Smith Chart)
            html.Div(id="freq-marker-section", children=[
                # Marker
                section("📍 Markers"),
                dcc.Store(id="num-markers", data={}),
                html.Div([
                    html.Button("−", id="btn-marker-dec", n_clicks=0, title="Remove last marker",
                        className="counter-btn",
                        style={"width": "28px", "height": "28px", "border": "1px solid #ced4da",
                               "borderRadius": "4px", "backgroundColor": "#fff", "cursor": "pointer",
                               "fontSize": "16px", "lineHeight": "1", "padding": "0", "fontWeight": "500"}),
                    html.Span("0", id="num-markers-display", style={
                        "minWidth": "28px", "textAlign": "center",
                        "fontSize": "15px", "fontWeight": "500", "color": "#212529"}),
                    html.Button("+", id="btn-marker-inc", n_clicks=0, title="Add marker",
                        className="counter-btn",
                        style={"width": "28px", "height": "28px", "border": "1px solid #ced4da",
                               "borderRadius": "4px", "backgroundColor": "#fff", "cursor": "pointer",
                               "fontSize": "16px", "lineHeight": "1", "padding": "0", "fontWeight": "500"}),
                ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "8px"}),
                html.Div(id="markers-container"),

                # Threshold Marker
                section("📐 Threshold Markers"),
                dcc.Store(id="num-thresholds", data={}),
                html.Div([
                    html.Button("−", id="btn-threshold-dec", n_clicks=0, title="Remove last threshold",
                        className="counter-btn",
                        style={"width": "28px", "height": "28px", "border": "1px solid #ced4da",
                               "borderRadius": "4px", "backgroundColor": "#fff", "cursor": "pointer",
                               "fontSize": "16px", "lineHeight": "1", "padding": "0", "fontWeight": "500"}),
                    html.Span("0", id="num-thresholds-display", style={
                        "minWidth": "28px", "textAlign": "center",
                        "fontSize": "15px", "fontWeight": "500", "color": "#212529"}),
                    html.Button("+", id="btn-threshold-inc", n_clicks=0, title="Add threshold",
                        className="counter-btn",
                        style={"width": "28px", "height": "28px", "border": "1px solid #ced4da",
                               "borderRadius": "4px", "backgroundColor": "#fff", "cursor": "pointer",
                               "fontSize": "16px", "lineHeight": "1", "padding": "0", "fontWeight": "500"}),
                ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "8px"}),
                html.Div(id="thresholds-container"),
            ]),

            # Smith Chart marker section (shown only when Smith Chart)
            html.Div(id="smith-marker-section", style={"display": "none"}, children=[
                section("📍 Smith Markers"),
                dcc.Store(id="num-smith-markers", data={}),
                html.Div([
                    html.Button("−", id="btn-smith-marker-dec", n_clicks=0, title="Remove last Smith marker",
                        className="counter-btn",
                        style={"width": "28px", "height": "28px", "border": "1px solid #ced4da",
                               "borderRadius": "4px", "backgroundColor": "#fff", "cursor": "pointer",
                               "fontSize": "16px", "lineHeight": "1", "padding": "0", "fontWeight": "500"}),
                    html.Span("0", id="num-smith-markers-display", style={
                        "minWidth": "28px", "textAlign": "center",
                        "fontSize": "15px", "fontWeight": "500", "color": "#212529"}),
                    html.Button("+", id="btn-smith-marker-inc", n_clicks=0, title="Add Smith marker",
                        className="counter-btn",
                        style={"width": "28px", "height": "28px", "border": "1px solid #ced4da",
                               "borderRadius": "4px", "backgroundColor": "#fff", "cursor": "pointer",
                               "fontSize": "16px", "lineHeight": "1", "padding": "0", "fontWeight": "500"}),
                ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "8px"}),
                html.Div(id="smith-markers-container"),
            ]),

            # 檔案顯示切換
            section("📋 Select Files"),
            html.Div("No files loaded", id="no-files-hint",
                     style={"fontSize": "12px", "color": "#aaa"}),
            dcc.Checklist(
                id="file-checklist",
                options=[],
                value=[],
                style={"display": "none"},
                inputStyle={"marginRight": "6px"},
                labelStyle={"display": "block", "marginBottom": "4px"},
            ),

            # 線條寬度
            section("🖊 Line Width"),
            dcc.Slider(id="line-width", min=1, max=5, step=1, value=2,
                       marks={i: str(i) for i in range(1, 6)},
                       tooltip={"placement": "bottom"}),

            # 清除快取
            html.Hr(style={"margin": "16px 0"}),
            html.Button("🗑️ Clear All Files", id="btn-clear", n_clicks=0,
                        style={"width": "100%", "padding": "6px", "cursor": "pointer",
                               "backgroundColor": "#fff0f0", "border": "1px solid #f5c6cb",
                               "borderRadius": "4px", "fontSize": "13px"}),
            html.Div(id="cache-status", style={"fontSize": "11px", "color": "#6c757d", "marginTop": "4px"}),

        ], id="sidebar", style=SIDEBAR_STYLE),

        # ════ 拖曳把手 + 折疊鈕 ════
        html.Div(
            html.Button("◀", id="btn-sidebar-toggle", n_clicks=0,
                        title="拖曳調整寬度 / 點擊收合",
                        style={
                            "position": "absolute", "top": "16px",
                            "left": "50%", "transform": "translateX(-50%)",
                            "cursor": "pointer",
                            "border": "1px solid #dee2e6",
                            "background": "white",
                            "borderRadius": "0 4px 4px 0",
                            "width": "14px", "height": "40px",
                            "fontSize": "9px", "padding": "0",
                            "lineHeight": "40px", "textAlign": "center",
                            "boxShadow": "1px 0 4px rgba(0,0,0,0.08)",
                            "userSelect": "none",
                        }),
            id="sidebar-gutter",
            style={
                "width": "14px", "flexShrink": "0",
                "position": "relative",
                "backgroundColor": "#f8f9fa",
                "borderLeft": "1px solid #dee2e6",
                "cursor": "col-resize",
                "userSelect": "none",
                "zIndex": "5",
            },
        ),

        # ════ 主內容區 ════
        html.Div([
            html.H1(f"📊 S-parameter Viewer  [v{APP_VERSION}]", style={"fontSize": "32px", "marginBottom": "4px", "marginTop": 0}),
            html.P("Upload multiple .snp files (.s1p, .s2p, .s4p, etc.) for plotting comparison and analysis",
                   style={"color": "#6c757d", "fontSize": "14px", "marginBottom": "8px"}),

            html.Div([
                html.Span("Layout:", style={"fontSize": "13px", "color": "#6c757d",
                                            "alignSelf": "center", "marginRight": "8px"}),
                *[html.Button(lbl, id={"type": "layout-btn", "index": mode}, n_clicks=0,
                              style=BTN_LAYOUT_ACTIVE if mode == "1x1" else BTN_LAYOUT_INACTIVE)
                  for mode, lbl in [("1x1", "1×1"), ("1x2", "1×2"), ("2x1", "2×1"), ("2x2", "2×2")]],
            ], style={"display": "flex", "gap": "4px", "alignItems": "center", "marginBottom": "12px"}),

            html.Div(id="plot-section", style={"overflow": "hidden", "paddingBottom": "10px"}, children=[
                html.Div(id="main-content", children=[
                    html.Div("Drop .snp files anywhere on the page, or browse from the sidebar",
                             style={"padding": "40px", "textAlign": "center", "color": "#6c757d",
                                    "backgroundColor": "#f8f9fa", "borderRadius": "8px",
                                    "border": "2px dashed #dee2e6"})
                ]),
            ]),
            html.Div(
                html.Div(style={"width": "48px", "height": "4px", "borderRadius": "2px",
                                "backgroundColor": "#ced4da", "margin": "0 auto"}),
                id="plot-table-gutter",
                style={"height": "14px", "cursor": "row-resize", "display": "flex",
                       "alignItems": "center", "margin": "4px 0", "userSelect": "none"},
            ),
            html.Div(id="tables-area"),
            html.Div(id="resize-trigger-dummy", style={"display": "none"}),
            html.Div(id="drag-gutter-inited",   style={"display": "none"}),

            html.Hr(style={"margin": "24px 0"}),
            html.P("💡 Tip: Upload .snp files, select parameters, and use markers for precise frequency analysis.",
                   style={"color": "#6c757d", "fontSize": "12px"}),
        ], style=CONTENT_STYLE),

    ], style={"display": "flex", "height": "100vh", "overflow": "hidden"}),

], style={"fontFamily": "system-ui, -apple-system, sans-serif"})


# ============================================================
# Callbacks
# ============================================================

# ── 1. 上傳檔案 → 更新 Store ─────────────────────────────
@callback(
    Output("store-files-data", "data"),
    Output("store-freq-meta", "data"),
    Output("upload-error", "children"),
    Output("cache-status", "children"),
    Input("store-upload-queue", "data"),
    State("store-files-data", "data"),
    prevent_initial_call=True,
)
def handle_upload(queue_data, existing_data):
    if not queue_data:
        raise PreventUpdate

    contents_list = queue_data.get("contents", [])
    filenames     = queue_data.get("filenames", [])
    sizes         = queue_data.get("sizes", [])

    if not contents_list:
        raise PreventUpdate

    existing_data = existing_data or {}
    errors = []

    for i, (content, filename) in enumerate(zip(contents_list, filenames)):
        if filename in existing_data:
            continue
        try:
            _, b64 = content.split(",", 1)
            file_bytes = base64.b64decode(b64)
            size_kb = sizes[i] if i < len(sizes) else round(len(file_bytes) / 1024, 1)
            try:
                file_str = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                file_str = file_bytes.decode("latin-1")
            df, z0 = parse_sparameter_file(filename, file_str)
            existing_data[filename] = {
                "df_json": df.to_json(orient="split"),
                "size_kb": size_kb,
                "z0": z0,
            }
        except Exception as e:
            errors.append(f"❌ {filename}: {str(e)}")

    # 計算頻率 meta（以第一個檔案為準）
    freq_meta = {}
    if existing_data:
        first_df = pd.read_json(io.StringIO(list(existing_data.values())[0]["df_json"]), orient="split", convert_dates=False)
        _, unit, factor = auto_convert_frequency(first_df, return_unit_factor=True)
        all_mins = []
        all_maxs = []
        for fdata in existing_data.values():
            df = pd.read_json(io.StringIO(fdata["df_json"]), orient="split", convert_dates=False)
            all_mins.append(df["Frequency"].min())
            all_maxs.append(df["Frequency"].max())
        freq_meta = {
            "unit": unit,
            "factor": factor,
            "raw_min": min(all_mins),
            "raw_max": max(all_maxs),
            "disp_min": min(all_mins) / factor,
            "disp_max": max(all_maxs) / factor,
        }

    error_msg = " | ".join(errors) if errors else ""
    status = f"📦 {len(existing_data)} file(s) loaded"
    return existing_data, freq_meta, error_msg, status


# ── 2. 清除全部 ──────────────────────────────────────────
@callback(
    Output("store-files-data", "data", allow_duplicate=True),
    Output("store-freq-meta", "data", allow_duplicate=True),
    Output("cache-status", "children", allow_duplicate=True),
    Input("btn-clear", "n_clicks"),
    prevent_initial_call=True,
)
def clear_files(n):
    if not n:
        raise PreventUpdate
    return {}, {}, "📦 0 file(s) loaded"


# ── 2b. 顯示已上傳檔案卡片 ───────────────────────────────
@callback(
    Output("uploaded-files-list", "children"),
    Input("store-files-data", "data"),
)
def render_uploaded_files_list(files_data):
    if not files_data:
        return []
    cards = []
    for filename, fdata in files_data.items():
        size_kb = fdata.get("size_kb", "?")
        cards.append(
            html.Div([
                html.Div("📄", style={
                    "width": "36px", "height": "36px",
                    "backgroundColor": "#1e293b",
                    "borderRadius": "6px",
                    "display": "flex", "alignItems": "center",
                    "justifyContent": "center",
                    "fontSize": "16px", "flexShrink": "0",
                }),
                html.Div([
                    html.Div(filename, style={
                        "fontSize": "12px", "fontWeight": "500",
                        "color": "#212529", "wordBreak": "break-all",
                        "lineHeight": "1.3",
                    }),
                    html.Div(f"{size_kb} KB", style={
                        "fontSize": "11px", "color": "#6c757d", "marginTop": "1px",
                    }),
                ], style={"flex": "1", "minWidth": "0", "marginLeft": "8px"}),
                html.Button(
                    "×",
                    id={"type": "file-del-btn", "index": filename},
                    n_clicks=0,
                    title=f"Remove {filename}",
                    style={
                        "width": "20px", "height": "20px",
                        "border": "1px solid #dee2e6", "borderRadius": "50%",
                        "backgroundColor": "#f8f9fa", "color": "#6c757d",
                        "cursor": "pointer", "fontSize": "14px", "lineHeight": "1",
                        "padding": "0", "fontWeight": "700", "flexShrink": "0",
                        "display": "inline-flex", "alignItems": "center",
                        "justifyContent": "center",
                    },
                ),
            ], style={
                "display": "flex", "alignItems": "center",
                "padding": "8px 10px",
                "backgroundColor": "#fff",
                "border": "1px solid #e9ecef",
                "borderRadius": "8px",
                "marginBottom": "6px",
            })
        )
    return cards


# ── 2c. 單檔刪除 ──────────────────────────────────────────
@callback(
    Output("store-files-data", "data", allow_duplicate=True),
    Output("store-freq-meta", "data", allow_duplicate=True),
    Output("cache-status", "children", allow_duplicate=True),
    Input({"type": "file-del-btn", "index": ALL}, "n_clicks"),
    State("store-files-data", "data"),
    prevent_initial_call=True,
)
def delete_single_file(del_clicks, files_data):
    if not any(del_clicks):
        raise PreventUpdate
    tid = ctx.triggered_id
    if not isinstance(tid, dict) or tid.get("type") != "file-del-btn":
        raise PreventUpdate

    files_data = dict(files_data or {})
    files_data.pop(tid["index"], None)

    freq_meta = {}
    if files_data:
        first_df = pd.read_json(io.StringIO(list(files_data.values())[0]["df_json"]), orient="split", convert_dates=False)
        _, unit, factor = auto_convert_frequency(first_df, return_unit_factor=True)
        all_mins, all_maxs = [], []
        for fdata in files_data.values():
            df = pd.read_json(io.StringIO(fdata["df_json"]), orient="split", convert_dates=False)
            all_mins.append(df["Frequency"].min())
            all_maxs.append(df["Frequency"].max())
        freq_meta = {
            "unit": unit, "factor": factor,
            "raw_min": min(all_mins), "raw_max": max(all_maxs),
            "disp_min": min(all_mins) / factor,
            "disp_max": max(all_maxs) / factor,
        }

    return files_data, freq_meta, f"📦 {len(files_data)} file(s) loaded"


# ── 3. 更新 Parameter Dropdown ───────────────────────────
@callback(
    Output("dd-param", "options"),
    Output("dd-param", "value"),
    Input("store-files-data", "data"),
)
def update_param_dropdown(files_data):
    if not files_data:
        return [], None
    first_df = pd.read_json(io.StringIO(list(files_data.values())[0]["df_json"]), orient="split", convert_dates=False)
    params = get_available_parameters(first_df)
    default = "S21" if "S21" in params else (params[0] if params else None)
    return params, default


# ── 4. 更新 DataType Dropdown ────────────────────────────
@callback(
    Output("dd-datatype", "options"),
    Output("dd-datatype", "value"),
    Input("dd-param", "value"),
    State("store-files-data", "data"),
    State("dd-datatype", "value"),
)
def update_datatype_dropdown(param, files_data, current_datatype):
    if not param or not files_data:
        return [], None
    first_df = pd.read_json(io.StringIO(list(files_data.values())[0]["df_json"]), orient="split", convert_dates=False)
    dtypes = get_data_types(first_df, param)
    value = current_datatype if current_datatype in dtypes else (dtypes[0] if dtypes else None)
    return dtypes, value


# ── 5. 初始化頻率範圍輸入 ────────────────────────────────
@callback(
    Output("freq-min", "value"),
    Output("freq-min", "min"),
    Output("freq-min", "max"),
    Output("freq-max", "value"),
    Output("freq-max", "min"),
    Output("freq-max", "max"),
    Output("freq-range-caption", "children"),
    Input("store-freq-meta", "data"),
)
def init_freq_range(meta):
    if not meta:
        return [no_update] * 7
    dmin = meta["disp_min"]
    dmax = meta["disp_max"]
    unit = meta["unit"]
    caption = f"📋 Full range: {dmin:.3f} ~ {dmax:.3f} {unit}"
    return dmin, dmin, dmax, dmax, dmin, dmax, caption


# ── 6. 初始化 Amplitude 範圍（依選擇的 param 自動計算）─
@callback(
    Output("amp-min", "value"),
    Output("amp-max", "value"),
    Output("amp-range-caption", "children"),
    Output("amp-range-section", "style"),
    Input("dd-param", "value"),
    Input("dd-datatype", "value"),
    Input("freq-min", "value"),
    Input("freq-max", "value"),
    State("store-files-data", "data"),
    State("store-freq-meta", "data"),
)
def update_amp_range(param, data_type, fmin, fmax, files_data, freq_meta):
    if not param or not data_type or not files_data:
        return no_update, no_update, no_update, {"display": "none"}
    is_smith = data_type == "smith"
    if is_smith:
        return no_update, no_update, no_update, {"display": "none"}

    selected_param_full = f"{param}_{data_type}"
    y_unit = get_y_axis_unit(data_type)
    fmin = fmin or freq_meta.get("disp_min", 0)
    fmax = fmax or freq_meta.get("disp_max", 1)

    all_vals = []
    for fdata in files_data.values():
        df = pd.read_json(io.StringIO(fdata["df_json"]), orient="split", convert_dates=False)
        df_conv, _ = auto_convert_frequency(df)
        df_filt = filter_by_frequency_range(df_conv, fmin, fmax)
        if selected_param_full in df_filt.columns:
            all_vals.extend(df_filt[selected_param_full].dropna().tolist())

    if not all_vals:
        return -100.0, 0.0, "", {"display": "block"}

    data_min, data_max = float(min(all_vals)), float(max(all_vals))
    pad = (data_max - data_min) * 0.05 if data_max != data_min else 1.0
    amp_min = round(data_min - pad, 1)
    amp_max = round(data_max + pad, 1)
    caption = f"📋 Data range: {data_min:.1f} ~ {data_max:.1f} {y_unit}"
    return amp_min, amp_max, caption, {"display": "block"}


# ── 6b. +/- 按鈕 & 個別刪除按鈕：Marker store (dict) ────
@callback(
    Output("num-markers", "data"),
    Output("num-markers-display", "children"),
    Input("btn-marker-inc", "n_clicks"),
    Input("btn-marker-dec", "n_clicks"),
    Input({"type": "marker-del-btn", "index": ALL}, "n_clicks"),
    Input({"type": "marker-freq",    "index": ALL}, "value"),   # 使用者改頻率 → 寫回 store
    State("num-markers", "data"),
    State("store-freq-meta", "data"),
    prevent_initial_call=True,
)
def update_num_markers(inc, dec, del_clicks, freq_vals, store, freq_meta):
    # store 格式: { str(id): freq_float }
    store = dict(store or {})
    tid = ctx.triggered_id
    freq_meta = freq_meta or {}
    dmin = freq_meta.get("disp_min", 0)
    dmax = freq_meta.get("disp_max", 1)
    mid_freq = round((dmin + dmax) / 2, 3)

    # 使用者（或拖動 sync）修改了某個 marker 的 freq → 全部同步寫回 store，不增減
    if isinstance(tid, dict) and tid.get("type") == "marker-freq":
        for item in ctx.inputs_list[3]:   # marker-freq inputs
            m_id = str(item["id"]["index"])
            if m_id in store and item.get("value") is not None:
                store[m_id] = item["value"]
        return store, str(len(store))

    if tid == "btn-marker-inc":
        if len(store) < 10:
            existing_ids = [int(k) for k in store]
            next_id = str(max(existing_ids) + 1 if existing_ids else 1)
            store[next_id] = mid_freq
    elif tid == "btn-marker-dec":
        if store:
            last_key = list(store.keys())[-1]
            del store[last_key]
    elif isinstance(tid, dict) and tid.get("type") == "marker-del-btn":
        del_key = str(tid["index"])
        store.pop(del_key, None)

    return store, str(len(store))


# ── 6c. +/- 按鈕 & 個別刪除：Threshold store (dict) ──────
@callback(
    Output("num-thresholds", "data"),
    Output("num-thresholds-display", "children"),
    Input("btn-threshold-inc", "n_clicks"),
    Input("btn-threshold-dec", "n_clicks"),
    Input({"type": "threshold-del-btn", "index": ALL}, "n_clicks"),
    Input({"type": "threshold-value",   "index": ALL}, "value"),
    State("num-thresholds", "data"),
    State("dd-datatype", "value"),
    prevent_initial_call=True,
)
def update_num_thresholds(inc, dec, del_clicks, t_vals, store, data_type):
    store = dict(store or {})
    tid = ctx.triggered_id
    y_unit = get_y_axis_unit(data_type or "mag")
    default_val = -30.0

    # 使用者修改了某個 threshold 的值 → 全部同步寫回 store
    if isinstance(tid, dict) and tid.get("type") == "threshold-value":
        for item in ctx.inputs_list[3]:
            t_id = str(item["id"]["index"])
            if t_id in store and item.get("value") is not None:
                store[t_id] = item["value"]
        return store, str(len(store))

    if tid == "btn-threshold-inc":
        if len(store) < 10:
            existing_ids = [int(k) for k in store]
            next_id = str(max(existing_ids) + 1 if existing_ids else 1)
            store[next_id] = default_val
    elif tid == "btn-threshold-dec":
        if store:
            last_key = list(store.keys())[-1]
            del store[last_key]
    elif isinstance(tid, dict) and tid.get("type") == "threshold-del-btn":
        del_key = str(tid["index"])
        store.pop(del_key, None)

    return store, str(len(store))


# ── 6d. data type 切換：顯示對應 marker 控制區 ────────────
@callback(
    Output("freq-marker-section", "style"),
    Output("smith-marker-section", "style"),
    Input("dd-datatype", "value"),
)
def toggle_marker_sections(data_type):
    is_smith = data_type == "smith"
    return (
        {"display": "none"} if is_smith else {"display": "block"},
        {"display": "block"} if is_smith else {"display": "none"},
    )


# ── 6e. +/- 按鈕 & 個別刪除：Smith Marker store (dict) ───
@callback(
    Output("num-smith-markers", "data"),
    Output("num-smith-markers-display", "children"),
    Input("btn-smith-marker-inc", "n_clicks"),
    Input("btn-smith-marker-dec", "n_clicks"),
    Input({"type": "smith-marker-del-btn", "index": ALL}, "n_clicks"),
    Input({"type": "smith-marker-freq",    "index": ALL}, "value"),
    State("num-smith-markers", "data"),
    State("store-freq-meta", "data"),
    prevent_initial_call=True,
)
def update_num_smith_markers(inc, dec, del_clicks, freq_vals, store, freq_meta):
    store = dict(store or {})
    tid = ctx.triggered_id
    freq_meta = freq_meta or {}
    dmin = freq_meta.get("disp_min", 0)
    dmax = freq_meta.get("disp_max", 1)
    mid_freq = round((dmin + dmax) / 2, 3)

    if isinstance(tid, dict) and tid.get("type") == "smith-marker-freq":
        for item in ctx.inputs_list[3]:
            m_id = str(item["id"]["index"])
            if m_id in store and item.get("value") is not None:
                store[m_id] = item["value"]
        return store, str(len(store))

    if tid == "btn-smith-marker-inc":
        if len(store) < 10:
            existing_ids = [int(k) for k in store]
            next_id = str(max(existing_ids) + 1 if existing_ids else 1)
            store[next_id] = mid_freq
    elif tid == "btn-smith-marker-dec":
        if store:
            last_key = list(store.keys())[-1]
            del store[last_key]
    elif isinstance(tid, dict) and tid.get("type") == "smith-marker-del-btn":
        del_key = str(tid["index"])
        store.pop(del_key, None)

    return store, str(len(store))


# ── 7. 動態產生 Marker 輸入欄位 ─────────────────────────
@callback(
    Output("markers-container", "children"),
    Input("num-markers", "data"),
    State("store-freq-meta", "data"),
)
def render_marker_inputs(store, freq_meta):
    if not store or not freq_meta:
        return []
    unit = freq_meta.get("unit", "GHz")
    children = []
    for pos, (m_id_str, freq_val) in enumerate(store.items()):
        m_id = int(m_id_str)
        summary = html.Summary(
            html.Div([
                html.Span(f"📌 Marker {m_id}",
                          style={"fontWeight": "500", "fontSize": "13px",
                                 "flexShrink": "0", "marginRight": "8px"}),
                dcc.Input(
                    id={"type": "marker-freq", "index": m_id},
                    type="number", value=freq_val, debounce=True, step="any",
                    style={"flex": "1", "minWidth": "0", "padding": "2px 4px",
                           "fontSize": "12px", "border": "1px solid #ced4da",
                           "borderRadius": "4px", "textAlign": "center"},
                ),
                html.Span(f" {unit}", style={"fontSize": "12px", "color": "#6c757d",
                                             "flexShrink": "0", "marginLeft": "2px"}),
                html.Button(
                    "×",
                    id={"type": "marker-del-btn", "index": m_id},
                    n_clicks=0,
                    title=f"Delete Marker {m_id}",
                    className="del-circle-btn",
                    style={
                        "marginLeft": "6px",
                        "width": "20px", "height": "20px",
                        "border": "1px solid #f5c6cb",
                        "borderRadius": "50%",
                        "backgroundColor": "#fff0f0",
                        "color": "#c0392b",
                        "cursor": "pointer",
                        "fontSize": "13px",
                        "lineHeight": "1",
                        "padding": "0",
                        "fontWeight": "700",
                        "display": "inline-flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "verticalAlign": "middle",
                        "flexShrink": "0",
                    },
                ),
            ], style={"display": "flex", "alignItems": "center", "width": "100%"}),
            style={"cursor": "pointer", "padding": "4px 0"},
        )
        body = html.Div([
            html.Div([
                html.Div([
                    label("Color"),
                    color_dropdown({"type": "marker-color", "index": m_id}, default_index=pos),
                ], style={"flex": 1, "marginRight": "8px"}),
                html.Div([
                    dcc.Checklist(
                        id={"type": "marker-show-legend", "index": m_id},
                        options=[{"label": " Show in Legend", "value": "show"}],
                        value=["show"],
                        style={"fontSize": "12px", "marginTop": "18px"},
                    ),
                ], style={"flex": 1}),
            ], style={"display": "flex"}),
        ], style={"padding": "4px 0 4px 8px"})
        children.append(html.Details(
            [summary, body], open=(pos == 0),
            style={"borderBottom": "1px solid #eee", "paddingBottom": "4px", "marginBottom": "2px"},
        ))
    return children


# ── 8. 動態產生 Threshold 輸入欄位 ─────────────────────
@callback(
    Output("thresholds-container", "children"),
    Input("num-thresholds", "data"),
    State("dd-datatype", "value"),
)
def render_threshold_inputs(store, data_type):
    if not store:
        return []
    y_unit = get_y_axis_unit(data_type or "mag")
    children = []
    for pos, (t_id_str, t_val) in enumerate(store.items()):
        t_id = int(t_id_str)
        summary = html.Summary(
            html.Div([
                html.Span(f"📐 Threshold {t_id}",
                          style={"fontWeight": "500", "fontSize": "13px",
                                 "flexShrink": "0", "marginRight": "8px"}),
                dcc.Input(
                    id={"type": "threshold-value", "index": t_id},
                    type="number", value=t_val, debounce=True, step="any",
                    style={"flex": "1", "minWidth": "0", "padding": "2px 4px",
                           "fontSize": "12px", "border": "1px solid #ced4da",
                           "borderRadius": "4px", "textAlign": "center"},
                ),
                html.Span(f" {y_unit}", style={"fontSize": "12px", "color": "#6c757d",
                                               "flexShrink": "0", "marginLeft": "2px"}),
                html.Button(
                    "×",
                    id={"type": "threshold-del-btn", "index": t_id},
                    n_clicks=0,
                    title=f"Delete Threshold {t_id}",
                    className="del-circle-btn",
                    style={
                        "marginLeft": "6px",
                        "width": "20px", "height": "20px",
                        "border": "1px solid #f5c6cb",
                        "borderRadius": "50%",
                        "backgroundColor": "#fff0f0",
                        "color": "#c0392b",
                        "cursor": "pointer",
                        "fontSize": "13px",
                        "lineHeight": "1",
                        "padding": "0",
                        "fontWeight": "700",
                        "display": "inline-flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "verticalAlign": "middle",
                        "flexShrink": "0",
                    },
                ),
            ], style={"display": "flex", "alignItems": "center", "width": "100%"}),
            style={"cursor": "pointer", "padding": "4px 0"},
        )
        body = html.Div([
            html.Div([
                html.Button(
                    "Peak", id={"type": "threshold-peak-btn", "index": t_id}, n_clicks=0,
                    title="Set to Peak (max within freq range)",
                    className="threshold-preset-btn",
                ),
                html.Button(
                    "- 3 dB", id={"type": "threshold-3db-btn", "index": t_id}, n_clicks=0,
                    title="Set to Peak - 3 dB (3dB bandwidth reference)",
                    className="threshold-preset-btn",
                ),
                html.Button(
                    "Dip", id={"type": "threshold-dip-btn", "index": t_id}, n_clicks=0,
                    title="Set to Dip (min within freq range)",
                    className="threshold-preset-btn",
                ),
            ], style={"display": "flex", "gap": "6px", "marginBottom": "6px", "paddingLeft": "8px", "paddingRight": "8px"}),
            html.Div([
                html.Div([
                    label("Color"),
                    color_dropdown({"type": "threshold-color", "index": t_id}, default_index=pos),
                ], style={"flex": 1, "marginRight": "8px"}),
                html.Div([
                    dcc.Checklist(
                        id={"type": "threshold-show-legend", "index": t_id},
                        options=[{"label": " Show in Legend", "value": "show"}],
                        value=["show"],
                        style={"fontSize": "12px", "marginTop": "18px"},
                    ),
                ], style={"flex": 1}),
            ], style={"display": "flex"}),
        ], style={"padding": "4px 0 4px 8px"})
        children.append(html.Details(
            [summary, body], open=(pos == 0),
            style={"borderBottom": "1px solid #eee", "paddingBottom": "4px", "marginBottom": "2px"},
        ))
    return children


# ── 8b. Threshold Peak / Dip 按鈕 ────────────────────────
@callback(
    Output({"type": "threshold-value", "index": ALL}, "value", allow_duplicate=True),
    Input({"type": "threshold-peak-btn", "index": ALL}, "n_clicks"),
    Input({"type": "threshold-dip-btn",  "index": ALL}, "n_clicks"),
    Input({"type": "threshold-3db-btn",  "index": ALL}, "n_clicks"),
    State("store-files-data", "data"),
    State("dd-param", "value"),
    State("dd-datatype", "value"),
    State("file-checklist", "value"),
    State("freq-min", "value"),
    State("freq-max", "value"),
    State("store-freq-meta", "data"),
    prevent_initial_call=True,
)
def set_threshold_peak_dip(
    peak_clicks, dip_clicks, db3_clicks,
    files_data, param, data_type, visible_filenames,
    fmin, fmax, freq_meta,
):
    tid = ctx.triggered_id
    if not isinstance(tid, dict) or tid.get("type") not in (
        "threshold-peak-btn", "threshold-dip-btn", "threshold-3db-btn"
    ):
        raise PreventUpdate
    if not any(peak_clicks + dip_clicks + db3_clicks):
        raise PreventUpdate
    if not files_data or not param or not data_type or data_type == "smith":
        raise PreventUpdate

    triggered_index = str(tid["index"])
    selected_param_full = f"{param}_{data_type}"
    freq_meta = freq_meta or {}
    fmin = fmin if fmin is not None else freq_meta.get("disp_min", 0)
    fmax = fmax if fmax is not None else freq_meta.get("disp_max", 1)
    visible_filenames = visible_filenames or list(files_data.keys())

    all_vals = []
    for fn in visible_filenames:
        if fn not in files_data:
            continue
        df = pd.read_json(io.StringIO(files_data[fn]["df_json"]), orient="split", convert_dates=False)
        df_conv, _ = auto_convert_frequency(df)
        df_filt = filter_by_frequency_range(df_conv, fmin, fmax)
        if selected_param_full in df_filt.columns:
            all_vals.extend(df_filt[selected_param_full].dropna().tolist())

    if not all_vals:
        raise PreventUpdate

    t_type = tid["type"]
    if t_type == "threshold-peak-btn":
        new_val = float(max(all_vals))
    elif t_type == "threshold-dip-btn":
        new_val = float(min(all_vals))
    else:  # -3dB
        new_val = float(max(all_vals)) - 3.0

    return [
        new_val if str(item["id"]["index"]) == triggered_index else no_update
        for item in ctx.inputs_list[0]
    ]


# ── 8b. 動態產生 Smith Marker 列表 ───────────────────────
@callback(
    Output("smith-markers-container", "children"),
    Input("num-smith-markers", "data"),
    State("store-freq-meta", "data"),
    State("store-active-smith-marker", "data"),
)
def render_smith_marker_inputs(store, freq_meta, active_id):
    if not store or not freq_meta:
        return []
    unit = freq_meta.get("unit", "GHz")
    children = []
    for pos, (m_id_str, freq_val) in enumerate(store.items()):
        m_id = int(m_id_str)
        _btn_base = {
            "width": "22px", "height": "22px",
            "border": "1px solid #ced4da",
            "borderRadius": "50%",
            "cursor": "pointer",
            "fontSize": "13px",
            "lineHeight": "1",
            "padding": "0",
            "display": "inline-flex",
            "alignItems": "center",
            "justifyContent": "center",
            "verticalAlign": "middle",
            "flexShrink": "0",
        }
        del_btn_style = {
            **_btn_base,
            "marginLeft": "4px",
            "border": "1px solid #f5c6cb",
            "backgroundColor": "#fff0f0",
            "color": "#c0392b",
            "fontWeight": "700",
        }
        is_active = (active_id == m_id)
        aim_btn_style = {
            **_btn_base,
            "marginLeft": "4px",
            "backgroundColor": "#cce5ff" if is_active else "#f8f9fa",
            "borderColor":     "#66b2ff" if is_active else "#ced4da",
            "color":           "#004085" if is_active else "#6c757d",
            "outline":         "none",
        }
        summary = html.Summary(
            html.Div([
                html.Span(f"📌 M{m_id}",
                          style={"fontWeight": "500", "fontSize": "13px",
                                 "flexShrink": "0", "marginRight": "8px"}),
                dcc.Input(
                    id={"type": "smith-marker-freq", "index": m_id},
                    type="number", value=freq_val, debounce=True, step="any",
                    style={"flex": "1", "minWidth": "0", "padding": "2px 4px",
                           "fontSize": "12px", "border": "1px solid #ced4da",
                           "borderRadius": "4px", "textAlign": "center"},
                ),
                html.Span(f" {unit}", style={"fontSize": "12px", "color": "#6c757d",
                                             "flexShrink": "0", "marginLeft": "2px"}),
                html.Button(
                    "🎯",
                    id={"type": "smith-marker-aim-btn", "index": m_id},
                    n_clicks=0,
                    title="Click chart to move this marker",
                    className="smith-aim-btn",
                    style=aim_btn_style,
                ),
                html.Button(
                    "×",
                    id={"type": "smith-marker-del-btn", "index": m_id},
                    n_clicks=0,
                    title=f"Delete Smith Marker {m_id}",
                    className="del-circle-btn",
                    style=del_btn_style,
                ),
            ], style={"display": "flex", "alignItems": "center", "width": "100%"}),
            style={"cursor": "pointer", "padding": "4px 0"},
        )
        body = html.Div([
            html.Div([
                label("Color"),
                color_dropdown({"type": "smith-marker-color", "index": m_id}, default_index=pos),
            ], style={"padding": "4px 0 4px 8px"}),
        ])
        children.append(html.Details(
            [summary, body], open=(pos == 0),
            style={"borderBottom": "1px solid #eee", "paddingBottom": "4px", "marginBottom": "2px"},
        ))
    return children


# ── 8c. 🎯 toggle → store-active-smith-marker ────────────
@callback(
    Output("store-active-smith-marker", "data"),
    Input({"type": "smith-marker-aim-btn", "index": ALL}, "n_clicks"),
    State("store-active-smith-marker", "data"),
    prevent_initial_call=True,
)
def toggle_active_smith_marker(aim_clicks, current_active):
    if not any(c for c in aim_clicks if c):
        raise PreventUpdate
    tid = ctx.triggered_id
    if not isinstance(tid, dict):
        raise PreventUpdate
    clicked_id = tid["index"]
    return None if current_active == clicked_id else clicked_id


# ── 8d. 🎯 按鈕樣式同步 active 狀態 ──────────────────────
@callback(
    Output({"type": "smith-marker-aim-btn", "index": ALL}, "style"),
    Input("store-active-smith-marker", "data"),
    State({"type": "smith-marker-aim-btn", "index": ALL}, "id"),
)
def update_aim_btn_styles(active_id, btn_ids):
    _base = {
        "width": "22px", "height": "22px",
        "borderRadius": "50%",
        "cursor": "pointer",
        "fontSize": "13px",
        "lineHeight": "1",
        "padding": "0",
        "marginLeft": "4px",
        "display": "inline-flex",
        "alignItems": "center",
        "justifyContent": "center",
        "verticalAlign": "middle",
        "flexShrink": "0",
    }
    styles = []
    for btn_id in (btn_ids or []):
        is_active = btn_id["index"] == active_id
        styles.append({
            **_base,
            "border":           "1px solid #66b2ff" if is_active else "1px solid #ced4da",
            "backgroundColor":  "#cce5ff"           if is_active else "#f8f9fa",
            "color":            "#004085"            if is_active else "#6c757d",
            "outline":          "none",
        })
    return styles


# ── 8e. Smith marker 刪除時若刪的是 active，清除 store ───
@callback(
    Output("store-active-smith-marker", "data", allow_duplicate=True),
    Input({"type": "smith-marker-del-btn", "index": ALL}, "n_clicks"),
    Input("btn-smith-marker-dec", "n_clicks"),
    State("store-active-smith-marker", "data"),
    State("num-smith-markers", "data"),
    prevent_initial_call=True,
)
def clear_active_on_delete(del_clicks, dec_click, active_id, smith_store):
    if active_id is None:
        raise PreventUpdate
    tid = ctx.triggered_id
    store_ids = list((smith_store or {}).keys())

    if isinstance(tid, dict) and tid.get("type") == "smith-marker-del-btn":
        if tid["index"] == active_id:
            return None
    elif tid == "btn-smith-marker-dec":
        # dec 刪除最後一個 marker
        if store_ids and int(store_ids[-1]) == active_id:
            return None
    raise PreventUpdate


# ── 8f. 點擊 Smith 圖曲線 → 移動 target marker ──────────
@callback(
    Output({"type": "smith-marker-freq", "index": ALL}, "value"),
    Input({"type": "cell-graph", "index": ALL}, "clickData"),
    State("store-active-smith-marker", "data"),
    State("num-smith-markers", "data"),
    State({"type": "smith-marker-freq", "index": ALL}, "value"),
    State("dd-datatype", "value"),
    State("store-cell-configs", "data"),
    prevent_initial_call=True,
)
def smith_click_to_set(click_data_list, active_id, smith_store, current_freqs, data_type_c0, cell_configs):
    triggered = ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        raise PreventUpdate

    triggered_cell = triggered["index"]
    click_data = None
    for i, item in enumerate(ctx.inputs_list[0]):
        if item["id"]["index"] == triggered_cell:
            click_data = click_data_list[i]
            break

    if not click_data:
        raise PreventUpdate

    # Determine this cell's data_type
    if triggered_cell == 0:
        cell_dt = data_type_c0
    else:
        cell_configs = cell_configs or []
        cfg = cell_configs[triggered_cell - 1] if triggered_cell - 1 < len(cell_configs) else {}
        cell_dt = cfg.get("data_type") or data_type_c0

    if cell_dt != "smith":
        raise PreventUpdate

    points = click_data.get("points", [])
    if not points:
        raise PreventUpdate

    # customdata 是純 float → 主曲線；是 list → marker dot，忽略
    freq_val = points[0].get("customdata")
    if not isinstance(freq_val, (int, float)):
        raise PreventUpdate

    store_ids = list((smith_store or {}).keys())
    if not store_ids:
        raise PreventUpdate

    # 決定要移動哪個 marker
    if active_id is not None and str(active_id) in store_ids:
        target_id = active_id
    else:
        target_id = int(store_ids[-1])

    new_freqs = list(current_freqs)
    for i, item in enumerate(ctx.states_list[2]):   # smith-marker-freq ALL
        if item["id"]["index"] == target_id:
            new_freqs[i] = round(float(freq_val), 4)
            break

    return new_freqs


# ── 9. 檔案勾選清單 ──────────────────────────────────────
_CHECKLIST_VISIBLE_STYLE = {
    "fontSize": "12px", "maxHeight": "200px", "overflowY": "auto",
    "border": "1px solid #dee2e6", "borderRadius": "4px", "padding": "8px",
}

@callback(
    Output("file-checklist", "options"),
    Output("file-checklist", "value"),
    Output("file-checklist", "style"),
    Output("no-files-hint", "style"),
    Input("store-files-data", "data"),
)
def render_file_checklist(files_data):
    if not files_data:
        return [], [], {"display": "none"}, {"fontSize": "12px", "color": "#aaa"}
    options = [{"label": f" {get_display_name(fn)}", "value": fn} for fn in files_data]
    return options, list(files_data.keys()), _CHECKLIST_VISIBLE_STYLE, {"display": "none"}


# ── 10. 側邊欄折疊 ───────────────────────────────────────
@callback(
    Output("sidebar", "style"),
    Output("btn-sidebar-toggle", "children"),
    Input("btn-sidebar-toggle", "n_clicks"),
)
def toggle_sidebar(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        return SIDEBAR_COLLAPSED_STYLE, "▶"
    return SIDEBAR_STYLE, "◀"


# ── 10b. 拖曳調整側邊欄寬度（clientside）────────────────
app.clientside_callback(
    """
    function(_) {
        var gutter  = document.getElementById('sidebar-gutter');
        var sidebar = document.getElementById('sidebar');
        if (!gutter || !sidebar) return window.dash_clientside.no_update;

        var dragging = false;

        gutter.addEventListener('mousedown', function(e) {
            if (e.target.id === 'btn-sidebar-toggle') return;
            dragging = true;
            e.preventDefault();
            document.body.style.cursor    = 'col-resize';
            document.body.style.userSelect = 'none';
            sidebar.style.transition = 'none';
        });

        document.addEventListener('mousemove', function(e) {
            if (!dragging) return;
            var w = Math.min(Math.max(e.clientX, 150), 600);
            sidebar.style.width    = w + 'px';
            sidebar.style.minWidth = w + 'px';
        });

        document.addEventListener('mouseup', function() {
            if (!dragging) return;
            dragging = false;
            document.body.style.cursor    = '';
            document.body.style.userSelect = '';
            sidebar.style.transition = '';
        });

        return window.dash_clientside.no_update;
    }
    """,
    Output("store-resize-init", "data"),
    Input("store-resize-init", "data"),
)


# ── 10c. 輪詢 pending upload（每 300ms 檢查一次）────────
# drop zone 事件綁定已移至 assets/upload_handler.js
app.clientside_callback(
    """
    function(n) {
        if (!window._pendingUpload) return window.dash_clientside.no_update;
        var d = window._pendingUpload;
        window._pendingUpload = null;
        return d;
    }
    """,
    Output("store-upload-queue", "data"),
    Input("interval-upload-check", "n_intervals"),
)


# ── 10d. main-content 更新後觸發 Plotly resize（修正 CSS grid 欄寬時序問題）─
app.clientside_callback(
    """
    function(children) {
        setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 50);
        return window.dash_clientside.no_update;
    }
    """,
    Output("resize-trigger-dummy", "children"),
    Input("main-content", "children"),
)


# ── 10e. 根據視窗高度動態計算 graph_height ─────────────────
app.clientside_callback(
    """
    function(layout) {
        var sec = document.getElementById('plot-section');
        if (sec) sec.style.height = '';
        var nRows = (layout === '2x1' || layout === '2x2') ? 2 : 1;
        var chrome = 155;   // title + subtitle + layout bar + content padding
        var perRow = 48;    // cell header + cell border padding per row
        var gap    = 8;
        var available = window.innerHeight - chrome - perRow * nRows - gap * (nRows - 1);
        var h = Math.floor(available / nRows * 0.8);
        h = Math.max(h, 200);
        h = Math.min(h, 900);
        return h;
    }
    """,
    Output("store-graph-height", "data"),
    Input("store-layout", "data"),
)


# ── 10f. plot / table 區分隔線拖拉 ───────────────────────
app.clientside_callback(
    """
    function(children) {
        var handle = document.getElementById('plot-table-gutter');
        if (!handle) return window.dash_clientside.no_update;
        if (handle._dragInited) return window.dash_clientside.no_update;
        handle._dragInited = true;

        var dragging    = false;
        var startY      = 0;
        var startH      = 0;
        var lastGraphH  = 0;

        handle.addEventListener('mousedown', function(e) {
            dragging = true;
            startY   = e.clientY;
            var sec  = document.getElementById('plot-section');
            startH   = sec ? sec.offsetHeight : 400;
            e.preventDefault();
            document.body.style.cursor    = 'row-resize';
            document.body.style.userSelect = 'none';
        });

        document.addEventListener('mousemove', function(e) {
            if (!dragging) return;
            var sec  = document.getElementById('plot-section');
            var grid = document.getElementById('cell-grid');
            if (!sec || !grid) return;
            var nRows  = parseInt(grid.getAttribute('data-rows')) || 1;
            var newH   = Math.max(nRows * 150, startH + (e.clientY - startY));
            sec.style.height = newH + 'px';

            lastGraphH = Math.max(100, Math.floor((newH - 8 * (nRows - 1)) / nRows) - 48);
            document.querySelectorAll('.graph-cell-container').forEach(function(c) {
                c.style.height = lastGraphH + 'px';
                var dg = c.querySelector('.dash-graph');
                if (dg) dg.style.height = lastGraphH + 'px';
            });
            window.dispatchEvent(new Event('resize'));
        });

        document.addEventListener('mouseup', function() {
            if (!dragging) return;
            dragging = false;
            document.body.style.cursor    = '';
            document.body.style.userSelect = '';
            if (lastGraphH > 0) window._pendingDragHeight = lastGraphH;
        });

        return window.dash_clientside.no_update;
    }
    """,
    Output("drag-gutter-inited", "children"),
    Input("main-content", "children"),
)


# ── 10g. mouseup 後把拖動高度寫回 store-graph-height ──────
app.clientside_callback(
    """
    function(n) {
        if (!window._pendingDragHeight) return window.dash_clientside.no_update;
        var h = window._pendingDragHeight;
        window._pendingDragHeight = null;
        return h;
    }
    """,
    Output("store-graph-height", "data", allow_duplicate=True),
    Input("interval-upload-check", "n_intervals"),
    prevent_initial_call=True,
)


# ── Layout 按鈕 → store-layout ───────────────────────────
@callback(
    Output("store-layout", "data"),
    Input({"type": "layout-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def set_layout(n_clicks_list):
    if not ctx.triggered_id or not isinstance(ctx.triggered_id, dict):
        raise PreventUpdate
    return ctx.triggered_id["index"]


@callback(
    Output({"type": "layout-btn", "index": ALL}, "style"),
    Input("store-layout", "data"),
)
def sync_layout_btn_styles(layout):
    return [BTN_LAYOUT_ACTIVE if m == layout else BTN_LAYOUT_INACTIVE
            for m in ["1x1", "1x2", "2x1", "2x2"]]


# ── 初始化 / 更新 extra-cell configs ─────────────────────
@callback(
    Output("store-cell-configs", "data"),
    Input("store-files-data", "data"),
    Input("store-layout", "data"),
    State("store-cell-configs", "data"),
    prevent_initial_call=True,
)
def init_cell_configs(files_data, layout, current_configs):
    current_configs = list(current_configs or [{"param": None, "data_type": None}] * 3)
    if not files_data:
        return [{"param": None, "data_type": None}] * 3
    first_df = pd.read_json(
        io.StringIO(list(files_data.values())[0]["df_json"]),
        orient="split", convert_dates=False,
    )
    params = get_available_parameters(first_df)
    if not params:
        return [{"param": None, "data_type": None}] * 3
    n_extra = {"1x1": 0, "1x2": 1, "2x1": 1, "2x2": 3}.get(layout, 0)
    default_params = [params[min(1, len(params) - 1)], params[0], params[min(1, len(params) - 1)]]
    default_dtypes = ["mag_db", "phase", "phase"]
    result = []
    for i in range(3):
        if i >= n_extra:
            result.append({"param": None, "data_type": None})
            continue
        cfg = current_configs[i] if i < len(current_configs) else {}
        p = cfg.get("param")
        dt = cfg.get("data_type")
        if p not in params:
            p = default_params[i]
        dtypes = get_data_types(first_df, p)
        if dt not in dtypes:
            dt = default_dtypes[i] if default_dtypes[i] in dtypes else (dtypes[0] if dtypes else "mag_db")
        result.append({"param": p, "data_type": dt})
    if result == current_configs:
        raise PreventUpdate
    return result


@callback(
    Output("store-cell-configs", "data", allow_duplicate=True),
    Output("dd-param", "value", allow_duplicate=True),
    Output("dd-datatype", "value", allow_duplicate=True),
    Input({"type": "cell-param", "index": ALL}, "value"),
    Input({"type": "cell-datatype", "index": ALL}, "value"),
    State("store-cell-configs", "data"),
    prevent_initial_call=True,
)
def update_cell_configs(params, datatypes, current_configs):
    if not ctx.triggered_id:
        raise PreventUpdate
    current_configs = list(current_configs or [{"param": None, "data_type": None}] * 3)
    sidebar_param = no_update
    sidebar_dtype = no_update
    for i, param_item in enumerate(ctx.inputs_list[0]):
        cell_index = param_item["id"]["index"]
        dt_i = next((j for j, d in enumerate(ctx.inputs_list[1])
                     if d["id"]["index"] == cell_index), i)
        if cell_index == 0:
            sidebar_param = params[i]
            sidebar_dtype = datatypes[dt_i] if dt_i < len(datatypes) else no_update
        else:
            store_idx = cell_index - 1
            if 0 <= store_idx < 3:
                current_configs[store_idx] = {
                    "param": params[i],
                    "data_type": datatypes[dt_i] if dt_i < len(datatypes) else None,
                }
    return current_configs, sidebar_param, sidebar_dtype


@callback(
    Output({"type": "cell-datatype", "index": ALL}, "options"),
    Output({"type": "cell-datatype", "index": ALL}, "value"),
    Input({"type": "cell-param", "index": ALL}, "value"),
    State("store-files-data", "data"),
    State({"type": "cell-datatype", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def update_cell_datatype_opts(params, files_data, current_dtypes):
    if not files_data:
        return [[]] * len(params), [None] * len(params)
    first_df = pd.read_json(
        io.StringIO(list(files_data.values())[0]["df_json"]),
        orient="split", convert_dates=False,
    )
    options_list, values_list = [], []
    for i, param in enumerate(params):
        dtypes = get_data_types(first_df, param) if param else []
        options_list.append(dtypes)
        cur = current_dtypes[i] if i < len(current_dtypes) else None
        values_list.append(cur if cur in dtypes else (dtypes[0] if dtypes else None))
    return options_list, values_list


# ── 11. 主繪圖 callback（只在「曲線相關」參數變動時觸發）─
# marker freq/label/color 改成 State，拖動 marker 不再觸發完整重畫
@callback(
    Output("main-content", "children"),
    Output("tables-area", "children"),
    Output("store-shape-index-map", "data"),
    Output("store-base-trace-count", "data"),
    Output("store-threshold-shape-map", "data"),
    Output("store-threshold-annotations", "data"),
    Input("store-files-data", "data"),
    Input("dd-param", "value"),
    Input("dd-datatype", "value"),
    Input("freq-min", "value"),
    Input("freq-max", "value"),
    Input("amp-min", "value"),
    Input("amp-max", "value"),
    Input("line-width", "value"),
    Input("file-checklist", "value"),
    Input({"type": "threshold-value",       "index": ALL}, "value"),
    Input({"type": "threshold-color",       "index": ALL}, "value"),
    Input({"type": "threshold-show-legend", "index": ALL}, "value"),
    Input("store-layout", "data"),
    Input("store-cell-configs", "data"),
    Input("store-graph-height", "data"),
    State({"type": "marker-freq",        "index": ALL}, "value"),
    State({"type": "marker-color",       "index": ALL}, "value"),
    State({"type": "marker-show-legend", "index": ALL}, "value"),
    State("num-markers", "data"),
    State("num-thresholds", "data"),
    State("store-freq-meta", "data"),
    State({"type": "smith-marker-freq",  "index": ALL}, "value"),
    State({"type": "smith-marker-color", "index": ALL}, "value"),
    State("num-smith-markers", "data"),
)
def update_main(
    files_data, param, data_type,
    fmin, fmax, amp_min, amp_max,
    line_width, visible_filenames,
    t_values, t_colors, t_shows,
    layout, cell_configs, graph_height_data,
    m_freqs, m_colors, m_shows,
    marker_store, threshold_store,
    freq_meta,
    sm_freqs, sm_colors, smith_marker_store,
):
    if not files_data:
        return (
            html.Div("👈 Browse .snp files from the sidebar, or drop files anywhere on the page",
                     style={"padding": "40px", "textAlign": "center", "color": "#6c757d",
                            "backgroundColor": "#f8f9fa", "font-size": "15px",
                            "borderRadius": "8px", "border": "2px dashed #dee2e6"}),
            [], {}, [0, 0, 0, 0], {}, [],
        )
    if not param or not data_type:
        raise PreventUpdate

    parsed_files = _parse_files(files_data)
    visible_filenames = visible_filenames if visible_filenames else list(files_data.keys())
    freq_meta  = freq_meta or {}
    unit       = freq_meta.get("unit", "GHz")
    fmin       = fmin    if fmin    is not None else freq_meta.get("disp_min", 0)
    fmax       = fmax    if fmax    is not None else freq_meta.get("disp_max", 1)
    amp_min    = amp_min if amp_min is not None else -100.0
    amp_max    = amp_max if amp_max is not None else 0.0
    line_width = line_width or 2

    y_unit             = get_y_axis_unit(data_type)
    markers_list       = _build_markers_list(m_freqs, m_colors, m_shows, marker_store, unit)
    threshold_markers_list = _build_threshold_markers_list(t_values, t_colors, t_shows, threshold_store, y_unit)
    smith_markers_list = _build_smith_markers_list(sm_freqs, sm_colors, smith_marker_store, unit)

    # Layout
    layout_cfg   = LAYOUT_GRID.get(layout or "1x1", LAYOUT_GRID["1x1"])
    n_cells      = layout_cfg["cells"]
    n_cols       = layout_cfg["cols"]
    n_rows       = layout_cfg["rows"]
    graph_height = graph_height_data or layout_cfg["graph_height"]

    cell_configs = list(cell_configs or [])
    all_cell_params  = [param]  + [
        (cell_configs[i].get("param")      or param)      if i < len(cell_configs) else param
        for i in range(3)
    ]
    all_cell_dtypes  = [data_type] + [
        (cell_configs[i].get("data_type")  or data_type)  if i < len(cell_configs) else data_type
        for i in range(3)
    ]

    first_df         = list(parsed_files.values())[0]["df"]
    available_params = get_available_parameters(first_df)

    cell_divs               = []
    base_trace_counts       = [0, 0, 0, 0]
    shared_shape_index_map  = {}
    shared_thresh_shape_map = {}
    shared_thresh_ann       = []
    tables_children         = []
    smith_marker_items      = []
    marker_items            = []
    threshold_items         = []
    stats_items             = []

    for cell_idx in range(n_cells):
        cp   = all_cell_params[cell_idx]
        cdt  = all_cell_dtypes[cell_idx]
        cell_y_unit  = get_y_axis_unit(cdt)
        cell_is_smith = cdt == "smith"
        cell_thresh_list = [] if cell_is_smith else threshold_markers_list

        fig, threshold_crossings, btc, tsm = build_base_figure(
            files_data=parsed_files,
            visible_filenames=visible_filenames,
            selected_param=cp,
            selected_data_type=cdt,
            freq_range=(fmin, fmax),
            amp_range=(amp_min, amp_max),
            threshold_markers_list=cell_thresh_list,
            line_width=line_width,
            num_markers=len(markers_list) if not cell_is_smith else len(smith_markers_list),
        )
        base_trace_counts[cell_idx] = btc

        if n_cells > 1:
            fig.update_layout(title=None)

        if cell_idx == 0:
            shared_thresh_ann       = [ann.to_plotly_json() for ann in fig.layout.annotations]
            shared_thresh_shape_map = {str(k): v for k, v in tsm.items()}

        # Marker overlay
        if cell_is_smith:
            sm_traces, cell_marker_values = build_smith_marker_overlays(
                files_data=parsed_files,
                visible_filenames=visible_filenames,
                selected_param=cp,
                smith_markers_list=smith_markers_list,
            )
            for trace in sm_traces:
                fig.add_trace(trace)
            cell_shape_index_map = {}
        else:
            ov_traces, shapes, annotations, cell_marker_values, cell_shape_index_map = compute_marker_overlay(
                files_data=parsed_files,
                visible_filenames=visible_filenames,
                selected_param=cp,
                selected_data_type=cdt,
                markers_list=markers_list,
            )
            for trace in ov_traces:
                fig.add_trace(trace)
            for shape in shapes:
                fig.add_shape(**shape)
            for ann in annotations:
                fig.add_annotation(**ann)

        if cell_idx == 0:
            n_thresh = len(tsm)
            shared_shape_index_map = {str(int(k) + n_thresh): v for k, v in cell_shape_index_map.items()}

        # Cell header
        if cell_idx == 0 and n_cells == 1:
            cell_header = html.Div(
                html.Span(f"{cp}  {cell_y_unit}",
                          style={"fontSize": "12px", "color": "#6c757d", "fontWeight": "500"}),
                style={"marginBottom": "4px"},
            )
        else:
            dt_opts = get_data_types(first_df, cp)
            cell_header = html.Div([
                dcc.Dropdown(
                    id={"type": "cell-param", "index": cell_idx},
                    options=available_params, value=cp, clearable=False,
                    style={"fontSize": "12px", "width": "100px",
                           "display": "inline-block", "marginRight": "6px"},
                ),
                dcc.Dropdown(
                    id={"type": "cell-datatype", "index": cell_idx},
                    options=dt_opts, value=cdt, clearable=False,
                    style={"fontSize": "12px", "width": "110px", "display": "inline-block"},
                ),
            ], style={"marginBottom": "4px"})

        cell_divs.append(html.Div([
            cell_header,
            html.Div(
                dcc.Graph(
                    id={"type": "cell-graph", "index": cell_idx},
                    figure=fig,
                    config={
                        "responsive": True, "displayModeBar": True, "displaylogo": False,
                        "editable": True,
                        "edits": {
                            "shapePosition": True,
                            "annotationPosition": False, "annotationTail": False,
                            "annotationText": False, "axisTitleText": False,
                            "colorbarPosition": False, "legendPosition": False,
                            "legendText": False, "titleText": False,
                        },
                        "toImageButtonOptions": {
                            "format": "png",
                            "filename": f"s_parameter_{cp}_{cdt}",
                            "height": 800, "width": max(1200 // n_cols, 600), "scale": 2,
                        },
                    },
                    style={"height": f"{graph_height}px"},
                ),
                className="graph-cell-container",
                style={"height": f"{graph_height}px", "position": "relative", "overflow": "hidden"},
            ),
        ], style={"border": "1px solid #dee2e6", "borderRadius": "6px",
                  "padding": "6px", "backgroundColor": "#fff"}))

        # Tables: collect per section type
        _cell_label = f"{cp}_{cdt}"
        _cell_hdr = html.Div([
            html.Span(cp,  style={"fontWeight": "600", "marginRight": "6px"}),
            html.Span(cdt, style={"color": "#6c757d", "fontSize": "12px"}),
        ], style={"fontSize": "13px", "color": "#495057",
                  "borderBottom": "1px solid #dee2e6", "paddingBottom": "4px",
                  "marginTop": "12px", "marginBottom": "6px"})
        if cell_is_smith:
            if smith_markers_list and cell_marker_values:
                _, freq_unit_str, _ = auto_convert_frequency(first_df, return_unit_factor=True)
                smith_df = build_smith_marker_table(cell_marker_values, visible_filenames, freq_unit_str)
                if not smith_df.empty:
                    smith_marker_items.append((_cell_hdr, df_to_dash_table(smith_df, f"table-smith-markers-{cell_idx}")))
        else:
            if markers_list and cell_marker_values:
                marker_df = build_marker_table(cell_marker_values, visible_filenames, unit, cell_y_unit)
                marker_items.append((_cell_hdr, df_to_dash_table(marker_df, f"table-markers-{cell_idx}")))
            thresh_tbl = _get_threshold_table(threshold_markers_list, threshold_crossings, visible_filenames, unit, cell_idx)
            if thresh_tbl is not None:
                threshold_items.append((_cell_hdr, thresh_tbl))
            stats_tbl = _get_stats_table(parsed_files, visible_filenames, _cell_label, fmin, fmax, cell_idx)
            if stats_tbl is not None:
                stats_items.append((_cell_hdr, stats_tbl))

    # Assemble tables grouped by section type
    _h3_style = {"fontSize": "16px", "marginTop": "20px"}
    for _title, _items in [
        ("📊 Smith Marker Values",            smith_marker_items),
        ("📊 Marker Values Overview",         marker_items),
        ("📐 Threshold Crossing Frequencies", threshold_items),
        ("📊 Data Statistics",                stats_items),
    ]:
        if _items:
            tables_children.append(html.H3(_title, style=_h3_style))
            for _hdr, _tbl in _items:
                tables_children += [_hdr, _tbl]

    # Raw data
    raw_sections = []
    for fname in visible_filenames:
        if fname not in parsed_files:
            continue
        df_conv, _ = auto_convert_frequency(parsed_files[fname]["df"])
        df_filt = filter_by_frequency_range(df_conv, fmin, fmax)
        raw_sections.append(html.Details([
            html.Summary(f"📄 {get_display_name(fname)}",
                         style={"cursor": "pointer", "fontSize": "14px", "marginBottom": "6px"}),
            df_to_dash_table(df_filt.round(6), f"raw-{fname}"),
        ], style={"marginBottom": "12px"}))
    if raw_sections:
        tables_children += [
            html.H3("📋 Raw Data", style=_h3_style),
            html.Div(raw_sections),
        ]

    content_children = [
        html.Div(cell_divs, id="cell-grid", **{"data-rows": str(n_rows)}, style={
            "display": "grid",
            "gridTemplateColumns": f"repeat({n_cols}, 1fr)",
            "gap": "8px",
        }),
    ]

    return (
        content_children,
        tables_children,
        shared_shape_index_map,
        base_trace_counts,
        shared_thresh_shape_map,
        shared_thresh_ann,
    )


# ── 12. 圖上拖動 marker/threshold → 同步 sidebar ──────────
@callback(
    Output({"type": "marker-freq",    "index": ALL}, "value"),
    Output({"type": "threshold-value","index": ALL}, "value"),
    Input({"type": "cell-graph", "index": ALL}, "relayoutData"),
    State("store-shape-index-map",     "data"),
    State("store-threshold-shape-map", "data"),
    State({"type": "marker-freq",    "index": ALL}, "value"),
    State({"type": "threshold-value","index": ALL}, "value"),
    prevent_initial_call=True,
)
def sync_marker_from_drag(
    relayout_data_list,
    shape_index_map, threshold_shape_map,
    current_freqs, current_thresholds,
):
    triggered = ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        raise PreventUpdate

    triggered_cell = triggered["index"]
    relayout_data = None
    for i, item in enumerate(ctx.inputs_list[0]):
        if item["id"]["index"] == triggered_cell:
            relayout_data = relayout_data_list[i]
            break

    if not relayout_data:
        raise PreventUpdate

    # 收集所有被拖動的 shape 的座標變化
    shape_coords: dict[int, dict[str, float]] = {}
    for key, val in relayout_data.items():
        if not key.startswith("shapes["):
            continue
        try:
            shape_idx = int(key.split("[")[1].split("]")[0])
            coord_key = key.split("].")[-1]   # x0 / x1 / y0 / y1
            if coord_key not in ("x0", "x1", "y0", "y1"):
                continue
            if shape_idx not in shape_coords:
                shape_coords[shape_idx] = {}
            shape_coords[shape_idx][coord_key] = float(val)
        except (IndexError, ValueError):
            continue

    if not shape_coords:
        raise PreventUpdate

    new_freqs      = list(current_freqs)
    new_thresholds = list(current_thresholds)
    freq_updated = False
    thresh_updated = False

    for shape_idx, coords in shape_coords.items():
        s_key = str(shape_idx)

        # ── 垂直 marker：x0/x1 → marker-freq ──
        if shape_index_map and s_key in shape_index_map:
            x_vals = [v for k, v in coords.items() if k in ("x0", "x1")]
            if x_vals:
                new_x = round(sum(x_vals) / len(x_vals), 4)
                marker_idx = shape_index_map[s_key]
                if marker_idx < len(new_freqs):
                    new_freqs[marker_idx] = new_x
                    freq_updated = True

        # ── 水平 threshold：y0/y1 → threshold-value ──
        elif threshold_shape_map and s_key in threshold_shape_map:
            y_vals = [v for k, v in coords.items() if k in ("y0", "y1")]
            if y_vals:
                new_y = round(sum(y_vals) / len(y_vals), 2)
                t_id_str = threshold_shape_map[s_key]
                # ctx.states_list[3] 對應第 4 個 State：threshold-value ALL
                for t_idx, item in enumerate(ctx.states_list[3]):
                    if str(item["id"]["index"]) == t_id_str:
                        if t_idx < len(new_thresholds):
                            new_thresholds[t_idx] = new_y
                            thresh_updated = True
                        break

    if not freq_updated and not thresh_updated:
        raise PreventUpdate

    return new_freqs, new_thresholds


# ── 12a. marker 變動 → 輕量更新所有 cell overlay trace + 表格 ─
@callback(
    Output({"type": "cell-graph", "index": ALL}, "figure"),
    Output("tables-area", "children", allow_duplicate=True),
    Output("store-shape-index-map", "data", allow_duplicate=True),
    Input({"type": "marker-freq",        "index": ALL}, "value"),
    Input({"type": "marker-color",       "index": ALL}, "value"),
    Input({"type": "marker-show-legend", "index": ALL}, "value"),
    Input("num-markers", "data"),
    Input({"type": "smith-marker-freq",  "index": ALL}, "value"),
    Input({"type": "smith-marker-color", "index": ALL}, "value"),
    Input("num-smith-markers", "data"),
    State("store-files-data", "data"),
    State("dd-param", "value"),
    State("dd-datatype", "value"),
    State("file-checklist", "value"),
    State("store-base-trace-count", "data"),
    State("store-freq-meta", "data"),
    State("freq-min", "value"),
    State("freq-max", "value"),
    State("store-threshold-shape-map", "data"),
    State("store-threshold-annotations", "data"),
    State({"type": "threshold-value", "index": ALL}, "value"),
    State({"type": "threshold-color", "index": ALL}, "value"),
    State("num-thresholds", "data"),
    State("store-cell-configs", "data"),
    prevent_initial_call=True,
)
def update_marker_overlay(
    m_freqs, m_colors, m_shows,
    marker_store,
    sm_freqs, sm_colors,
    smith_marker_store,
    files_data, param, data_type,
    visible_filenames, base_trace_counts,
    freq_meta, fmin, fmax,
    threshold_shape_map,
    threshold_annotations_stored,
    t_values, t_colors,
    threshold_store,
    cell_configs,
):
    if not files_data or not param or not data_type:
        raise PreventUpdate

    from dash import Patch

    # Normalise base_trace_counts to list
    if not isinstance(base_trace_counts, list):
        base_trace_counts = [base_trace_counts or 0] * 4

    freq_meta = freq_meta or {}
    unit = freq_meta.get("unit", "GHz")
    y_axis_unit_c0 = get_y_axis_unit(data_type)
    n_threshold_shapes = len(threshold_shape_map) if threshold_shape_map else 0
    threshold_annotations_stored = threshold_annotations_stored or []

    parsed_files = _parse_files(files_data)
    visible_filenames = visible_filenames if visible_filenames else list(files_data.keys())
    fmin = fmin if fmin is not None else freq_meta.get("disp_min", 0)
    fmax = fmax if fmax is not None else freq_meta.get("disp_max", 1)

    threshold_markers_list = _build_threshold_markers_list(t_values, t_colors, None, threshold_store, y_axis_unit_c0)

    _invisible_shape = dict(
        type="line", x0=0, x1=0, y0=0, y1=0, xref="paper", yref="paper",
        line=dict(color="rgba(0,0,0,0)", width=0), opacity=0,
    )
    _empty_scatter      = {"type": "scatter",     "x": [],    "y": [],    "showlegend": False, "hoverinfo": "skip"}
    _empty_scattersmith = {"type": "scattersmith", "real": [], "imag": [], "showlegend": False, "hoverinfo": "skip"}

    cell_configs = list(cell_configs or [])

    def cell_param_dtype(cell_idx):
        if cell_idx == 0:
            return param, data_type
        cfg = cell_configs[cell_idx - 1] if cell_idx - 1 < len(cell_configs) else {}
        return cfg.get("param") or param, cfg.get("data_type") or data_type

    patches = []
    tables_children = []
    smith_marker_items = []
    marker_items       = []
    threshold_items    = []
    stats_items        = []
    updated_shape_index_map = {}

    for output_item in ctx.outputs_list[0]:
        cell_idx = output_item["id"]["index"]
        cp, cdt = cell_param_dtype(cell_idx)
        cell_is_smith = cdt == "smith"
        cell_btc = base_trace_counts[cell_idx] if cell_idx < len(base_trace_counts) else 0
        cell_y_unit = get_y_axis_unit(cdt)

        _cell_thresh_list = [] if cell_is_smith else threshold_markers_list
        _cell_thresh_crossings = _compute_threshold_crossings(
            _cell_thresh_list, parsed_files, f"{cp}_{cdt}", fmin, fmax,
        ) if _cell_thresh_list else {}
        _cell_hdr = html.Div([
            html.Span(cp,  style={"fontWeight": "600", "marginRight": "6px"}),
            html.Span(cdt, style={"color": "#6c757d", "fontSize": "12px"}),
        ], style={"fontSize": "13px", "color": "#495057",
                  "borderBottom": "1px solid #dee2e6", "paddingBottom": "4px",
                  "marginTop": "12px", "marginBottom": "6px"})

        patched_fig = Patch()

        if cell_is_smith:
            smith_markers_list = _build_smith_markers_list(sm_freqs, sm_colors, smith_marker_store, unit)
            for i in range(50):
                patched_fig["data"][cell_btc + i] = _empty_scattersmith

            if not smith_markers_list:
                patches.append(patched_fig)
                continue

            ov_traces, marker_values = build_smith_marker_overlays(
                files_data=parsed_files,
                visible_filenames=visible_filenames,
                selected_param=cp,
                smith_markers_list=smith_markers_list,
            )
            for i, trace in enumerate(ov_traces):
                patched_fig["data"][cell_btc + i] = trace.to_plotly_json()

            if marker_values:
                _, freq_unit_str, _ = auto_convert_frequency(
                    list(parsed_files.values())[0]["df"], return_unit_factor=True,
                )
                smith_df = build_smith_marker_table(marker_values, visible_filenames, freq_unit_str)
                if not smith_df.empty:
                    smith_marker_items.append((_cell_hdr, df_to_dash_table(smith_df, f"table-smith-markers-{cell_idx}")))

        else:
            # Non-Smith
            if not m_freqs or not marker_store:
                patched_fig["layout"]["annotations"] = list(threshold_annotations_stored)
                for i in range(50):
                    patched_fig["layout"]["shapes"][n_threshold_shapes + i] = _invisible_shape
                for i in range(50):
                    patched_fig["data"][cell_btc + i] = _empty_scatter
                thresh_tbl = _get_threshold_table(_cell_thresh_list, _cell_thresh_crossings, visible_filenames, unit, cell_idx)
                if thresh_tbl is not None:
                    threshold_items.append((_cell_hdr, thresh_tbl))
                stats_tbl = _get_stats_table(parsed_files, visible_filenames, f"{cp}_{cdt}", fmin, fmax, cell_idx)
                if stats_tbl is not None:
                    stats_items.append((_cell_hdr, stats_tbl))
                patches.append(patched_fig)
                continue

            markers_list = _build_markers_list(m_freqs, m_colors, m_shows, marker_store, unit)
            ov_traces, shapes, annotations, marker_values, cell_sim = compute_marker_overlay(
                files_data=parsed_files,
                visible_filenames=visible_filenames,
                selected_param=cp,
                selected_data_type=cdt,
                markers_list=markers_list,
            )

            for i, s in enumerate(shapes):
                patched_fig["layout"]["shapes"][n_threshold_shapes + i] = s
            for i in range(len(shapes), 50):
                patched_fig["layout"]["shapes"][n_threshold_shapes + i] = _invisible_shape
            patched_fig["layout"]["annotations"] = list(threshold_annotations_stored) + list(annotations)

            new_data = [t.to_plotly_json() for t in ov_traces]
            for i in range(50):
                patched_fig["data"][cell_btc + i] = _empty_scatter
            for i, trace_json in enumerate(new_data):
                patched_fig["data"][cell_btc + i] = trace_json

            if markers_list and marker_values:
                marker_df = build_marker_table(marker_values, visible_filenames, unit, cell_y_unit)
                marker_items.append((_cell_hdr, df_to_dash_table(marker_df, f"table-markers-{cell_idx}")))
            thresh_tbl = _get_threshold_table(_cell_thresh_list, _cell_thresh_crossings, visible_filenames, unit, cell_idx)
            if thresh_tbl is not None:
                threshold_items.append((_cell_hdr, thresh_tbl))
            stats_tbl = _get_stats_table(parsed_files, visible_filenames, f"{cp}_{cdt}", fmin, fmax, cell_idx)
            if stats_tbl is not None:
                stats_items.append((_cell_hdr, stats_tbl))
            updated_shape_index_map = {
                str(int(k) + n_threshold_shapes): v for k, v in cell_sim.items()
            }

        patches.append(patched_fig)

    # Assemble tables grouped by section type
    _h3_style = {"fontSize": "16px", "marginTop": "20px"}
    for _title, _items in [
        ("📊 Smith Marker Values",            smith_marker_items),
        ("📊 Marker Values Overview",         marker_items),
        ("📐 Threshold Crossing Frequencies", threshold_items),
        ("📊 Data Statistics",                stats_items),
    ]:
        if _items:
            tables_children.append(html.H3(_title, style=_h3_style))
            for _hdr, _tbl in _items:
                tables_children += [_hdr, _tbl]

    return patches, tables_children, updated_shape_index_map


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, port=8501)