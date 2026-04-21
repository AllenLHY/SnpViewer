import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import warnings
import tempfile
import skrf as rf
import numpy as np

# 抑制 Plotly 的過時參數警告
warnings.filterwarnings('ignore', message='.*keyword arguments have been deprecated.*')

st.set_page_config(
    page_title="SNP Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 S-parameter Viewer")
st.write("Upload multiple .snp files (.s1p, .s2p, .s4p, etc.) for plotting comparison and analysis")

# ============== 解析 S-parameter 檔案的函數 ==============

@st.cache_data(ttl=3600)
def parse_sparameter_file_cached(filename, file_content):
    try:
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        # 用 SKRF 解析
        network = rf.Network(tmp_path)
        
        # 清理臨時檔案
        os.remove(tmp_path)
        
        freq = network.frequency.f
        n_ports = network.number_of_ports
        data_dict = {'Frequency': freq}
        for i in range(n_ports):
            for j in range(n_ports):
                s_param = network.s[:, i, j]
                param_name = f'S{i+1}{j+1}'
                data_dict[f'{param_name}_mag'] = 20 * np.log10(np.abs(s_param) + 1e-12)
                data_dict[f'{param_name}_phase'] = np.angle(s_param, deg=True)
        return pd.DataFrame(data_dict)
    except Exception as e:
        raise Exception(f"SKRF parsing failed: {str(e)}")


def get_available_parameters(df):
    """取得可用的參數列表(S11, S21, S12, S22)"""
    cols = df.columns.tolist()
    cols.remove('Frequency')
    
    # 提取參數類型(S11, S21 等)
    param_types = set()
    for col in cols:
        if '_' in col:
            param_type = col.split('_')[0]  # S11, S21 等
            param_types.add(param_type)
    
    return sorted(list(param_types))


def get_data_types(df, param_type):
    """取得特定參數的數據類型(mag, phase)"""
    cols = df.columns.tolist()
    data_types = set()
    for col in cols:
        if col.startswith(param_type + '_'):
            data_type = col.split('_')[1]  # mag, phase
            data_types.add(data_type)
    
    return sorted(list(data_types))


def auto_convert_frequency(df, return_original_unit=False):
    """自動轉換頻率單位(Hz -> MHz 或 GHz),並返回轉換後的 DataFrame 和單位"""
    freq = df['Frequency'].copy()
    max_freq = freq.max()
    
    if max_freq >= 1e9:  # GHz
        converted_freq = freq / 1e9
        unit = 'GHz'
        original_unit = 1e9
    elif max_freq >= 1e6:  # MHz
        converted_freq = freq / 1e6
        unit = 'MHz'
        original_unit = 1e6
    elif max_freq >= 1e3:  # kHz
        converted_freq = freq / 1e3
        unit = 'kHz'
        original_unit = 1e3
    else:  # Hz
        converted_freq = freq
        unit = 'Hz'
        original_unit = 1
    
    df_copy = df.copy()
    df_copy['Frequency'] = converted_freq
    
    if return_original_unit:
        return df_copy, unit, original_unit
    return df_copy, unit


def get_display_name(filename):
    """從檔案名稱中移除副檔名"""
    return os.path.splitext(filename)[0]


def get_y_axis_unit(data_type):
    """根據數據類型返回 Y 軸單位"""
    if 'mag' in data_type.lower():
        return 'dB'
    elif 'phase' in data_type.lower():
        return 'deg'
    else:
        return ''


def filter_by_frequency_range(df, freq_min, freq_max):
    """根據頻率範圍過濾數據"""
    mask = (df['Frequency'] >= freq_min) & (df['Frequency'] <= freq_max)
    return df[mask].copy()


def find_nearest_value(df, freq_target, param_full):
    """找到最接近目標頻率的數值"""
    idx = (df['Frequency'] - freq_target).abs().idxmin()
    return df.loc[idx, 'Frequency'], df.loc[idx, param_full]


def add_markers_to_plot(fig, markers_list, visible_files, selected_param_full, freq_unit, y_axis_unit, color_palette):
    """在圖表上添加自訂標記點"""
    
    # 為每個 marker 收集所有檔案的數值
    marker_values = {}  # {marker_label: [(filename, actual_freq, actual_value), ...]}
    
    for marker in markers_list:
        marker_freq = marker['freq']
        marker_label = marker['label']
        marker_color = marker['color']
        marker_style = marker['style']
        show_in_legend = marker.get('show_in_legend', True)  # 預設顯示在 legend
        
        marker_values[marker_label] = []
        
        # 添加垂直線(只添加一次)
        if marker_style in ['vertical', 'both']:
            fig.add_vline(
                x=marker_freq,
                line_dash="dash",
                line_color=marker_color,
                line_width=2,
                opacity=0.7,
                annotation_text=marker_label,
                annotation_position="top",
                annotation_font_size=10,
                annotation_font_color=marker_color
            )
        
        # 對每個可見的檔案,在 marker 頻率處添加標記點
        for idx, (filename, file_info) in enumerate(visible_files.items()):
            df = file_info['df']
            df_converted, _ = auto_convert_frequency(df)
            
            # 找到最接近 marker_freq 的點
            actual_freq, actual_value = find_nearest_value(df_converted, marker_freq, selected_param_full)
            
            display_name = get_display_name(filename)
            marker_values[marker_label].append((display_name, actual_freq, actual_value))
            
            # 取得該線段的顏色
            line_color = color_palette[idx % len(color_palette)]
            
            # 在 legend 中顯示該檔案在此 marker 的數值
            # 格式: 線段名@頻率 單位: 值 單位
            legend_text = f"{display_name} @ {marker_label} {freq_unit}: {actual_value:.3f} {y_axis_unit}"
            
            # 為了在 legend 中顯示 ---◆--- 的效果,我們需要創建一個包含 3 個點的 trace
            # 左側點(透明)- 中心點(菱形)- 右側點(透明)
            # 這樣虛線會貫穿整個圖標
            x_coords = [actual_freq - 0.001, actual_freq, actual_freq + 0.001]  # 三個點
            y_coords = [actual_value, actual_value, actual_value]
            
            # 添加標記點(使用 lines+markers 模式來顯示虛線和菱形的組合)
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines+markers',  # lines+markers 模式
                line=dict(
                    color=marker_color,  # 虛線使用 marker 顏色
                    width=2,
                    dash='dash'  # 虛線樣式
                ),
                marker=dict(
                    size=[0, 12, 0],  # 只有中間的點顯示,左右兩點大小為 0
                    color=line_color,  # 菱形使用線段顏色
                    symbol='diamond',
                    line=dict(width=2, color='white')
                ),
                name=legend_text,
                legendgroup=f"marker_{marker_label}",  # 使用 legendgroup 將同一 marker 的點歸類
                showlegend=show_in_legend,  # 根據設定決定是否顯示在 legend
                hovertemplate=(
                    f"<b>{marker_label}</b><br>" +
                    f"{display_name}<br>" +
                    f"Frequency: {actual_freq:.3f} {freq_unit}<br>" +
                    f"Value: {actual_value:.3f} {y_axis_unit}<br>" +
                    "<extra></extra>"
                ),
                hoverinfo='skip',  # 左右兩個輔助點不顯示 hover
                hoverlabel=dict(namelength=-1)
            ))
            
            # 如果需要水平線,添加在第一個檔案上即可
            if idx == 0 and marker_style == 'horizontal':
                fig.add_hline(
                    y=actual_value,
                    line_dash="dot",
                    line_color=marker_color,
                    line_width=1,
                    opacity=0.5
                )
    
    return marker_values


DEFAULT_MARKER_COLORS = ["#FF0000", "#00FF00", "#0000FF", "#FF00FF", "#FFFF00",
                         "#00FFFF", "#FFA500", "#800080", "#FFC0CB", "#A52A2A"]

# ============== Session State 初始化 ==============
if 'file_checkboxes' not in st.session_state:
    st.session_state.file_checkboxes = {}

# 用於追蹤 marker 標籤和頻率
if 'marker_labels' not in st.session_state:
    st.session_state.marker_labels = {}
if 'marker_freqs' not in st.session_state:
    st.session_state.marker_freqs = {}

# 用於追蹤是否自訂 marker 樣式
if 'custom_marker_style' not in st.session_state:
    st.session_state.custom_marker_style = False


# ============== 側邊欄設置 ==============
st.sidebar.header("⚙️ Settings")

# 上傳檔案
uploaded_files = st.sidebar.file_uploader(
    "📎 Upload .snp files (.s1p, .s2p, .s4p, etc.)",
    type=['s1p', 's2p', 's3p', 's4p', 's5p', 's6p', 'snp'],
    accept_multiple_files=True,
    help="Upload multiple files for plotting comparison"
)

# ============== 主要邏輯 ==============
if not uploaded_files:
    st.info("👈 Please upload .snp files from the left sidebar to start")
else:
    # 解析所有檔案
    files_data = {}
    errors = []
    
    # 進度提示
    progress_placeholder = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            # 更新進度
            progress_placeholder.info(f"⏳ Reading {uploaded_file.name}... ({idx+1}/{len(uploaded_files)})")
            
            content = uploaded_file.read().decode('utf-8')
            df = parse_sparameter_file_cached(uploaded_file.name, content)
            files_data[uploaded_file.name] = {'df': df}
            
        except Exception as e:
            errors.append(f"❌ {uploaded_file.name}: {str(e)}")
    
    # 清除進度提示
    progress_placeholder.empty()
    
    if errors:
        for error in errors:
            st.error(error)
    
    if files_data:
        # 取得第一個檔案的參數選項
        first_df = list(files_data.values())[0]['df']
        available_params = get_available_parameters(first_df)
        
        # 側邊欄:選擇要顯示的參數類型
        st.sidebar.subheader("📈 Select Parameters")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            # 設定預設值為 S21(如果存在)
            default_index = 0
            if 'S21' in available_params:
                default_index = available_params.index('S21')
            
            selected_param = st.selectbox(
                "Parameter Type",
                options=available_params,
                index=default_index,
                help="Select S-parameter (S11, S21, etc.)"
            )
        
        # 取得該參數可用的數據類型
        available_data_types = get_data_types(first_df, selected_param)
        
        with col2:
            selected_data_type = st.selectbox(
                "Data Type",
                options=available_data_types,
                help="Select Magnitude or Phase"
            )
        
        # 完整的列名
        selected_param_full = f"{selected_param}_{selected_data_type}"
        
        # Y 軸單位
        y_axis_unit = get_y_axis_unit(selected_data_type)
        
        # ============== 頻率範圍選擇 ==============
        st.sidebar.subheader("🔍 Frequency Range")
        
        # 取得所有檔案的頻率範圍(使用原始 Hz 單位)
        all_freq_min = min([file_info['df']['Frequency'].min() for file_info in files_data.values()])
        all_freq_max = max([file_info['df']['Frequency'].max() for file_info in files_data.values()])
        
        # 自動判斷頻率單位
        _, freq_unit, freq_multiplier = auto_convert_frequency(first_df, return_original_unit=True)
        
        # 轉換到顯示單位
        display_freq_min = all_freq_min / freq_multiplier
        display_freq_max = all_freq_max / freq_multiplier
        
        # 計算合適的步進值
        step_value = (display_freq_max - display_freq_min) / 1000
        
        # 使用兩個 number_input 讓使用者可以直接輸入
        col_freq1, col_freq2 = st.sidebar.columns(2)
        
        with col_freq1:
            freq_min_input = st.number_input(
                f"Min ({freq_unit})",
                min_value=float(display_freq_min),
                max_value=float(display_freq_max),
                value=float(display_freq_min),
                step=step_value,
                format="%.3f",
                help=f"Enter minimum frequency ({freq_unit})"
            )
        
        with col_freq2:
            freq_max_input = st.number_input(
                f"Max ({freq_unit})",
                min_value=float(display_freq_min),
                max_value=float(display_freq_max),
                value=float(display_freq_max),
                step=step_value,
                format="%.3f",
                help=f"Enter maximum frequency ({freq_unit})"
            )
        
        # 確保最小值不大於最大值
        if freq_min_input > freq_max_input:
            st.sidebar.warning("⚠️ Minimum value cannot be greater than maximum value")
            freq_range = (float(display_freq_min), float(display_freq_max))
        else:
            freq_range = (freq_min_input, freq_max_input)
        
        # 顯示選擇的頻率範圍
        st.sidebar.caption(f"📋 Range: {freq_range[0]:.3f} ~ {freq_range[1]:.3f} {freq_unit}")
        
        # ============== Marker 功能 ==============
        st.sidebar.subheader("📍 Add Marker")
        
        # 讓使用者決定要添加幾個 marker(0 表示不啟用)
        num_markers = st.sidebar.number_input(
            "Markers", 
            min_value=0, 
            max_value=10, 
            value=0, 
            step=1,
            help="Set to 0 to disable markers"
        )
        
        markers_list = []
        if num_markers > 0:
            if not st.session_state.custom_marker_style:
                st.sidebar.caption("💡 Enter marker frequencies")
                # 使用 container 讓輸入區可滾動
                markers_simple_container = st.sidebar.container(height=250)
                with markers_simple_container:
                    for i in range(int(num_markers)):
                        # 計算預設頻率
                        default_freq = float(display_freq_min + (display_freq_max - display_freq_min) * (i+1) / (num_markers+1))
                        
                        # 頻率輸入(簡化版)
                        marker_freq = st.number_input(
                            f"Marker {i+1} ({freq_unit})",
                            min_value=float(display_freq_min),
                            max_value=float(display_freq_max),
                            value=st.session_state.marker_freqs.get(i, default_freq),
                            step=step_value,
                            format="%.3f",
                            key=f"marker_freq_{i}",
                            help=f"Frequency position for Marker {i+1}"
                        )
                        
                        st.session_state.marker_freqs[i] = marker_freq
                        marker_label = f"{marker_freq:.3f}"
                        marker_color = DEFAULT_MARKER_COLORS[i % 10]
                        marker_style = "both"
                        
                        markers_list.append({
                            'freq': marker_freq,
                            'label': marker_label,
                            'color': marker_color,
                            'style': marker_style,
                            'show_in_legend': True  # 簡化模式預設顯示在 legend
                        })
                
                # 在輸入區下方顯示自訂樣式勾選框
                custom_marker_style_new = st.sidebar.checkbox(
                    "🎨 Customize Marker Style", 
                    value=st.session_state.custom_marker_style, 
                    help="Check to customize color and style for each marker",
                    key="custom_marker_style_checkbox"
                )
                # 如果勾選狀態改變,更新 session state 並重新運行
                if custom_marker_style_new != st.session_state.custom_marker_style:
                    st.session_state.custom_marker_style = custom_marker_style_new
                    st.rerun()
            
            # 勾選自訂樣式:顯示完整的摺疊選單
            else:
                # 先顯示取消勾選的選項
                custom_marker_style_new = st.sidebar.checkbox(
                    "🎨 Customize Marker Style", 
                    value=st.session_state.custom_marker_style, 
                    help="Check to customize color and style for each marker",
                    key="custom_marker_style_checkbox"
                )
                # 如果勾選狀態改變,更新 session state 並重新運行
                if custom_marker_style_new != st.session_state.custom_marker_style:
                    st.session_state.custom_marker_style = custom_marker_style_new
                    st.rerun()
                
                markers_container = st.sidebar.container(height=300)
                with markers_container:
                    for i in range(int(num_markers)):
                        with st.expander(f"📌 Marker {i+1}", expanded=(i==0)):
                            # 計算預設頻率
                            default_freq = float(display_freq_min + (display_freq_max - display_freq_min) * (i+1) / (num_markers+1))
                            
                            # 頻率輸入
                            marker_freq = st.number_input(
                                f"Frequency ({freq_unit})",
                                min_value=float(display_freq_min),
                                max_value=float(display_freq_max),
                                value=st.session_state.marker_freqs.get(i, default_freq),
                                step=step_value,
                                format="%.3f",
                                key=f"marker_freq_{i}",
                                help=f"Frequency position for Marker {i+1}"
                            )
                            
                            # 檢查頻率是否改變,如果改變則更新標籤
                            if i not in st.session_state.marker_freqs or st.session_state.marker_freqs[i] != marker_freq:
                                st.session_state.marker_freqs[i] = marker_freq
                                st.session_state.marker_labels[i] = f"{marker_freq:.3f}"
                            
                            # 標籤輸入(使用 session state 中的值)
                            marker_label = st.text_input(
                                "Marker Name",
                                value=st.session_state.marker_labels.get(i, f"{marker_freq:.3f}"),
                                key=f"marker_label_{i}",
                                help="Display name for the marker"
                            )
                            # 更新 session state
                            st.session_state.marker_labels[i] = marker_label
                            
                            col_marker1, col_marker2 = st.columns(2)

                            with col_marker1:
                                marker_color = st.color_picker(
                                    "Color",
                                    value=DEFAULT_MARKER_COLORS[i % 10],
                                    key=f"marker_color_{i}",
                                    help="Marker color"
                                )
                            
                            with col_marker2:
                                marker_style = st.selectbox(
                                    "Style",
                                    options=["vertical", "horizontal", "both"],
                                    index=2,
                                    key=f"marker_style_{i}",
                                    help="vertical: vertical line, horizontal: horizontal line, both: crosshair"
                                )
                            
                            # 是否顯示在 Legend
                            show_in_legend = st.checkbox(
                                "Show in Legend",
                                value=True,
                                key=f"marker_show_legend_{i}",
                                help="Check to show this marker in the chart legend"
                            )
                            
                            markers_list.append({
                                'freq': marker_freq,
                                'label': marker_label,
                                'color': marker_color,
                                'style': marker_style,
                                'show_in_legend': show_in_legend
                            })
        
        # 側邊欄:選擇要顯示的檔案(用 container 讓滾動更順暢)
        st.sidebar.subheader("📋 Select Files to Display")
        
        # 初始化或更新勾選狀態
        current_filenames = list(files_data.keys())
        for filename in current_filenames:
            if filename not in st.session_state.file_checkboxes:
                st.session_state.file_checkboxes[filename] = True
        
        # 檔案選擇欄位(scrollable)
        file_selection_container = st.sidebar.container(height=200)
        with file_selection_container:
            for filename in current_filenames:
                display_name = get_display_name(filename)
                st.session_state.file_checkboxes[filename] = st.checkbox(
                    display_name, 
                    value=st.session_state.file_checkboxes[filename],
                    key=f"checkbox_{filename}"
                )
        
        # 快取管理(側邊欄底部)
        st.sidebar.divider()
        
        # 線條寬度設定
        line_width = st.sidebar.slider("Line Width", min_value=1, max_value=5, value=2, step=1)
        
        st.sidebar.subheader("🗑️ Cache Management")
        if st.sidebar.button("Clear Cache", help="Clear all parsed file caches", key="clear_cache_btn"):
            st.cache_data.clear()
            st.session_state.file_checkboxes = {}
            st.success("✅ Cache cleared")
            st.rerun()
        
        # 顯示快取狀態
        st.sidebar.caption(f"📦 {len(files_data)} file(s) cached")
        
        # 根據勾選狀態篩選可見的檔案
        visible_files = {}
        for filename in current_filenames:
            if st.session_state.file_checkboxes.get(filename, True):
                visible_files[filename] = files_data[filename]
        
        # 創建 Plotly 圖表
        if visible_files:
            fig = go.Figure()
            
            # 使用 Plotly 內建的顏色方案(支持 16+ 顏色)
            color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Bold
            
            for idx, (filename, file_info) in enumerate(visible_files.items()):
                df = file_info['df']
                # 轉換頻率單位
                df_converted, _ = auto_convert_frequency(df)
                
                # 根據頻率範圍過濾數據
                df_filtered = filter_by_frequency_range(df_converted, freq_range[0], freq_range[1])
                
                display_name = get_display_name(filename)
                color = color_palette[idx % len(color_palette)]  # 循環使用顏色
                
                show_line_legend = not markers_list
                
                # 添加數據到圖表
                fig.add_trace(go.Scatter(
                    x=df_filtered['Frequency'],
                    y=df_filtered[selected_param_full],
                    mode='lines',
                    name=display_name,
                    line=dict(color=color, width=line_width),
                    marker=dict(size=4),
                    showlegend=show_line_legend,  # 根據是否有 marker 決定是否顯示
                    hovertemplate=(
                        f"<b>{display_name}</b><br>" +
                        f"Frequency: %{{x:.3f}} {freq_unit}<br>" +
                        f"{selected_param_full}: %{{y:.3f}} {y_axis_unit}<br>" +
                        "<extra></extra>"
                    )
                ))
            
            marker_values = add_markers_to_plot(fig, markers_list, visible_files, selected_param_full, freq_unit, y_axis_unit, color_palette) if markers_list else None
            
            # Layout 設定
            y_title = f"{selected_param_full} ({y_axis_unit})" if y_axis_unit else selected_param_full
            
            base_height = 600
            if markers_list:
                # 計算顯示在 legend 中的 marker 數量
                markers_in_legend = sum(1 for marker in markers_list if marker.get('show_in_legend', True))
                
                # 有 marker 時,每個顯示在 legend 的 marker 會為每個檔案添加一個 legend 項目
                num_legend_items = len(visible_files) * markers_in_legend
                # 每個 legend 項目大約需要 21.5 像素,再加上一些 padding
                legend_height = num_legend_items * 21.5 + 100  # 加 100px 作為上下邊距
                # plot height = max(600, legend_height),但不超過 1100px
                plot_height = max(base_height, min(legend_height, 1100))
            else:
                plot_height = base_height
            
            layout_config = {
                'title': f"{selected_param_full}",
                'xaxis_title': f"Freq ({freq_unit})",
                'yaxis_title': y_title,
                'plot_bgcolor': 'rgba(240, 240, 240, 0.5)',
                'height': plot_height,
                'showlegend': True,
                'legend': {
                    'bgcolor': 'rgba(255, 255, 255, 0.9)',
                    'bordercolor': 'rgba(200, 200, 200, 0.5)',
                    'borderwidth': 1,
                    'orientation': 'v',
                    'yanchor': 'top',
                    'y': 1,
                    'xanchor': 'left',
                    'x': 1.02,
                    'itemsizing': 'constant',  # 保持圖標大小一致
                    'tracegroupgap': 0  # 減少同組之間的間距
                },
                'margin': {'t': 80, 'b': 80, 'l': 80, 'r': 150},
                'xaxis': {
                    'showgrid': True,
                    'gridcolor': 'rgba(200, 200, 200, 0.3)',
                    'gridwidth': 1
                },
                'yaxis': {
                    'showgrid': True,
                    'gridcolor': 'rgba(200, 200, 200, 0.3)',
                    'gridwidth': 1
                }
            }

            fig.update_layout(**layout_config)

            st.plotly_chart(
                fig,
                config={
                    'responsive': True,
                    'displayModeBar': True,
                    'displaylogo': False,
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f's_parameter_{selected_param_full}',
                        'height': 800,
                        'width': 1400,
                        'scale': 2
                    }
                }
            )
            
            if markers_list and marker_values:
                st.subheader("📊 Marker Values Overview")
                
                # 重新組織資料:行為線段,列為各 marker
                # 建立以檔案為 row 的資料結構
                table_data = {}
                
                # 先收集所有檔案名稱
                all_files = list(visible_files.keys())
                
                # 為每個檔案建立一個 row
                for filename in all_files:
                    display_name = get_display_name(filename)
                    table_data[display_name] = {'File': display_name}
                
                # 為每個 marker 添加一列
                for marker_label, values in marker_values.items():
                    # 建立欄位名稱: Value @ 頻率 單位 (數值單位)
                    column_name = f"Value @ {marker_label} {freq_unit} ({y_axis_unit})"
                    
                    # 為每個檔案填入該 marker 的數值
                    for display_name, actual_freq, actual_value in values:
                        if display_name in table_data:
                            table_data[display_name][column_name] = f"{actual_value:.3f}"
                
                # 轉換為 DataFrame
                legend_df = pd.DataFrame(list(table_data.values()))
                
                # 設定檔案名稱為 index
                legend_df = legend_df.set_index('File')
                
                st.dataframe(legend_df, use_container_width=True)
            
            
            # 顯示數據統計(改為表格形式)- 根據頻率範圍過濾
            st.subheader("📊 Data Statistics")
            
            stats_data = []
            for filename, file_info in visible_files.items():
                display_name = get_display_name(filename)
                df = file_info['df']
                # 轉換頻率單位
                df_converted, _ = auto_convert_frequency(df)
                # 根據頻率範圍過濾
                df_filtered = filter_by_frequency_range(df_converted, freq_range[0], freq_range[1])
                
                data = df_filtered[selected_param_full]
                
                stats_data.append({
                    'File Name': display_name,
                    'Max': f"{data.max():.3f}",
                    'Min': f"{data.min():.3f}",
                    'Mean': f"{data.mean():.3f}",
                    'Range': f"{(data.max() - data.min()):.3f}",
                    'Data Points': len(data)
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # 顯示原始數據表格
            with st.expander("📋 Show Raw Data Table"):
                for filename, file_info in visible_files.items():
                    display_name = get_display_name(filename)
                    st.subheader(f"📄 {display_name}")
                    # 顯示過濾後的數據
                    df_converted, _ = auto_convert_frequency(file_info['df'])
                    df_filtered = filter_by_frequency_range(df_converted, freq_range[0], freq_range[1])
                    st.dataframe(df_filtered, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ No files selected")

# ============== 頁腳 ==============
st.divider()
st.caption("💡 Tip: Drag or click the left sidebar to upload .snp files. Use the sidebar to control display parameters and files. Enable marker feature to precisely analyze values at specific frequency points!")