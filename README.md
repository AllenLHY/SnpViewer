# S-Parameter Viewer

A web-based Dash app for viewing and analyzing S-parameter files (.snp format).

## Features

- Support multiple .snp files (.s1p, .s2p, .s4p, etc.)
- Interactive plotting with Plotly
- Multi-cell layout (1×1, 1×2, 2×1, 2×2) with per-cell parameter selection
- Resizable plot/table split via drag handle
- Frequency markers with drag-to-set and Smith chart support
- Threshold markers with crossing detection
- Smith chart view
- Statistics and raw data tables

## How to Use

1. Upload .snp files via the sidebar (drag & drop or browse)
2. Select parameters and data type from the sidebar or inline cell dropdowns
3. Switch layout (1×1 / 1×2 / 2×1 / 2×2) from the toolbar
4. Add markers for frequency analysis; drag markers directly on the chart
5. Drag the divider bar to resize the plot vs table area

## Technology Stack

- **Dash / Plotly**: Web framework and interactive charts
- **scikit-rf**: S-parameter file parsing
- **Pandas**: Data processing

## Local Development

```bash
pip install -r requirements.txt
python snp_viewer_dash.py
```

## Changelog

### v1.1.0
- Multi-cell layout support (1×1 / 1×2 / 2×1 / 2×2) with shared markers
- Per-cell inline param/datatype dropdowns
- Viewport-adaptive graph height with drag-to-resize plot/table boundary
- Fixed graph overflow and table overlap issues
- Fixed marker not appearing after layout switch

### v1.0.4
- Smith marker click-to-set; del/aim button hover effects

### v1.0.3
- Smith marker table: show Γ, Z(Ω), Y(mS) as complex strings; read Z₀ from file

### v1.0.2
- Threshold Peak/−3dB/Dip buttons; fix param/precision/step bugs

### v1.0.1
- Threshold crossings respect freq range; UI title version, empty state msg

## License

MIT License