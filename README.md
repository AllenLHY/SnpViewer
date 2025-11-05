# S-Parameter Viewer

A web-based tool for viewing and analyzing S-parameter files (.snp format).

## Features

- 📊 Support multiple .snp files (.s1p, .s2p, .s4p, etc.)
- 📈 Interactive plotting with Plotly
- 🎯 Custom frequency markers
- 📉 Multiple parameter comparison
- 💾 Data caching for better performance
- 📊 Statistics and raw data viewing

## How to Use

1. Upload your .snp files using the sidebar
2. Select parameters (S11, S21, S12, S22, etc.)
3. Choose data type (magnitude or phase)
4. Adjust frequency range
5. Add markers for specific frequency analysis

## Technology Stack

- **Streamlit**: Web framework
- **Plotly**: Interactive charts
- **scikit-rf**: S-parameter file parsing
- **Pandas**: Data processing

## Local Development

```bash
pip install -r requirements.txt
streamlit run snp_viewer.py
```

## License

MIT License