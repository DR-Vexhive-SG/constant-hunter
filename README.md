🎯 Constant Hunter

GPU-accelerated search engine for physical constants in irrational numbers with scietific precision and speed

https://img.shields.io/badge/License-Apache_2.0-blue.svg
https://img.shields.io/badge/CUDA-13.1-green.svg
https://img.shields.io/badge/Python-3.10+-yellow.svg
https://img.shields.io/badge/Platform-Fedora%252043-orange.svg
✨ Features

    🚀 GPU Acceleration: 1GB search in <1 second using CUDA

    🔍 Multi-constant Search: Predefined physical constants + custom patterns

    🎨 Interactive GUI: PyQt6-based interface with real-time visualizations

    📊 Data Analysis: Statistical analysis and result export (CSV/JSON)

    ⚡ Cross-platform: Optimized for Fedora Linux with CUDA 13.1

🚀 Quick Start
Prerequisites

    NVIDIA GPU with CUDA support

    Fedora 43 (recommended) or compatible Linux

    CUDA Toolkit 13.1+

    Python 3.10+

Installation

# Clone repository
git clone [https://github.com/DR-Vexhive-SG/constant-hunter]
cd constant-hunter

# Run setup script
chmod +x setup_constant_hunter.sh
./setup_constant_hunter.sh

# Launch application
python src/python/run_gui.py
📊 Performance Metrics
Metric	Value
Throughput	20-25 GB/s
Max File Size	1GB+ (chunked processing)
Search Accuracy	100% exact matches
GPU Memory Usage	~300MB per 1GB search
Supported Constants	50+ physical constants
🛠️ Tech Stack

    GPU Computing: CUDA 13.1+, NVIDIA drivers

    Frontend: PyQt6, Matplotlib, Seaborn

    Backend: Python 3.10+, NumPy, SciPy

    Compilation: GCC 14, CMake

    OS: Fedora 43 (optimized), Linux

📁 Project Structure
constant-hunter/
├── src/
│   ├── cuda/           # CUDA kernels (.cu files)
│   ├── python/         # Python application
│   └── scripts/        # Utility scripts
├── docs/              # Documentation
├── datasets/          # Sample datasets
├── tests/            # Unit tests
└── results/          # Search outputs (.gitignored)
🧪 Usage Examples
GUI Mode
python src/python/run_gui.py
    Select digit file (π, e, φ, √2, or custom)

    Choose constants from database

    View interactive results with charts

CLI Mode
python src/python/cuda_wrapper.py \
    --file datasets/Pi\ -\ Dec.txt \
    --constant c,G,k \
    --output results/latest_search
Benchmark
./src/scripts/run_benchmark.sh
📈 Supported Constants

    Fundamental: c (speed of light), h (Planck), G (gravitational)

    Electromagnetic: α (fine structure), e (electron charge)

    Thermodynamic: k (Boltzmann), R (gas constant)

    Mathematical: φ (golden ratio), π, e

    Custom: Any numeric pattern (up to 20 digits)

🔧 Configuration
Fedora 43 Specific
# Always compile with gcc-14
nvcc -ccbin /usr/bin/gcc-14 -O3 ...
Memory Management

    Chunk size: 1GB default (adjustable)

    GPU memory: Auto-detected

    Cache: 256MB file cache

📚 Documentation

    Full Installation Guide

    Technical Architecture

    API Reference (TODO)

    Constants Database (TODO)

🤝 Contributing

    Fork the repository

    Create a feature branch

    Commit changes

    Push to branch

    Open Pull Request

📄 License
Apache 2.0 - See LICENSE for details.

👨‍💻 Author

Daniel Ricardo Segura González

    Email: vexhive@tuta.io

    GitHub: DR-Vexhive-SG

🙏 Acknowledgments

    NVIDIA CUDA Toolkit

    PyQt6 Development Team

    NIST CODATA for physical constants

    Fedora Project

⭐ If you find this project useful, please give it a star!
