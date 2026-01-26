🎯 Constant Hunter

📋 Project Overview

Constant Hunter is a high-performance GPU-accelerated search engine written in CUDA/C++, designed to find physical constants within massive datasets (like Pi digits). This repository contains a demonstration version showcasing my advanced CUDA programming and parallel computing skills.

<img width="1920" height="1039" alt="image" src="https://github.com/user-attachments/assets/202c404d-84be-4b2b-b566-a7d947861a52" />

⚠️ Intellectual Property Notice

THIS IS DEMONSTRATION CODE - Contains only basic architecture and example functionality. Advanced optimizations, specialized kernels, and proprietary techniques are PROTECTED AND NOT INCLUDED.

_____________________________________________________________________________________________________________

✨ Features (Full version)

    🚀 GPU Acceleration: 1GB search in <1 second using CUDA

    🔍 Multi-constant Search: Predefined physical constants + custom patterns

    🎨 Interactive GUI: PyQt6-based interface with real-time visualizations

    📊 Data Analysis: Statistical analysis and result export (CSV/JSON)

    ⚡ Cross-platform: Optimized for Fedora Linux with CUDA 13.1

<img width="1920" height="1039" alt="image" src="https://github.com/user-attachments/assets/151a82d0-3dc4-4eba-9a71-8b014d77a7da" />


🚀 Quick Start
Prerequisites

    NVIDIA GPU with CUDA support

    Fedora 43 (recommended) or compatible Linux

    CUDA Toolkit 13.1+

    Python 3.10+


📊 Performance Metrics (Full version)
    
    Metric	Value
    Throughput	20-25 GB/s
    Max File Size	2.5GB+ (chunked processing)
    Search Accuracy	100% exact matches
    GPU Memory Usage	~300MB per 1GB search
    Supported Constants	50+ physical constants

🛠️ Tech Stack (Full version)

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

📈 Supported Constants

    Fundamental: c (speed of light), h (Planck), G (gravitational)

    Electromagnetic: α (fine structure), e (electron charge)

    Thermodynamic: k (Boltzmann), R (gas constant)

    Mathematical: φ (golden ratio), π, e

    Custom: Any numeric pattern (up to 20 digits)

<img width="1919" height="1040" alt="image" src="https://github.com/user-attachments/assets/df432662-5953-4313-9805-4c2f2e01ccf0" />



🛡️ License

TECHNICAL PORTFOLIO LICENSE

    ✅ May be reviewed for skill evaluation

    ✅ May be compiled and run locally

    ❌ MAY NOT be used commercially

    ❌ MAY NOT be modified or redistributed

    ❌ DOES NOT include proprietary optimizations

View full license
🎯 Demo Version Features
Included (in this demo):

    Core system architecture

    Basic CUDA search kernel

    GPU/CPU memory management

    Simple results system

    3 test physical constants

NOT Included (full version):

    Memory coalescing optimized kernels

    Warp-level optimizations

    Advanced shared memory patterns

    Multi-stream processing

    Texture memory optimizations

    15+ physical constant searches

    320-480 GB/s throughput

    📊 Performance Metrics (Full Version)
    Metric	Demo Version	Full Version*
    Throughput	~10 GB/s	30 GB/s
    Constants	3	15+
    Data Size	KB	100+ GB
    Optimizations	Basic	Advanced

*Available under NDA
🏗️ Technical Architecture

    System Architecture (simplified):
    ├── Host (CPU)
    │   ├── I/O Management
    │   ├── Data Preparation
    │   └── Pipeline Control
    └── Device (GPU)
        ├── Memory Manager
        ├── Search Engine (kernels)
        └── Results Collector

🔧 Compilation & Execution

        # Compile demo version
    make
    
    # Run
    ./constant_hunter_demo



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

# Constant Hunter - Motor de Optimización CUDA

📞 Access to Full Version

Complete code with all optimizations available for:
1. Hiring Processes

    Available under NDA

    Full review in technical interviews

    Live demonstration

2. Commercial Licensing

    For production use

    Customization available

    Technical support

Contact: [vexhive@tuta.io]

🎓 Demonstrated Skills

    Advanced CUDA programming

    Hierarchical memory optimization

    Massively parallel GPU computing

    Parallel algorithm design

    GPU profiling and debugging

    Large-scale data management

📁 Code Structure

    constant_hunter_demo/
    ├── constant_hunter_demo.cu  # Main source code
    ├── Makefile                 # Build script
    ├── LICENSE                  # Restrictive license
    ├── README.md               # This file
    └── docs/                   # Technical documentation
        ├── ARCHITECTURE.md     # System design
        └── PERFORMANCE.md      # Metrics & benchmarks

⚡ Technical Features (Full Version)
<details> <summary>🔒 Click for technical details (no code)</summary>
Optimizations Implemented:

    Memory Coalescing: Aligned global memory accesses

    Warp Shuffle: Intra-warp communication without shared memory

    Bank Conflict Avoidance: Optimized access patterns

    Constant Caching: Optimal use of constant memory

    Stream Overlap: Concurrent computation and transfer

Specialized Algorithms:

    Parallel multi-pattern search

    Probabilistic filtering (GPU Bloom filters)

    In-GPU compression for repetitive data

    Hierarchical pattern caching

Scalability:

    Multi-GPU support

    Dynamic load balancing

    Partial fault tolerance

    Automatic checkpointing

</details>
🤝 Collaboration

Interested in:

    CUDA developer positions

    HPC research projects

    GPU optimization consulting

    High-performance system development

Available for: Full-time positions, contract work, or consulting.
