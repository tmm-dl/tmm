FROM mcr.microsoft.com/devcontainers/cpp:ubuntu

RUN <<EOF
set -e
apt update

# Dev Tools
apt install -y cmake cppcheck gdb clang-format-19 clang-tidy-19

# Dependencies
apt install -y libssl-dev

# PyTorch Integration
apt install -y python3-pip
pip install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages

EOF