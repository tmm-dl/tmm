FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04

RUN <<EOF
apt update

# Dev Tools
apt install -y cmake git cppcheck gdb clang-format-19 clang-tidy-19

# PyTorch Integration
apt install -y python3-pip
pip install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages

EOF