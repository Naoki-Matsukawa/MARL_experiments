grep -v '^\s*#' requirements.txt | while read -r pkg; do
    pip install "$pkg" || echo "FAIL: $pkg" >&2
done