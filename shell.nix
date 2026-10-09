{ pkgs ? import <nixpkgs> {} }:

let
  name = "npcgym";
  # MONA is a native executable, separately from the optional Python compiler.
  mona = pkgs.stdenv.mkDerivation {
    pname = "mona";
    version = "1.4-18";
    src = pkgs.fetchurl {
      url = "https://www.brics.dk/mona/download/mona-1.4-18.tar.gz";
      sha256 = "ece10e1e257dcae48dd898ed3da48f550c6b590f8e5c5a6447d0f384ac040e4c";
    };
    nativeBuildInputs = [ pkgs.flex pkgs.bison ];
  };
in
(pkgs.buildFHSEnv {
  inherit name;

  targetPkgs = pkgs: [
    mona
    pkgs.uv
    pkgs.libGL
    pkgs.libGLU
    pkgs.zlib
    # Pygame's bundled SDL loads desktop display libraries at runtime. Without
    # these it can silently fall back to an invisible offscreen window.
    pkgs.libx11
    pkgs.libxext
    pkgs.libxcursor
    pkgs.libxi
    pkgs.libxfixes
    pkgs.libxrandr
    pkgs.libxscrnsaver
    pkgs.wayland
    pkgs.libxkbcommon
    pkgs.xdummy
    pkgs.ghostscript
    pkgs.flamegraph
  ];

  runScript = "bash";

  profile = ''
    # Anchored to this file's directory so entering from another working
    # directory cannot create or clear an unrelated `.venv`.
    project_root="${toString ./.}"
    venv="$project_root/.venv"
    stamp="$venv/.npcgym-environment"
    fingerprint="$(${pkgs.coreutils}/bin/sha256sum "$project_root/pyproject.toml" "$project_root/environment.yml" "$project_root/shell.nix" | ${pkgs.coreutils}/bin/sha256sum | ${pkgs.coreutils}/bin/cut -d ' ' -f 1)"

    unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL
    unset MAMBA_EXE MAMBA_ROOT_PREFIX UV_PYTHON VIRTUAL_ENV
    export MPLCONFIGDIR="$venv/.cache/matplotlib"
    export UV_PYTHON_DOWNLOADS=automatic
    export UV_PYTHON_PREFERENCE=only-managed
    export UV_LINK_MODE=copy

    if [ ! -f "$stamp" ] || [ "$(<"$stamp")" != "$fingerprint" ] \
      || ! "$venv/bin/python" -c "" 2>/dev/null; then
      echo "Creating NPC Gym development environment..."
      if ! uv venv --clear --python 3.13 "$venv"; then
        # uv keeps a managed interpreter registered after its files disappear,
        # so it will not re-download one; force that before the second attempt.
        uv python install --reinstall 3.13
        uv venv --clear --python 3.13 "$venv"
      fi
      # NumPy pin: keep in step with environment.yml and the Tox typing environment.
      if [ ! -x "$venv/bin/python" ] \
        || ! uv pip install --python "$venv/bin/python" "numpy>=2,<2.3" -e "$project_root[asp,dev,ltlf,optuna,plots,render,sb3,test]"; then
        echo "Failed to create the NPC Gym development environment." >&2
        exit 1
      fi
      echo "$fingerprint" > "$stamp"
    fi

    if ! source "$venv/bin/activate" \
      || [ "$(command -v python)" != "$venv/bin/python" ] \
      || ! python -c "" 2>/dev/null; then
      echo "Failed to activate the NPC Gym development environment." >&2
      exit 1
    fi
  '';
}).env
