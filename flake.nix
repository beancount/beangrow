{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils?ref=main";
    nix-filter.url = "github:numtide/nix-filter?ref=main";

    poetry2nix = {
      url = "github:nix-community/poetry2nix?ref=master";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system:
      let
        # pkgs = inputs.nixpkgs.legacyPackages.${system};
        pkgs = import inputs.nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        python = pkgs.python3;

        builder = (inputs.poetry2nix.lib.mkPoetry2Nix {
          inherit pkgs;
        }).mkPoetryApplication;
      in {
        packages.default = builder {
          inherit python;

          pyproject = ./pyproject.toml;
          poetrylock = ./poetry.lock;

          src = let filter = inputs.nix-filter.lib;
          in filter {
            root = ./.;
            include = [ "README.md" "poetry.lock" "pyproject.toml" "beangrow" ];
            exclude = [ (filter.matchName "__pycache__") ];
          };
        };

        devShells.default = pkgs.mkShell {
          packages = [ python ]
            ++ (with python.pkgs; [ black pip pytest pytest-cov ])
            ++ (with pkgs; [
              engage
              nixpkgs-fmt
              poetry
              pyright
              ruff
            ]) ++ (with pkgs.nodePackages; [ markdownlint-cli ]);

          NIX_PYTHON_SITE_PACKAGES = python.sitePackages;
          shellHook = ''
            export LD_LIBRARY_PATH="${
              pkgs.lib.makeLibraryPath [ pkgs.zlib ]
            }:$LD_LIBRARY_PATH"

            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
          '';

        };
      });
}
