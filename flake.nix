{
  description = "haystackfm — GPU-accelerated FM-index construction for DNA sequences via WebGPU";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
      }));

      # Cargo.toml sets rust-version = "1.87"; we build on current stable and
      # carry the wasm32 target so the same toolchain serves the devShell.
      rustToolchainFor = pkgs: pkgs.rust-bin.stable.latest.default.override {
        extensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];
        targets = [ "wasm32-unknown-unknown" ];
      };

      # Only the files cargo actually reads. Keeps docs/, web/, and .github/
      # edits from invalidating the build.
      srcFor = pkgs: pkgs.lib.fileset.toSource {
        root = ./.;
        fileset = pkgs.lib.fileset.unions [
          ./Cargo.toml
          ./Cargo.lock
          ./.cargo
          ./src
          ./shaders
          ./benches
          ./examples
          ./tests
          ./README.md
          ./LICENSE
        ];
      };
    in
    {
      packages = forAllSystems (pkgs:
        let
          rustToolchain = rustToolchainFor pkgs;
          rustPlatform = pkgs.makeRustPlatform {
            cargo = rustToolchain;
            rustc = rustToolchain;
          };
        in
        rec {
          haystackfm = rustPlatform.buildRustPackage {
            pname = "haystackfm";
            version = "0.5.0";

            src = srcFor pkgs;
            cargoLock.lockFile = ./Cargo.lock;

            # Default features = ["cpu"]. The gpu/wasm features are devShell
            # concerns: gpu needs a physical device to test, and wasm needs a
            # wasm-bindgen CLI version-matched to the crate.
            buildNoDefaultFeatures = false;

            # Mirrors CI: CPU-only test suite. GPU tests are cfg-gated out and
            # criterion benches are not run by cargo test.
            doCheck = true;

            meta = with pkgs.lib; {
              description = "GPU-accelerated FM-index construction for DNA sequences via WebGPU";
              homepage = "https://github.com/sriram98v/haystackfm";
              license = licenses.asl20;
              platforms = platforms.unix;
            };
          };

          default = haystackfm;
        });

      devShells = forAllSystems (pkgs:
        let
          rustToolchain = rustToolchainFor pkgs;

          # The wasm-bindgen CLI must match the wasm-bindgen crate version
          # exactly — a mismatch fails with an opaque schema error. Cargo.lock
          # is the source of truth, so pin the CLI to it rather than taking
          # whatever nixpkgs happens to ship (0.2.121 at time of writing).
          #
          # Note js-sys pins `wasm-bindgen = "=0.2.117"`, so the crate version
          # can only move by bumping the js-sys/web-sys family together.
          #
          # To bump: change wasmBindgenVersion, set both hashes to
          # pkgs.lib.fakeHash, run `nix develop`, and paste the "got:" values.
          # The shellHook check below catches it if you forget.
          wasmBindgenVersion = "0.2.117";
          wasmBindgenSrc = pkgs.fetchCrate {
            pname = "wasm-bindgen-cli";
            version = wasmBindgenVersion;
            hash = "sha256-vtDQXL8FSgdutqXG7/rBUWgrYCtzdmeVQQkWkjasvZU=";
          };
          wasmBindgenCli = pkgs.buildWasmBindgenCli {
            src = wasmBindgenSrc;
            cargoDeps = pkgs.rustPlatform.fetchCargoVendor {
              src = wasmBindgenSrc;
              hash = "sha256-eKe7uwneUYxejSbG/1hKqg6bSmtL0KQ9ojlazeqTi88=";
            };
          };

          # wgpu loads Vulkan at runtime. On NixOS these come from the store; on
          # other distros the system ICD manifests in /usr/share/vulkan/icd.d are
          # still found, and lavapipe is here as a software fallback.
          gpuLibs = pkgs.lib.optionals pkgs.stdenv.isLinux [
            pkgs.vulkan-loader
            pkgs.mesa
          ];
        in
        {
          default = pkgs.mkShell {
            packages = [
              rustToolchain
              pkgs.cargo-semver-checks
            ]
            ++ [
              # WASM pipeline, mirroring deploy.yml
              wasmBindgenCli
              pkgs.wasm-pack
              pkgs.binaryen # wasm-opt, used by the release profile
              # Note: deploy.yml pins Node 20, which is EOL and no longer in
              # nixpkgs. Vite 5 builds fine on 22.
              pkgs.nodejs_22
            ]
            ++ [
              # docs/ mdBook site
              pkgs.mdbook
            ]
            ++ [ pkgs.pkg-config ]
            ++ gpuLibs
            ++ pkgs.lib.optionals pkgs.stdenv.isLinux [ pkgs.vulkan-tools ];

            env = {
              # rust-analyzer needs an explicit sysroot source path.
              RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
            };

            shellHook = ''
              ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
                export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath gpuLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

                # Vendor Vulkan drivers (NVIDIA, AMDVLK) install their ICD JSON
                # under /usr/share/vulkan/icd.d but their .so under /usr/lib,
                # which the loader can't see from a nix shell. Append — never
                # prepend — so nixpkgs libraries still win every lookup.
                # NixOS is exempt: drivers come from /run/opengl-driver there.
                if [ ! -e /etc/NIXOS ] && [ -d /usr/lib ]; then
                  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib"
                fi

                # Software Vulkan, for running GPU code paths without a device
                # (headless CI, or reproducing a driver-specific result):
                #   VK_DRIVER_FILES=$LAVAPIPE_ICD cargo test --features gpu
                for _icd in ${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.*.json; do
                  [ -e "$_icd" ] && export LAVAPIPE_ICD="$_icd"
                done
                unset _icd
              ''}

              # wasm-bindgen refuses to run when the CLI and the crate disagree
              # on schema version, and the error is opaque. Warn up front.
              lock_wb="$(sed -n '/^name = "wasm-bindgen"$/{n;s/^version = "\(.*\)"$/\1/p;}' Cargo.lock 2>/dev/null | head -1)"
              cli_wb="$(wasm-bindgen --version 2>/dev/null | awk '{print $2}')"
              if [ -n "$lock_wb" ] && [ -n "$cli_wb" ] && [ "$lock_wb" != "$cli_wb" ]; then
                echo "warning: wasm-bindgen CLI $cli_wb != crate $lock_wb (from Cargo.lock)." >&2
                echo "         wasm builds will fail with a schema mismatch until these agree." >&2
              fi

              echo "haystackfm dev shell — cargo $(cargo --version | awk '{print $2}')"
            '';
          };
        });

      checks = forAllSystems (pkgs: {
        inherit (self.packages.${pkgs.stdenv.hostPlatform.system}) haystackfm;
      });

      formatter = forAllSystems (pkgs: pkgs.nixpkgs-fmt);
    };
}
