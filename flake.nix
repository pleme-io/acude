{
  description = "açude — the retained WAL reservoir: a typed, content-addressed Postgres WAL archiver";

  # Canonical pleme-io Rust-tool consumer flake. substrate.rust.tool
  # pre-binds nixpkgs / crate2nix / flake-utils / fenix / devenv / gen.
  inputs.substrate.url = "github:pleme-io/substrate";

  outputs = { substrate, ... }: substrate.rust.tool {
    src = ./.;
    module = {
      description = "açude — the retained WAL reservoir: a typed, content-addressed Postgres WAL archiver";
    };
  };
}
