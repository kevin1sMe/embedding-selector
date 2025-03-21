{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    stdenv.cc.cc.lib
    libgcc
    zlib
    python311
    python311Packages.scipy
    python311Packages.scikit-learn
    python311Packages.numpy
    python311Packages.openai
    python311Packages.python-dotenv
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"
  '';
}
