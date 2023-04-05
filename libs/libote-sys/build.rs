use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=src");
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);

    let _dst = cmake::Config::new("libOTe")
        .define("LIBOTE_BUILD_DIR",out_dir.join("build"))
        .define("OC_THIRDPARTY_CLONE_DIR",&out_dir)
        .define("OC_THIRDPARTY_HINT",&out_dir)
        .define("OC_THIRDPARTY_INSTALL_PREFIX",&out_dir)
        .define("COPROTO_STAGE", &out_dir)
        .define("LIBDIVIDE_INCLUDE_DIRS", &out_dir.join("libdivide"))
        .define("ENABLE_BITPOLYMUL", "OFF")
        .define("FETCH_COPROTO", "ON")
        .define("FETCH_LIBDIVIDE", "ON")
        .define("ENABLE_PIC", "ON")
        .build_target("libOTe")
        .build();

    // let mut b = autocxx_build::Builder::new("src/lib.rs", &[
    //     Path::new("src"),
    //     Path::new("libOTe"),
    //     Path::new("libOTe/cryptoTools"),
    //     out_dir.join("build/cryptoTools").as_path(),
    //     out_dir.join("span-lite/include").as_path(),
    //     out_dir.join("variant-lite").as_path(),
    // ])
    //     .extra_clang_args(&["-mpclmul"])
    //     .build().unwrap();
    // b.flag("-std=c++14").flag("-mpclmul").compile("tmp");

    cxx_build::bridge("src/lib.rs")
        .file("libOTe/libOTe/Tools/LDPC/LdpcEncoder.cpp")
        .file("libOTe/cryptoTools/cryptoTools/Common/Timer.cpp")
        .includes(&[
            Path::new("src"),
            Path::new("libOTe"),
            Path::new("libOTe/cryptoTools"),
            out_dir.join("build/").as_path(),
            out_dir.join("build/cryptoTools").as_path(),
            out_dir.join("span-lite/include").as_path(),
            out_dir.join("variant-lite").as_path(),
     ]).flag("-mpclmul").flag("-std=c++14").compile("silver_bridge");
}