fn main() {
    if cfg!(target_feature = "avx2") {
        cc::Build::new()
            .files([
                "bitpolymul2/bc.c",
                "bitpolymul2/bitpolymul.c",
                "bitpolymul2/encode.c",
                "bitpolymul2/butterfly_net.c",
                "bitpolymul2/gfext_aesni.c",
                "bitpolymul2/gf2128_cantor_iso.c",
                "bitpolymul2/btfy.c",
                "bitpolymul2/trunc_btfy_tab.c",
                "bitpolymul2/gf264_cantor_iso.c",
                "bitpolymul2/trunc_btfy_tab_64.c",
            ])
            .flag("-mavx2")
            .flag("-mpclmul")
            .flag("-funroll-loops")
            .compile("bitpolymul")
    } else {
        panic!("target_feature = \"avx2\" must be enabled to compile bitpolymul-sys")
    }
}
