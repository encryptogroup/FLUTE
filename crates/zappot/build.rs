fn main() {
    // Only compile the transpose implementation in C when the c_sse feature is enabled
    if cfg!(feature = "c_sse") {
        cc::Build::new()
            .file("c_transpose/sse_transpose.c")
            .flag("-maes")
            .flag("-msse4.1")
            .compile("libtranspose.a");
    }
}
