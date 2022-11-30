#[cfg(all(target_feature = "avx", target_feature = "avx2"))]
mod bindings;
#[cfg(all(target_feature = "avx", target_feature = "avx2"))]
pub use bindings::*;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
