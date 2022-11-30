use proc_macro::TokenStream;

use syn::{parse_macro_input, ItemFn};

mod sub_circuit;

#[proc_macro_attribute]
pub fn sub_circuit(_attr: TokenStream, input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ItemFn);
    sub_circuit::sub_circuit(input).into()
}
