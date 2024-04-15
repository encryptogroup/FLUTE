use crate::errors::LutCircuitError;
use crate::parse::{count_sm, integer_ws, take_until_consume, ws};
use bitvec::field::BitField;
use bitvec::order::{Lsb0, Msb0};
use indexmap::IndexSet;
use nom::branch::alt;
use nom::bytes::complete::{tag, take_till1};
use nom::character::complete::{hex_digit1, one_of};
use nom::combinator::{all_consuming, map, map_res};
use nom::error::ErrorKind;
use nom::multi::{length_count, many1};
use nom::sequence::{preceded, tuple};
use nom::{error, IResult};
use num_bigint::BigUint;
use num_traits::Num;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use std::ops::{BitXor, Not as NotOp};
use std::path::Path;
use std::{fs, iter};
use tracing::{error, trace};

type BitSlice<O> = bitvec::slice::BitSlice<u8, O>;
type BitVec<O> = bitvec::vec::BitVec<u8, O>;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Circuit {
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
    pub gates: Vec<Gate>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Gate {
    Lut(Lut),
    Xor(Xor),
    Xnor(Xnor),
    Not(Not),
    Assign(Assign),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Lut {
    pub input_wires: SmallVec<[Wire; 4]>,
    pub masked_luts: SmallVec<[MaskedLut; 2]>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Xor {
    pub input: [Wire; 2],
    pub output: Wire,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Xnor {
    pub input: [Wire; 2],
    pub output: Wire,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Not {
    pub input: Wire,
    pub output: Wire,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Assign {
    Constant { constant: bool, output: Wire },
    Wire { input: Wire, output: Wire },
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MaskedLut {
    pub wire_mask: WireMask,
    pub output: WireOutput,
    pub out_wire: Wire,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Wire {
    Input(Input),
    Output(Output),
    Internal(Internal),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Input(pub String);
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Output(pub String);
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Internal(pub String);

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WireMask {
    pub mask: BitVec<Msb0>,
}
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WireOutput {
    pub unexpanded: BitVec<Msb0>,
}

#[derive(Debug, PartialEq)]
pub enum LutParseError<I> {
    WireMask {
        inp: I,
        expected_set: usize,
        actual_set: usize,
    },
    Nom(error::Error<I>),
}

impl Circuit {
    #[tracing::instrument]
    pub fn load(path: &Path) -> Result<Self, LutCircuitError> {
        let file_content = fs::read_to_string(path)?;
        circuit(&file_content).map_err(|err| err.map(|inner| inner.to_owned()).into())
    }

    pub fn execute(&self, input: &BitSlice<Lsb0>) -> BitVec<Lsb0> {
        assert_eq!(
            input.len(),
            self.inputs.len(),
            "Input has wrong len. Expected: {}, Actual: {}",
            self.inputs.len(),
            input.len()
        );
        let mut wire_vals: HashMap<Wire, bool> = iter::zip(&self.inputs, input)
            .map(|(in_wire, inp)| (Wire::Input(in_wire.clone()), (*inp)))
            .collect();
        for gate in &self.gates {
            gate.execute(&mut wire_vals);
        }
        self.outputs
            .iter()
            .map(|out| wire_vals[&Wire::Output(out.clone())])
            .collect()
    }
}

impl MaskedLut {
    pub fn expanded(&self) -> BitVec<Lsb0> {
        expand(&self.output.unexpanded, self.wire_mask.mask())
    }
}

impl WireMask {
    pub fn mask(&self) -> &BitSlice<Msb0> {
        &self.mask
    }

    pub fn wires_set(&self) -> usize {
        self.mask.count_ones()
    }
}

fn circuit(i: &str) -> Result<Circuit, nom::Err<LutParseError<&str>>> {
    let mut input_wires = HashSet::new();
    let mut output_wires = HashSet::new();
    let (i, inputs) = header_inputs(&mut input_wires)(i)?;
    let (i, outputs) = header_outputs(&mut output_wires)(i)?;
    let (i, _) = take_until_consume("#LUTs")(i)?;
    let (_, gates) = all_consuming(many1(ws(gate(&input_wires, &output_wires))))(i)?;
    Ok(Circuit {
        inputs,
        outputs,
        gates,
    })
}

fn header_inputs(
    input_wires: &mut HashSet<Input>,
) -> impl FnMut(&str) -> IResult<&str, Vec<Input>, LutParseError<&str>> + '_ {
    move |i: &str| {
        let (i, _) = take_until_consume("#INPUTS")(i)?;
        let (i, inputs) = length_count(integer_ws, ws(new_input(input_wires)))(i)?;
        Ok((i, inputs))
    }
}

fn header_outputs(
    output_wires: &mut HashSet<Output>,
) -> impl FnMut(&str) -> IResult<&str, Vec<Output>, LutParseError<&str>> + '_ {
    move |i: &str| {
        let (i, _) = take_until_consume("#OUTPUTS")(i)?;
        let (i, outputs) = length_count(integer_ws, ws(new_output(output_wires)))(i)?;
        Ok((i, outputs))
    }
}

fn gate<'c>(
    input_wires: &'c HashSet<Input>,
    output_wires: &'c HashSet<Output>,
) -> impl Fn(&str) -> IResult<&str, Gate, LutParseError<&str>> + 'c {
    move |i: &str| {
        alt((
            map(lut(input_wires, output_wires), Gate::Lut),
            map(xor(input_wires, output_wires), Gate::Xor),
            map(xnor(input_wires, output_wires), Gate::Xnor),
            map(not(input_wires, output_wires), Gate::Not),
            map(assign(input_wires, output_wires), Gate::Assign),
        ))(i)
    }
}

fn lut<'c>(
    input_wires: &'c HashSet<Input>,
    output_wires: &'c HashSet<Output>,
) -> impl Fn(&str) -> IResult<&str, Lut, LutParseError<&str>> + 'c {
    move |i: &str| {
        let (i, (in_count, out_count)) = preceded(tag("LUT"), tuple((integer_ws, integer_ws)))(i)?;
        let (i, lut_input_wires): (&str, SmallVec<[_; 8]>) =
            count_sm(ws(wire(input_wires, output_wires)), in_count)(i)?;
        // hacky shit to fix the duplicate input wires...
        let mut is_duplicate: SmallVec<[bool; 8]> = SmallVec::new();
        let mut input_wires_set: IndexSet<_> = IndexSet::new();
        for inp_wire in lut_input_wires {
            let is_dupl = !input_wires_set.insert(inp_wire);
            is_duplicate.push(is_dupl);
        }
        let lut_input_wires = SmallVec::from_iter(input_wires_set);
        let (i, masked_luts) = count_sm(
            ws(masked_lut(input_wires, output_wires, &is_duplicate)),
            out_count,
        )(i)?;
        Ok((
            i,
            Lut {
                input_wires: lut_input_wires,
                masked_luts,
            },
        ))
    }
}

fn xor<'c>(
    input_wires: &'c HashSet<Input>,
    output_wires: &'c HashSet<Output>,
) -> impl Fn(&str) -> IResult<&str, Xor, LutParseError<&str>> + 'c {
    move |i: &str| {
        let wire = wire(input_wires, output_wires);
        let (i, (in_a, in_b, out)) = preceded(tag("X"), tuple((ws(wire), ws(wire), ws(wire))))(i)?;
        Ok((
            i,
            Xor {
                input: [in_a, in_b],
                output: out,
            },
        ))
    }
}

fn xnor<'c>(
    input_wires: &'c HashSet<Input>,
    output_wires: &'c HashSet<Output>,
) -> impl Fn(&str) -> IResult<&str, Xnor, LutParseError<&str>> + 'c {
    move |i: &str| {
        let wire = wire(input_wires, output_wires);
        let (i, (in_a, in_b, out)) = preceded(tag("XN"), tuple((ws(wire), ws(wire), ws(wire))))(i)?;
        Ok((
            i,
            Xnor {
                input: [in_a, in_b],
                output: out,
            },
        ))
    }
}

fn not<'c>(
    input_wires: &'c HashSet<Input>,
    output_wires: &'c HashSet<Output>,
) -> impl Fn(&str) -> IResult<&str, Not, LutParseError<&str>> + 'c {
    move |i: &str| {
        let wire = wire(input_wires, output_wires);
        let (i, (input, output)) = preceded(tag("N"), tuple((ws(wire), ws(wire))))(i)?;
        Ok((i, Not { input, output }))
    }
}

fn assign<'c>(
    input_wires: &'c HashSet<Input>,
    output_wires: &'c HashSet<Output>,
) -> impl Fn(&str) -> IResult<&str, Assign, LutParseError<&str>> + 'c {
    move |i: &str| {
        let wire = wire(input_wires, output_wires);
        let (i, _) = ws(tag("A"))(i)?;
        match one_of::<_, _, LutParseError<&str>>("01")(i) {
            Ok((i, constant)) => {
                let constant = constant == '1';
                let (i, output) = ws(wire)(i)?;
                Ok((i, Assign::Constant { constant, output }))
            }
            Err(nom::Err::Error(_err)) => {
                let (i, (input, output)) = tuple((ws(wire), ws(wire)))(i)?;
                Ok((i, Assign::Wire { input, output }))
            }
            Err(err) => return Err(err),
        }
    }
}

fn masked_lut<'d, 'c: 'd>(
    input_wires: &'c HashSet<Input>,
    output_wires: &'c HashSet<Output>,
    is_duplicate: &'d [bool],
) -> impl Fn(&str) -> IResult<&str, MaskedLut, LutParseError<&str>> + 'd {
    move |i: &str| {
        let (i, mask) = wire_mask(is_duplicate)(i)?;
        let (i, output) = ws(wire_output(mask.mask()))(i)?;
        let (i, out_wire) = wire(input_wires, output_wires)(i)?;
        Ok((
            i,
            MaskedLut {
                wire_mask: mask,
                output,
                out_wire,
            },
        ))
    }
}

fn wire_mask(
    is_duplicate: &[bool],
) -> impl Fn(&str) -> IResult<&str, WireMask, LutParseError<&str>> + '_ {
    move |i: &str| {
        let err_inp = i;
        let (i, wires_used) = integer_ws(i)?;
        let (i, bits): (&str, SmallVec<[char; 4]>) = count_sm(one_of("01"), is_duplicate.len())(i)?;
        let mut mask = BitVec::new();
        for (char_bit, is_dupl) in bits.into_iter().zip(is_duplicate) {
            // skip the bits which are for duplicate wires
            if !*is_dupl {
                mask.push(char_bit == '1')
            }
        }
        // consume superfluous bits in wire mask
        // let (i, cnt) = fold_many0(one_of("01"), || 0_usize, |cnt, _| cnt + 1)(i)?;
        if wires_used != mask.count_ones() {
            error!(
                expected_set = wires_used,
                actual_sec = mask.count_ones(),
                input = get_prefix(err_inp, i),
                "Expected set wires does not match actual set."
            );
            // return Err(nom::Err::Failure(LutParseError::WireMask {
            //     inp: get_prefix(err_inp, i),
            //     expected_set: wires_used,
            //     actual_set: mask.count_ones(),
            // }));
        }
        Ok((i, WireMask { mask }))
    }
}

fn wire_output<'w, 'i>(
    wire_mask: &'w BitSlice<Msb0>,
) -> impl Fn(&'i str) -> IResult<&'i str, WireOutput, LutParseError<&'i str>> + 'w {
    move |i: &str| {
        // drop the 0x prefix and parse the hexadecimal number into a BigUint
        let (i, out_mask) = map_res(preceded(tag("0x"), hex_digit1), |hex_val| {
            BigUint::from_str_radix(hex_val, 16)
        })(i)?;
        // the big integer is split into its bytes in big endian order
        // this means, the most significant bytes come first
        let be_bytes = out_mask.to_bytes_be();
        // these big endian bytes are put into a bit vector with most significant **bit** first
        // ordering (Msb0)
        let mut out_bit_mask = BitVec::<Msb0>::from_slice(&be_bytes);
        // because the relevant bits of the parse number can be smaller than a single byte (for an
        // LUT with two active output wires, output bit mask should have 4 bits), or the bit we need
        // can be larger than initial hex number as a bitvec, we calculate the required shift to
        // the right or left
        let used_bits = 2_usize.pow(wire_mask.count_ones() as u32);
        let shift = used_bits as i32 - out_bit_mask.len() as i32;
        if shift < 0 {
            // out_bit_mask.len() > used_bits
            // this happens if we have two active output wires (wire_mask.count_ones() == 2) but
            // the minimum size of the out_bit_mask is 8, as we always get at least one byte
            // from out_mask.to_bytes_be()
            // therefore, we shift the out_bit_mask by the absolute shift amount to the left
            // (thus removing superflous 0 bits) and truncate the bitvec to the correct size
            out_bit_mask.shift_left(shift.abs() as usize);
            out_bit_mask.truncate(used_bits);
        } else if shift > 0 {
            // out_bit_mask.len() < used_bits
            // this happens if we require a large output_bit_mask, e.g. for a LUT with 8 outputs
            // which would require a size 256 output, but the initial hex number was something
            // small like 0x8, which only results in a single byte. Thus, we first resize the
            // bitmask to the needed length and then shift_right the original out_bit_mask so the
            // used bits are at the end of the bitvec
            out_bit_mask.resize(used_bits, false);
            out_bit_mask.shift_right(shift as usize);
        }
        Ok((
            i,
            WireOutput {
                unexpanded: out_bit_mask,
            },
        ))
    }
}

fn wire<'c>(
    input_wires: &'c HashSet<Input>,
    output_wires: &'c HashSet<Output>,
) -> impl Fn(&str) -> IResult<&str, Wire, LutParseError<&str>> + Copy + 'c {
    move |i: &str| {
        let in_wire = map(input(input_wires), Wire::Input);
        let out_wire = map(output(output_wires), Wire::Output);
        let internal_wire = map(internal, Wire::Internal);
        // The order is important as sometime the internal wire prefix is also used for
        // the input/output wires
        alt((in_wire, out_wire, internal_wire))(i)
    }
}

fn new_input(
    input_wires: &mut HashSet<Input>,
) -> impl FnMut(&str) -> IResult<&str, Input, LutParseError<&str>> + '_ {
    move |i: &str| {
        let (i, wire_val) = take_till1(|c: char| c.is_whitespace())(i)?;
        let inp = Input(wire_val.to_string());
        input_wires.insert(inp.clone());
        Ok((i, inp))
    }
}

fn input(
    input_wires: &HashSet<Input>,
) -> impl Fn(&str) -> IResult<&str, Input, LutParseError<&str>> + '_ {
    move |i: &str| {
        let (i, wire_val) = take_till1(|c: char| c.is_whitespace())(i)?;
        let inp = Input(wire_val.to_string());
        if input_wires.contains(&inp) {
            Ok((i, inp))
        } else {
            Err(nom::Err::Error(LutParseError::Nom(error::Error::new(
                i,
                ErrorKind::Alt,
            ))))
        }
    }
}

fn new_output(
    output_wires: &mut HashSet<Output>,
) -> impl FnMut(&str) -> IResult<&str, Output, LutParseError<&str>> + '_ {
    |i: &str| {
        let (i, wire_val) = take_till1(|c: char| c.is_whitespace())(i)?;
        let out = Output(wire_val.to_string());
        output_wires.insert(out.clone());
        Ok((i, out))
    }
}

fn output(
    output_wires: &HashSet<Output>,
) -> impl Fn(&str) -> IResult<&str, Output, LutParseError<&str>> + '_ {
    |i: &str| {
        let (i, wire_val) = take_till1(|c: char| c.is_whitespace())(i)?;
        let out = Output(wire_val.to_string());
        if output_wires.contains(&out) {
            Ok((i, out))
        } else {
            Err(nom::Err::Error(LutParseError::Nom(error::Error::new(
                i,
                ErrorKind::Alt,
            ))))
        }
    }
}

fn internal(i: &str) -> IResult<&str, Internal, LutParseError<&str>> {
    map(take_till1(|c: char| c.is_whitespace()), |s: &str| {
        Internal(s.to_string())
    })(i)
}

impl<I> error::ParseError<I> for LutParseError<I> {
    fn from_error_kind(input: I, kind: error::ErrorKind) -> Self {
        LutParseError::Nom(error::Error::new(input, kind))
    }

    fn append(_: I, _: error::ErrorKind, other: Self) -> Self {
        other
    }
}

impl<I, E> error::FromExternalError<I, E> for LutParseError<I> {
    fn from_external_error(input: I, kind: ErrorKind, _e: E) -> Self {
        Self::Nom(error::Error::new(input, kind))
    }
}

impl LutParseError<&str> {
    fn to_owned(&self) -> LutParseError<String> {
        match self {
            LutParseError::WireMask {
                inp,
                expected_set,
                actual_set,
            } => LutParseError::WireMask {
                inp: (*inp).to_owned(),
                expected_set: *expected_set,
                actual_set: *actual_set,
            },
            LutParseError::Nom(err) => {
                LutParseError::Nom(error::Error::new(err.input.to_string(), err.code))
            }
        }
    }
}

impl LutParseError<String> {
    pub fn truncate_input(&mut self, len: usize) {
        match self {
            LutParseError::WireMask { inp, .. } => {
                inp.truncate(len);
            }
            LutParseError::Nom(err) => {
                err.input.truncate(len);
            }
        }
    }
}

fn expand(unexpanded: &BitSlice<Msb0>, wire_mask: &BitSlice<Msb0>) -> BitVec<Lsb0> {
    let used_bits = 2_usize.pow(wire_mask.len() as u32);
    let mut expanded = BitVec::with_capacity(used_bits);
    expanded.extend_from_bitslice(unexpanded);
    let mut new_expanded = BitVec::repeat(false, 2_usize.pow(wire_mask.len() as u32));
    for (idx, bit) in wire_mask.iter().rev().enumerate() {
        if *bit {
            continue;
        }
        for (idx, chunk) in expanded
            .chunks_exact(2_usize.pow(idx as u32))
            .enumerate()
            .take(2_usize.pow((wire_mask.len() - idx - 1) as u32))
        {
            let idx = 2 * idx;
            new_expanded[(idx) * chunk.len()..(idx + 1) * chunk.len()].copy_from_bitslice(chunk);
            new_expanded[(idx + 1) * chunk.len()..(idx + 2) * chunk.len()]
                .copy_from_bitslice(chunk);
        }
        expanded = new_expanded.clone();
    }
    expanded
}

impl Gate {
    fn execute(&self, wire_vals: &mut HashMap<Wire, bool>) {
        match self {
            Gate::Lut(Lut {
                input_wires,
                masked_luts,
            }) => {
                for masked_lut in masked_luts {
                    let inp: BitVec<Msb0> = input_wires
                        .iter()
                        .zip(masked_lut.wire_mask.mask())
                        .filter_map(
                            |(inp, sel_bit)| if *sel_bit { Some(wire_vals[inp]) } else { None },
                        )
                        .collect();
                    let out_idx: usize = inp.load_be();
                    let out = masked_lut.output.unexpanded[out_idx];
                    trace!(?inp, out_idx, lut = ?masked_lut.output.unexpanded, out, out_wire = ?masked_lut.out_wire);
                    wire_vals.insert(masked_lut.out_wire.clone(), out);
                }
            }
            Gate::Xor(Xor { input, output }) => {
                let out = input
                    .iter()
                    .map(|inp| wire_vals[inp])
                    .reduce(BitXor::bitxor)
                    .unwrap();
                assert_eq!(None, wire_vals.insert(output.clone(), out));
            }
            Gate::Xnor(Xnor { input, output }) => {
                let out = input
                    .iter()
                    .map(|inp| wire_vals[inp])
                    .reduce(BitXor::bitxor)
                    .unwrap()
                    .not();
                assert_eq!(None, wire_vals.insert(output.clone(), out));
            }
            Gate::Not(Not { input, output }) => {
                let out = !wire_vals[input];
                assert_eq!(None, wire_vals.insert(output.clone(), out));
            }
            Gate::Assign(Assign::Wire { input, output }) => {
                assert_eq!(None, wire_vals.insert(output.clone(), wire_vals[input]));
            }
            Gate::Assign(Assign::Constant { constant, output }) => {
                assert_eq!(None, wire_vals.insert(output.clone(), *constant));
            }
        }
    }
}

fn get_prefix<'a, 'b>(with_prefix: &'a str, sub_str: &'b str) -> &'a str {
    &with_prefix[..with_prefix.len() - sub_str.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::private_test_utils::init_tracing;
    use bitvec::order::Msb0;
    use bitvec::{bits, bitvec};

    fn inp(num: u32) -> Input {
        Input(num.to_string())
    }

    fn out(num: u32) -> Output {
        Output(num.to_string())
    }

    fn in_cache(wires: &[u32]) -> HashSet<Input> {
        wires.iter().copied().map(inp).collect()
    }

    fn out_cache(wires: &[u32]) -> HashSet<Output> {
        wires.iter().copied().map(out).collect()
    }

    #[test]
    fn test_expand() {
        let val = bitvec![u8, Msb0; 0,1,1,0,1,0,0,1,0,0,0,0,0,0,0,0];
        let wire_mask = bitvec![u8, Msb0; 1, 1, 0, 1];
        let expected = bits![u8, Msb0; 0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1];
        let expanded = expand(&val, &wire_mask[0..4]);
        assert_eq!(expected, &expanded);

        let val = bitvec![u8, Msb0; 1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
        let wire_mask = bitvec![u8, Msb0; 0, 1, 0, 0];
        let expected = bits![u8, Msb0; 1; 16];
        let expanded = expand(&val, &wire_mask[0..4]);
        assert_eq!(expected, &expanded);

        let val = bitvec![u8, Msb0; 0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0];
        let wire_mask = bitvec![u8, Msb0; 0, 1, 1, 0];
        let expected = bits![u8, Msb0; 0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1];
        let expanded = expand(&val, &wire_mask[0..4]);
        assert_eq!(expected, &expanded);

        let val = bitvec![u8, Msb0; 0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0];
        let wire_mask = bitvec![u8, Msb0; 1, 1, 1, 1];
        let expected = bits![u8, Msb0; 0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0];
        let expanded = expand(&val, &wire_mask[0..4]);
        assert_eq!(expected, &expanded);

        let val = bitvec![u8, Msb0; 1,0,1,0,1,0,1,0,0,0,1,1,1,0,0,1];
        let wire_mask = bitvec![u8, Msb0; 1, 1, 1, 1];
        let expected = bits![u8, Msb0; 1,0,1,0,1,0,1,0,0,0,1,1,1,0,0,1];
        let expanded = expand(&val, &wire_mask[0..4]);
        assert_eq!(expected, &expanded);
    }

    #[test]
    fn parse_wire() {
        let in_cache = in_cache(&[0][..]);
        let out_cache = out_cache(&[42][..]);
        let wire = wire(&in_cache, &out_cache);
        assert_eq!(Ok(("", Wire::Input(inp(0)))), wire("0"));
        assert_eq!(Ok(("", Wire::Output(out(42)))), wire("42"));
        assert_eq!(
            Ok(("", Wire::Internal(Internal("n42".into())))),
            wire("n42")
        );
    }

    #[test]
    fn parse_wire_output() {
        assert_eq!(
            Ok((
                "",
                WireOutput {
                    unexpanded: bitvec![u8, Msb0; 1,0,]
                }
            )),
            wire_output(bits![u8, Msb0; 1])("0x2")
        );
    }

    #[test]
    fn parse_wire_mask() {
        let mask = bitvec![u8, Msb0; 1,0,1,1];
        assert_eq!(
            Ok(("", WireMask { mask })),
            wire_mask(&[false; 4])("3 1011")
        );
    }

    #[test]
    fn parse_masked_lut() {
        let in_cache = in_cache(&[42][..]);
        let out_cache = out_cache(&[5][..]);
        let exp_masked_lut = MaskedLut {
            wire_mask: WireMask {
                mask: bitvec![u8, Msb0; 1,0,0,0],
            },
            output: WireOutput {
                unexpanded: bitvec![u8, Msb0; 1,1],
            },
            out_wire: Wire::Output(out(5)),
        };
        let parsed = masked_lut(&in_cache, &out_cache, &[false; 4])("1 1000 0x3 5").unwrap();

        assert_eq!(("", exp_masked_lut), parsed);
        assert_eq!(bits![u8, Msb0; 1; 16], &parsed.1.expanded())
    }

    #[test]
    fn parse_lut() {
        let exp_masked_lut = MaskedLut {
            wire_mask: WireMask {
                mask: bitvec![u8, Msb0; 1,0,0,0],
            },
            output: WireOutput {
                unexpanded: bitvec![u8, Msb0; 1,1,],
            },
            out_wire: Wire::Output(out(5)),
        };

        let expected_lut = Lut {
            input_wires: SmallVec::from([
                Wire::Input(inp(0)),
                Wire::Input(inp(1)),
                Wire::Input(inp(2)),
                Wire::Internal(Internal("n6".into())),
            ]),
            masked_luts: SmallVec::from_elem(exp_masked_lut, 2),
        };

        let in_cache = in_cache(&[0, 1, 2][..]);
        let out_cache = out_cache(&[5][..]);

        assert_eq!(
            Ok(("", expected_lut)),
            lut(&in_cache, &out_cache)("LUT 4 2 0 1 2 n6 1 1000 0x3 5 1 1000 0x3 5")
        );
    }

    #[test]
    fn parse_sample_circuit() {
        let circ =
            Circuit::load(Path::new("test_resources/lut_circuits/Sample LUT file.lut")).unwrap();
        dbg!(circ);
    }

    #[test]
    fn gate_execute_lut() {
        let wire = |num: i32| Wire::Internal(Internal(num.to_string()));
        let gate = Gate::Lut(Lut {
            input_wires: SmallVec::from_vec(vec![wire(1), wire(2), wire(3), wire(4)]),
            masked_luts: SmallVec::from(vec![MaskedLut {
                wire_mask: WireMask {
                    mask: bitvec![u8, Msb0; 1,1,0,1],
                },
                output: WireOutput {
                    unexpanded: bitvec![u8, Msb0; 0,1,1,0,1,1,0,1],
                },
                out_wire: wire(5),
            }]),
        });
        let mut wire_vals = [(wire(1), false), (wire(2), false), (wire(4), false)]
            .into_iter()
            .collect();
        gate.execute(&mut wire_vals);
        assert_eq!(false, wire_vals[&wire(4)]);
    }

    // #[test]
    // fn circuit_execute() {
    //     let _g = init_tracing();
    //     let circ = Circuit::load(Path::new("test_resources/lut_circuits/lfa32.lut")).unwrap();
    //     let inp = bitvec![u8, Lsb0; 0;64];
    //     let out = circ.execute(&inp);
    //     assert_eq!(bits![u8, Lsb0; 0;33], out);
    // }
}
