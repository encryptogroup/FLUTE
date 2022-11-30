use std::fs;
use std::path::Path;

use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::multispace0;
use nom::combinator::all_consuming;
use nom::multi::{count, fill};
use nom::sequence::tuple;
use nom::IResult;
use smallvec::SmallVec;

use crate::errors::BristolError;
use crate::parse;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Circuit {
    pub header: Header,
    pub gates: Vec<Gate>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Header {
    pub gates: usize,
    pub wires: usize,
    /// number n1 and n2 of wires in the inputs to the function given by the circuit
    pub input_wires: [usize; 2],
    /// n3, number of wires in the output
    pub output_wires: usize,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Gate {
    And(GateData),
    Xor(GateData),
    Inv(GateData),
}

impl Circuit {
    pub fn load(path: impl AsRef<Path>) -> Result<Circuit, BristolError> {
        let bristol_text = fs::read_to_string(path)?;
        circuit(&bristol_text).map_err(|err| err.to_owned().into())
    }
}

impl Gate {
    pub fn get_data(&self) -> &GateData {
        let (Gate::And(data) | Gate::Xor(data) | Gate::Inv(data)) = self;
        data
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct GateData {
    pub input_wires: SmallVec<[usize; 2]>,
    pub output_wires: SmallVec<[usize; 1]>,
}

fn header(i: &str) -> IResult<&str, Header> {
    // parse the first line of the header
    let int_ws = parse::integer_ws;
    let (i, (gates, wires)) = tuple((int_ws, int_ws))(i)?;

    let (i, (input_wires, output_wires)) = tuple((array(int_ws), int_ws))(i)?;
    let header = Header {
        gates,
        wires,
        input_wires,
        output_wires,
    };
    Ok((i, header))
}

fn gate(i: &str) -> IResult<&str, Gate> {
    let (i, (num_in_wires, num_out_wires)) =
        tuple((parse::ws(parse::uint), parse::ws(parse::uint)))(i)?;
    let mut input_wires = SmallVec::from_elem(0, num_in_wires);
    let mut output_wires = SmallVec::from_elem(0, num_out_wires);
    let (i, _) = fill(parse::integer_ws, &mut input_wires)(i)?;
    let (i, _) = fill(parse::integer_ws, &mut output_wires)(i)?;
    let gate_data = GateData {
        input_wires,
        output_wires,
    };
    let (i, gate) = match alt((tag("AND"), tag("XOR"), tag("INV")))(i)? {
        (i, "AND") => (i, Gate::And(gate_data)),
        (i, "XOR") => (i, Gate::Xor(gate_data)),
        (i, "INV") => (i, Gate::Inv(gate_data)),
        _ => unreachable!("Bug: Parsed unknown gate"),
    };
    Ok((i, gate))
}

pub fn circuit(input: &str) -> Result<Circuit, nom::Err<nom::error::Error<&str>>> {
    let (i, header) = header(input)?;
    let (i, gates) = count(gate, header.gates)(i)?;
    let _ = all_consuming(multispace0)(i)?;
    Ok(Circuit { header, gates })
}

fn array<'a, F: 'a, O: Default + Copy, const N: usize>(
    element: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, [O; N]>
where
    F: Fn(&'a str) -> IResult<&'a str, O>,
{
    move |i: &str| {
        let mut buf = [O::default(); N];
        let (i, ()) = fill(&element, &mut buf[..])(i)?;
        Ok((i, buf))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::bristol::{circuit, gate, header, Gate, GateData, Header};

    #[test]
    fn parse_header() {
        let header_text = "33616 33872\n128 128   128";
        let parsed = header(header_text).unwrap().1;
        assert_eq!(
            Header {
                gates: 33616,
                wires: 33872,
                input_wires: [128, 128],
                output_wires: 128
            },
            parsed
        );
    }

    #[test]
    fn parse_xor_gate() {
        let gate_text = "2 1 215 87 32601 XOR";
        let parsed = gate(gate_text).unwrap().1;
        assert_eq!(
            Gate::Xor(GateData {
                input_wires: vec![215, 87].into(),
                output_wires: vec![32601].into()
            }),
            parsed
        );
    }

    #[test]
    fn parse_inv_gate() {
        let gate_text = "1 3 215 87 32601 42 INV";
        let parsed = gate(gate_text).unwrap().1;
        assert_eq!(
            Gate::Inv(GateData {
                input_wires: vec![215].into(),
                output_wires: vec![87, 32601, 42].into()
            }),
            parsed
        );
    }

    #[test]
    fn parse_and_gate() {
        let gate_text = "2 1 215 87 32601 AND";
        let parsed = gate(gate_text).unwrap().1;
        assert_eq!(
            Gate::And(GateData {
                input_wires: vec![215, 87].into(),
                output_wires: vec![32601].into()
            }),
            parsed
        );
    }

    #[test]
    fn parse_aes_circuit() {
        let aes_text =
            fs::read_to_string("test_resources/bristol-circuits/AES-non-expanded.txt").unwrap();
        let parsed = circuit(&aes_text).unwrap();
        assert_eq!(33616, parsed.header.gates);
        assert_eq!(33872, parsed.header.wires);
        assert_eq!(parsed.header.gates, parsed.gates.len());
    }
}
