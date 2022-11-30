use crate::errors::AbyError;
use crate::parse::{integer_ws, take_until_consume, ws};
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete;
use nom::character::complete::multispace0;
use nom::combinator::{all_consuming, map};
use nom::multi::{many0, many1};
use nom::IResult;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Circuit {
    pub header: Header,
    pub gates: Vec<Gate>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Header {
    pub input_wires_server: Vec<i64>,
    pub input_wires_client: Vec<i64>,
    pub constant_wires: Vec<(bool, i64)>,
    pub output_wires: Vec<i64>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Gate {
    And(GateData),
    Xor(GateData),
    Mux(GateData),
}

impl Circuit {
    pub fn load(path: &Path) -> Result<Circuit, AbyError> {
        let file_content = fs::read_to_string(path)?;
        circuit(&file_content).map_err(|err| err.to_owned().into())
    }
}

impl Gate {
    pub fn get_data(&self) -> &GateData {
        let (Gate::And(data) | Gate::Xor(data) | Gate::Mux(data)) = self;
        data
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct GateData {
    pub input_wires: Vec<i64>,
    pub output_wire: i64,
}

pub fn circuit(input: &str) -> Result<Circuit, nom::Err<nom::error::Error<&str>>> {
    let (i, mut header) = partial_header(input)?;
    let (i, _) = take_until_consume("#Gates")(i)?;
    let (i, gates) = many1(ws(gate))(i)?;
    let (i, _) = take_until_consume("O ")(i)?;
    let (i, output_wires) = many1(ws(complete::i64))(i)?;
    header.output_wires = output_wires;
    let _ = all_consuming(multispace0)(i)?;
    Ok(Circuit { header, gates })
}

fn partial_header(i: &str) -> IResult<&str, Header> {
    let (i, _) = take_until_consume("S ")(i)?;
    let (i, input_wires_server) = many0(ws(complete::i64))(i)?;
    let (i, _) = take_until_consume("C ")(i)?;
    let (i, input_wires_client) = many0(ws(complete::i64))(i)?;
    let (i, constant_wires) = many0(parse_constant)(i)?;
    let header = Header {
        input_wires_server,
        input_wires_client,
        constant_wires,
        output_wires: vec![],
    };
    Ok((i, header))
}

fn parse_constant(i: &str) -> IResult<&str, (bool, i64)> {
    let (i, _) = take_until_consume("#constant ")(i)?;
    let (i, _) = alt((tag("one"), tag("zero")))(i)?;
    let (i, constant) = map(integer_ws, |constant| constant != 0)(i)?;
    let (i, wire) = ws(complete::i64)(i)?;
    Ok((i, (constant, wire)))
}

fn gate(i: &str) -> IResult<&str, Gate> {
    let (i, gate_t) = alt((tag("A"), tag("X"), tag("M")))(i)?;

    let (i, mut wires) = many1(ws(complete::i64))(i)?;
    let gate_data = GateData {
        output_wire: wires.pop().expect("many1 parses at least 1"),
        input_wires: wires,
    };

    let gate = match gate_t {
        "A" => Gate::And(gate_data),
        "X" => Gate::Xor(gate_data),
        "M" => Gate::Mux(gate_data),
        _ => unreachable!(),
    };

    Ok((i, gate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_test_circuit() -> anyhow::Result<()> {
        let _circuit = Circuit::load(Path::new("test_resources/aby-circuits/fp_ieee_add_32.aby"))?;
        Ok(())
    }
}
