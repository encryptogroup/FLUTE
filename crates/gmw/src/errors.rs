use remoc::rch::base;
use std::io;

use crate::parse::lut_circuit::LutParseError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExecutorError {}

#[derive(Error, Debug)]
pub enum CircuitError {
    #[error("Unable to save circuit as dot file")]
    SaveAsDot(#[source] io::Error),
    #[error("Unable to load bristol file")]
    LoadBristol(#[from] BristolError),
    #[error("Unable to load aby file")]
    LoadAby(#[from] AbyError),
    #[error("Unable to load lut circuit file")]
    LoadLutCircuit(#[from] LutCircuitError),
    #[error("Unable to convert circuit description")]
    ConversionError,
}

#[derive(Debug, Error)]
pub enum BristolError {
    #[error("Unable to read bristol file")]
    ReadFailed(#[from] io::Error),
    #[error("Unable to parse bristol file")]
    ParseFailed(#[from] nom::Err<nom::error::Error<String>>),
}

#[derive(Debug, Error)]
pub enum AbyError {
    #[error("Unable to read aby file")]
    ReadFailed(#[from] io::Error),
    #[error("Unable to parse aby file")]
    ParseFailed(#[from] nom::Err<nom::error::Error<String>>),
}

#[derive(Debug, Error)]
pub enum LutCircuitError {
    #[error("Unable to read LUT file")]
    ReadFailed(#[from] io::Error),
    #[error("Unable to parse LUT file")]
    ParseFailed(#[from] nom::Err<LutParseError<String>>),
}

#[derive(Debug, Error)]
pub enum MTProviderError<Request> {
    #[error("Sending MT request failed")]
    RequestFailed(#[from] base::SendError<Request>),
    #[error("Receiving MTs failed")]
    ReceiveFailed(#[from] base::RecvError),
    #[error("Remote unexpectedly closed")]
    RemoteClosed,
    #[error("Received illegal message from provided")]
    IllegalMessage,
}

impl LutCircuitError {
    pub fn truncate_input(&mut self, len: usize) {
        if let Self::ParseFailed(parse_err) = self {
            match parse_err {
                nom::Err::Error(err) | nom::Err::Failure(err) => err.truncate_input(len),
                nom::Err::Incomplete(_) => {}
            }
        }
    }
}
