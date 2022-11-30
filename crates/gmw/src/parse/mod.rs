use nom::bytes::complete;
use nom::bytes::complete::take_until;
use nom::character::complete::{digit1, multispace0};
use nom::combinator::map_res;
use nom::error::{ErrorKind, FromExternalError, ParseError};
use nom::sequence::delimited;
use nom::{IResult, Parser};
use smallvec::SmallVec;
use std::num::ParseIntError;

pub mod aby;
pub mod bristol;
pub mod lut_circuit;

fn take_until_consume<'a, 't: 'a, E: ParseError<&'a str>>(
    tag: &'t str,
) -> impl Fn(&'a str) -> IResult<&'a str, &'a str, E> {
    move |i: &str| {
        let (i, _) = take_until(tag)(i)?;
        // take_until doesn't consume the pattern
        let (i, tag) = complete::tag(tag)(i)?;
        Ok((i, tag))
    }
}

fn uint<'a, E: ParseError<&'a str> + FromExternalError<&'a str, ParseIntError> + 'a>(
    i: &'a str,
) -> IResult<&'a str, usize, E> {
    map_res(digit1, |s: &str| s.parse())(i)
}

fn integer_ws<'a, E: ParseError<&'a str> + FromExternalError<&'a str, ParseIntError> + 'a>(
    i: &'a str,
) -> IResult<&'a str, usize, E> {
    ws(uint)(i)
}

/// A combinator that takes a parser `inner` and produces a parser that also consumes both leading and
/// trailing whitespace, returning the output of `inner`.
/// Source: https://docs.rs/nom/latest/nom/recipes/index.html#wrapper-combinators-that-eat-whitespace-before-and-after-a-parser
fn ws<'a, F, O, E: ParseError<&'a str>>(inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: FnMut(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(multispace0, inner, multispace0)
}

/// count parser from nom adapted to return smallvec
fn count_sm<I, O, E, F, const N: usize>(
    mut f: F,
    count: usize,
) -> impl FnMut(I) -> IResult<I, SmallVec<[O; N]>, E>
where
    I: Clone + PartialEq,
    F: Parser<I, O, E>,
    E: ParseError<I>,
{
    move |i: I| {
        let mut input = i.clone();
        let mut res = SmallVec::new();

        for _ in 0..count {
            let input_ = input.clone();
            match f.parse(input_) {
                Ok((i, o)) => {
                    res.push(o);
                    input = i;
                }
                Err(nom::Err::Error(e)) => {
                    return Err(nom::Err::Error(E::append(i, ErrorKind::Count, e)));
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        Ok((input, res))
    }
}
