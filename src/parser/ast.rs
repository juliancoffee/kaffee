use super::{
    lex::lexer,
    stmt::{statement, Statement},
};
use chumsky::{prelude::*, Stream};

use super::{Spanned, SubParser};

fn ast_parser() -> impl SubParser<Vec<Spanned<Statement>>> {
    statement()
        .repeated()
        .collect::<Vec<_>>()
        .then_ignore(end())
}

#[derive(Debug)]
pub struct Ast {
    pub defs: Vec<Spanned<Statement>>,
}

impl Ast {
    /// Parse all definitions from file
    ///
    /// # Panics
    /// If file contains invalid code, this function will panic
    #[must_use]
    pub fn parse_expect(src: &str) -> Self {
        let tokens = lexer().parse(src).unwrap();
        let end = src.chars().count();

        #[allow(clippy::range_plus_one)]
        let token_stream = Stream::from_iter(end..end + 1, tokens.into_iter());

        let defs = ast_parser().parse(token_stream).unwrap();

        Self { defs }
    }
}
