#![warn(clippy::pedantic)]

pub mod ast;
pub mod bind;
pub mod expr;
pub mod lex;
pub mod pattern;
pub mod stmt;
pub mod ty;

use lex::Token;

use chumsky::prelude::*;

// aliases
trait SubParser<O>: Parser<Token, O, Error = Simple<Token>> {}
impl<T, O> SubParser<O> for T where T: Parser<Token, O, Error = Simple<Token>> {}

type Span = std::ops::Range<usize>;

// We probably don't want spans in tests.
// If we will want, we may add feature like [span-test] and write tests with it
#[cfg(test)]
pub type Spanned<T> = T;

#[cfg(not(test))]
#[derive(Debug, PartialEq, Clone)]
pub struct Spanned<T> {
    span: Span,
    item: T,
}

#[cfg(not(test))]
impl<T> Spanned<T> {
    fn new(item: T, span: Span) -> Self {
        Self { span, item }
    }

    pub fn span(&self) -> &Span {
        &self.span
    }

    pub fn to_item(&self) -> &T {
        &self.item
    }

    pub fn into_item(self) -> T {
        self.item
    }

    fn reach<O>(&self, other: &Spanned<O>) -> Span {
        let start = self.span().start;
        let end = other.span().end;

        start..end
    }

    fn transfer<O>(self, other: O) -> Spanned<O> {
        spanned(other, self.span)
    }
}

#[cfg(not(test))]
fn spanned<T>(item: T, span: Span) -> Spanned<T> {
    Spanned::new(item, span)
}

#[cfg(not(test))]
fn transfer_span<T, R>(t: Spanned<T>, r: R) -> Spanned<R> {
    t.transfer(r)
}

#[cfg(not(test))]
fn lookup<T>(s: &Spanned<T>) -> &T {
    s.to_item()
}

#[cfg(not(test))]
fn unspan<T>(s: Spanned<T>) -> T {
    s.into_item()
}

#[cfg(test)]
fn spanned<T>(item: T, _: Span) -> Spanned<T> {
    item
}

#[cfg(test)]
fn transfer_span<T, R>(_: Spanned<T>, r: R) -> Spanned<R> {
    r
}

#[cfg(test)]
fn lookup<T>(s: &Spanned<T>) -> &T {
    s
}

#[cfg(test)]
fn unspan<T>(s: Spanned<T>) -> T {
    s
}

/* * * * * * * * * * * * * *
 * Useful generic parsers  *
 * * * * * * * * * * * * * */

// Parses enclosed item
//
// Result in whatever item were inside parens, so basically works by removing
// parens.
fn grouping<I, P: SubParser<I> + Clone>(expr: P) -> impl SubParser<I> + Clone {
    expr.delimited_by(just(Token::OpenParen), just(Token::CloseParen))
}

// Parses comma separated items delimited by parens
fn tuple_like<S, O, F, P>(item: P, f: F) -> impl SubParser<O> + Clone
where
    F: Fn(Vec<S>) -> O + Clone,
    P: SubParser<S> + Clone,
{
    item.separated_by(just(Token::Comma))
        .delimited_by(just(Token::OpenParen), just(Token::CloseParen))
        .collect::<Vec<_>>()
        .map(f)
}

// Parses `a op b op .. op z` chains
//
// If none of `op` were found, just return first parsed item.
//
// `op` may be multiple operators, but remember that each precedence step
// should be separate.
fn foldl_binops<T, OP, TP, F>(
    item: TP,
    op: OP,
    f: F,
) -> impl SubParser<T> + Clone
where
    TP: SubParser<T> + Clone,
    OP: SubParser<Token> + Clone,
    F: Fn(T, (Token, T)) -> T + Clone,
{
    item.clone().then(op.then(item).repeated()).foldl(f)
}

#[derive(Debug, PartialEq, Clone)]
pub enum Name {
    Plain(String),
    Spell(String),
}

impl Name {
    #[cfg(test)]
    fn plain(name: &str) -> Self {
        Name::Plain(name.to_owned())
    }

    #[cfg(test)]
    fn spell(name: &str) -> Self {
        Name::Spell(name.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::{lex::lexer, *};
    use chumsky::Stream;

    pub(super) fn tokens(
        src: &str,
    ) -> Stream<Token, Span, std::vec::IntoIter<(Token, Span)>> {
        let tokens = lexer().parse(src).unwrap();
        let end = src.chars().count();

        #[allow(clippy::range_plus_one)]
        let token_stream = Stream::from_iter(end..end + 1, tokens.into_iter());

        token_stream
    }
}
