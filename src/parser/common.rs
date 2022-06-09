use crate::parser::{lex::Token, spanned, Spanned, SubParser};
use chumsky::prelude::*;

pub type RecordEntry<T> = (Spanned<String>, Spanned<T>);

// Parses enclosed item
//
// Result in whatever item were inside parens, so basically works by removing
// parens.
pub(super) fn grouping<I, P: SubParser<I> + Clone>(
    expr: P,
) -> impl SubParser<I> + Clone {
    expr.delimited_by(just(Token::OpenParen), just(Token::CloseParen))
}

// Parses comma separated items delimited by parens
pub(super) fn tuple_like<S, O, F, P>(item: P, f: F) -> impl SubParser<O> + Clone
where
    F: Fn(Vec<S>) -> O + Clone,
    P: SubParser<S> + Clone,
{
    item.separated_by(just(Token::Comma))
        .delimited_by(just(Token::OpenParen), just(Token::CloseParen))
        .collect::<Vec<_>>()
        .map(f)
}

// Parses module subscription
pub(super) fn from_module<T, F, P>(
    item: P,
    f: F,
) -> impl SubParser<Spanned<T>> + Clone
where
    P: SubParser<Spanned<T>> + Clone + 'static,
    F: Fn(Spanned<String>, Spanned<T>) -> T + Clone + 'static,
    T: 'static,
{
    recursive(|submodule| {
        let def = submodule.or(item);

        select! {Token::Ident(m) => m}
            .map_with_span(spanned)
            .then_ignore(just(Token::Dot))
            .then(def)
            .map(move |(mod_name, expr)| f(mod_name, expr))
            .map_with_span(spanned)
    })
}

// Parses `a op b op .. op z` chains
//
// If none of `op` were found, just return first parsed item.
//
// `op` may be multiple operators, but remember that each precedence step
// should be separate.
pub(super) fn foldl_binops<T, OP, TP, F>(
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
