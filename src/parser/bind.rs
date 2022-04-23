use chumsky::prelude::*;

use super::{
    lex::Token,
    pattern::{atomic_pattern, complex_pattern, pattern, Pattern},
    spanned, Name, Spanned, SubParser,
};

#[derive(Debug, PartialEq, Clone)]
pub enum Bind {
    Abstraction(Name, Vec<Spanned<Pattern>>),
    Bound(Spanned<Pattern>),
}

pub(super) fn bind() -> impl SubParser<Spanned<Bind>> + Clone {
    // used for things like
    // `let x = 5`
    // `let (x, y) = (5, 10);
    let bound = pattern().map(Bind::Bound).map_with_span(spanned);

    // used for "function" declaration
    let abstraction = select! {
        Token::Ident(i) => Name::Plain(i),
        Token::LiteralSpell(s) => Name::Spell(s),
    }
    .then(
        atomic_pattern(pattern())
            .or(complex_pattern(pattern())
                .delimited_by(just(Token::OpenParen), just(Token::CloseParen)))
            .repeated()
            .at_least(1)
            .collect::<Vec<_>>(),
    )
    .map(|(name, args)| Bind::Abstraction(name, args))
    .map_with_span(spanned);

    #[cfg(debug_assertions)]
    let abstraction = abstraction.boxed();

    abstraction.or(bound)
}
