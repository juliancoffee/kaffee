use chumsky::prelude::*;

use super::{
    common::{grouping, tuple_like},
    spanned, Spanned, SubParser, Token,
};

#[derive(Debug, PartialEq, Clone)]
pub enum Pattern {
    Wildcard,
    Ident(String),
    Int(u64),
    Tuple(Vec<Spanned<Self>>),
    Variant(String, Box<Spanned<Self>>),
    Choice(Box<Spanned<Self>>, Box<Spanned<Self>>),
}

impl Pattern {
    #[cfg(test)]
    pub(super) fn ident(name: &str) -> Self {
        Self::Ident(name.to_owned())
    }

    pub(super) fn variant(name: String, inner: Spanned<Self>) -> Self {
        Self::Variant(name, Box::new(inner))
    }

    #[cfg(test)]
    pub(super) fn variant_with(name: &str, inner: Spanned<Self>) -> Self {
        Self::Variant(name.to_owned(), Box::new(inner))
    }

    pub(super) fn choice(a: Spanned<Self>, b: Spanned<Self>) -> Spanned<Self> {
        #[cfg(not(test))]
        let span = a.reach(&b);
        #[cfg(test)]
        let span = 0..0;

        let choice = Self::Choice(Box::new(a), Box::new(b));

        spanned(choice, span)
    }
}

// Parses variant pattern (<Name> <pat>)
fn variant<P: SubParser<Spanned<Pattern>> + Clone>(
    pat: P,
) -> impl SubParser<Spanned<Pattern>> + Clone {
    select! {Token::Ident(i) => i}
        .then(pat)
        .map(|(name, pat)| Pattern::variant(name, pat))
        .map_with_span(spanned)
}

// Parses pattern choice `<pat> | <pat>` (for any amount of choices)
//
// Returns pattern `as is` if no choices were found
fn choice_pat<P: SubParser<Spanned<Pattern>> + Clone>(
    pat: P,
) -> impl SubParser<Spanned<Pattern>> + Clone {
    pat.clone()
        .then(just(Token::Either).ignore_then(pat).repeated())
        .foldl(Pattern::choice)
}

// Artificial spilt of pattern handler
//
// Parses "atomic" patterns like ints, identifiers, wildcards,
// tuples
pub(super) fn atomic_pattern<P: SubParser<Spanned<Pattern>> + Clone>(
    pat: P,
) -> impl SubParser<Spanned<Pattern>> + Clone {
    let literal = select! {
        // FIXME: this unwrap may panic (u64 is "finite")
        Token::Int(i) => Pattern::Int(i.parse().unwrap()),
        Token::Ident(i) => Pattern::Ident(i),
        Token::Wildcard => Pattern::Wildcard,
    };
    let literal = literal.map_with_span(spanned);

    // Grouping
    let grouping = grouping(pat.clone());

    // Tuple pattern: (a, b, c)
    let tuple = tuple_like(pat, Pattern::Tuple).map_with_span(spanned);
    choice((grouping, tuple, literal))
}

// Artificial spilt of pattern handler
//
// Parses "free" patterns like variants or choice patterns
#[allow(clippy::let_and_return)]
pub(super) fn complex_pattern<
    P: SubParser<Spanned<Pattern>> + Clone + 'static,
>(
    pat: P,
) -> impl SubParser<Spanned<Pattern>> + Clone {
    // Variant pattern: <Name> <pat>
    let variant = variant(pat.clone());
    let term = variant.or(pat);
    #[cfg(debug_assertions)]
    let term = term.boxed();
    // Choice pattern: <pat> | <pat> | <pat>
    let choice_pat = choice_pat(term);
    let term = choice_pat;
    #[cfg(debug_assertions)]
    let term = term.boxed();

    term
}

// Parses pattern
//
// Split in two parts, so that abstraction syntax can be described using
// atomic and complex part in unambiguous way
pub(super) fn pattern() -> impl SubParser<Spanned<Pattern>> + Clone {
    recursive(|pat| {
        let atom = atomic_pattern(pat.clone());

        #[cfg(debug_assertions)]
        let atom = atom.boxed();

        let complex = complex_pattern(atom.clone());
        #[cfg(debug_assertions)]
        let complex = complex.boxed();

        complex
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::tests::tokens;

    fn parse_pattern(src: &str) -> Result<Pattern, Vec<Simple<Token>>> {
        let token_stream = tokens(src);

        pattern().parse(token_stream)
    }
    #[test]
    fn parse_choice_pattern() {
        let src = "0 | 1 | 5";

        assert_eq!(
            parse_pattern(src),
            Ok(Pattern::choice(
                Pattern::choice(Pattern::Int(0), Pattern::Int(1),),
                Pattern::Int(5),
            ))
        );
    }

    #[test]
    fn parse_choice_deep_pattern() {
        let src = "Empty | Str (Just _, _) | Bytes _";

        assert_eq!(
            parse_pattern(src),
            Ok(Pattern::choice(
                Pattern::choice(
                    Pattern::ident("Empty"),
                    Pattern::variant_with(
                        "Str",
                        Pattern::Tuple(vec![
                            Pattern::variant_with("Just", Pattern::Wildcard),
                            Pattern::Wildcard,
                        ])
                    )
                ),
                Pattern::variant_with("Bytes", Pattern::Wildcard)
            ))
        );
    }
}
