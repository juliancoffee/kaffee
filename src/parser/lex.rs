use chumsky::prelude::*;

use super::Span;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub(super) enum Op {
    // +
    Plus,
    // -
    Minus,
    // *
    Product,
    // /
    Divide,
    // <
    Less,
    // >
    Greater,
    // ==
    Equal,
    // !=
    NotEqual,
    // ||
    Or,
    // &&
    And,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub(super) enum Token {
    // Misc
    Doc(String),
    // Words
    Int(String),
    Str(String),
    Ident(String),
    LiteralSpell(String),
    TypeParameter(String),
    // Operators
    Op(Op),
    // Declarators
    Let,
    Type,
    Module,
    Struct,
    // Branchers
    If,
    Then,
    Else,
    Match,
    MatchArrow,
    Wildcard,
    // Binders
    Of,
    Is,
    In,
    Dot,
    // Finishers
    Semicolon,
    End,
    // Groupers
    OpenParen,
    CloseParen,
    OpenBracket,
    CloseBracket,
    OpenBrace,
    CloseBrace,
    // Separators
    Comma,
    Either,
}

impl Token {
    pub(super) fn op_expect(&self) -> &Op {
        match self {
            Self::Op(op) => op,
            token => panic!("expected Token::Op(_), got {token:?}"),
        }
    }
}

pub(super) fn lexer(
) -> impl Parser<char, Vec<(Token, Span)>, Error = Simple<char>> {
    let row_comment = just("//")
        .ignore_then(take_until(text::newline()))
        .map(|(line, ())| line.into_iter().skip_while(|c| c.is_whitespace()))
        .collect::<String>();

    let doc = row_comment
        .repeated()
        .at_least(1)
        .collect::<Vec<_>>()
        .map(|comments| Token::Doc(comments.join("\n")));

    let int = text::int(10).map(Token::Int);
    let string = just('"')
        .ignore_then(filter(|c| *c != '"').repeated())
        .then_ignore(just('"'))
        .collect::<String>()
        .map(Token::Str);

    let atom = int.or(string);

    let starter_char = |c: &char| c.is_ascii_alphabetic() || *c == '_';
    let middle_char = |c: &char| c.is_ascii_alphanumeric() || *c == '_';
    let ident_like = filter(starter_char)
        .chain(filter(middle_char).repeated())
        .collect::<String>();

    let ident = ident_like.map(|word| match word.as_str() {
        "let" => Token::Let,
        "type" => Token::Type,
        "of" => Token::Of,
        "if" => Token::If,
        "then" => Token::Then,
        "else" => Token::Else,
        "match" => Token::Match,
        "in" => Token::In,
        "module" => Token::Module,
        "struct" => Token::Struct,
        "end" => Token::End,
        word => Token::Ident(word.to_owned()),
    });
    let typevar = just('\'').ignore_then(ident_like).map(Token::TypeParameter);
    let spell = just("@").ignore_then(ident_like).map(Token::LiteralSpell);

    let word = choice((ident, typevar, spell));

    let symbol = choice((
        // operators
        just('<').to(Token::Op(Op::Less)),
        just('>').to(Token::Op(Op::Greater)),
        just("==").to(Token::Op(Op::Equal)),
        just("!=").to(Token::Op(Op::NotEqual)),
        just("||").to(Token::Op(Op::Or)),
        just("&&").to(Token::Op(Op::And)),
        just('+').to(Token::Op(Op::Plus)),
        just('-').to(Token::Op(Op::Minus)),
        just('*').to(Token::Op(Op::Product)),
        just('/').to(Token::Op(Op::Divide)),
        // control symbols
        just("=>").to(Token::MatchArrow),
        just('_').to(Token::Wildcard),
        just('=').to(Token::Is),
        just('.').to(Token::Dot),
        just(';').to(Token::Semicolon),
        just(',').to(Token::Comma),
        just('|').to(Token::Either),
        just('(').to(Token::OpenParen),
        just(')').to(Token::CloseParen),
        just('[').to(Token::OpenBracket),
        just(']').to(Token::CloseBracket),
        just('{').to(Token::OpenBrace),
        just('}').to(Token::CloseBrace),
    ));

    let token = choice((doc, atom, symbol, word));

    token
        .map_with_span(|tok, span| (tok, span))
        .padded()
        .repeated()
        .then_ignore(end())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn remove_spans<X, E>(tt: Result<Vec<(X, Span)>, E>) -> Result<Vec<X>, E> {
        tt.map(|tt| tt.into_iter().map(|(t, _s)| t).collect())
    }

    #[test]
    fn tokenize_ident_expr() {
        let src = "zero";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![Token::Ident("zero".to_owned())])
        );
    }

    #[test]
    fn tokenize_integer_expr() {
        let src = "50";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![Token::Int("50".to_owned())])
        );
    }

    #[test]
    fn tokenize_string_expr() {
        let src = r#"
        "hello, kaffee!"
        "#;
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![Token::Str("hello, kaffee!".to_owned())])
        );
    }

    #[test]
    fn tokenize_tuple_expr() {
        let src = "(zero, 5, next)";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::OpenParen,
                Token::Ident("zero".to_owned()),
                Token::Comma,
                Token::Int("5".to_owned()),
                Token::Comma,
                Token::Ident("next".to_owned()),
                Token::CloseParen,
            ])
        );
    }

    #[test]
    fn tokenize_list_expr() {
        let src = "[5 25]";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::OpenBracket,
                Token::Int("5".to_owned()),
                Token::Int("25".to_owned()),
                Token::CloseBracket,
            ])
        );
    }

    #[test]
    fn tokenize_group_expr() {
        let src = "(next 25)";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::OpenParen,
                Token::Ident("next".to_owned()),
                Token::Int("25".to_owned()),
                Token::CloseParen,
            ])
        );
    }

    #[test]
    fn tokenize_from_module_expr() {
        let src = "Nat.succ 5";

        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Ident("Nat".to_owned()),
                Token::Dot,
                Token::Ident("succ".to_owned()),
                Token::Int("5".to_owned()),
            ])
        );
    }

    #[test]
    fn tokenize_arithmetic_expr() {
        let src = "-x + y * z + y";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Op(Op::Minus),
                Token::Ident("x".to_owned()),
                Token::Op(Op::Plus),
                Token::Ident("y".to_owned()),
                Token::Op(Op::Product),
                Token::Ident("z".to_owned()),
                Token::Op(Op::Plus),
                Token::Ident("y".to_owned()),
            ])
        );
    }

    #[test]
    fn tokenize_logic_expr() {
        let src = "x < y && z > y || x == 0 && z != 100";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Ident("x".to_owned()),
                Token::Op(Op::Less),
                Token::Ident("y".to_owned()),
                Token::Op(Op::And),
                Token::Ident("z".to_owned()),
                Token::Op(Op::Greater),
                Token::Ident("y".to_owned()),
                Token::Op(Op::Or),
                Token::Ident("x".to_owned()),
                Token::Op(Op::Equal),
                Token::Int("0".to_owned()),
                Token::Op(Op::And),
                Token::Ident("z".to_owned()),
                Token::Op(Op::NotEqual),
                Token::Int("100".to_owned()),
            ])
        );
    }

    #[test]
    fn tokenize_if_expr() {
        let src = "if x < y then 0 else 1";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::If,
                Token::Ident("x".to_owned()),
                Token::Op(Op::Less),
                Token::Ident("y".to_owned()),
                Token::Then,
                Token::Int("0".to_owned()),
                Token::Else,
                Token::Int("1".to_owned()),
            ])
        );
    }

    #[test]
    fn tokenize_let_in_expr() {
        let src = "let x = 5 in Some x";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Let,
                Token::Ident("x".to_owned()),
                Token::Is,
                Token::Int("5".to_owned()),
                Token::In,
                Token::Ident("Some".to_owned()),
                Token::Ident("x".to_owned()),
            ])
        );
    }

    #[test]
    fn tokenize_let_in_with_spell_expr() {
        let src = r#"
let @plus x y = Int.plus
in x + y
        "#;

        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Let,
                Token::LiteralSpell("plus".to_owned()),
                Token::Ident("x".to_owned()),
                Token::Ident("y".to_owned()),
                Token::Is,
                Token::Ident("Int".to_owned()),
                Token::Dot,
                Token::Ident("plus".to_owned()),
                Token::In,
                Token::Ident("x".to_owned()),
                Token::Op(Op::Plus),
                Token::Ident("y".to_owned()),
            ])
        );
    }

    #[test]
    fn tokenize_match_expr() {
        let src = "match m | Some x => f x | _ => None";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Match,
                Token::Ident("m".to_owned()),
                Token::Either,
                Token::Ident("Some".to_owned()),
                Token::Ident("x".to_owned()),
                Token::MatchArrow,
                Token::Ident("f".to_owned()),
                Token::Ident("x".to_owned()),
                Token::Either,
                Token::Wildcard,
                Token::MatchArrow,
                Token::Ident("None".to_owned()),
            ])
        );
    }

    #[test]
    fn tokenize_let_stmt() {
        let src = "let x = 50;";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Let,
                Token::Ident("x".to_owned()),
                Token::Is,
                Token::Int("50".to_owned()),
                Token::Semicolon,
            ])
        );
    }

    #[test]
    fn tokenize_type_alias_tuple_stmt() {
        let src = "type point = int * int;";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Type,
                Token::Ident("point".to_owned()),
                Token::Is,
                Token::Ident("int".to_owned()),
                Token::Op(Op::Product),
                Token::Ident("int".to_owned()),
                Token::Semicolon,
            ])
        );
    }

    #[test]
    fn tokenize_type_alias_with_spell_stmt() {
        let src = "type @integer = int64;";

        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Type,
                Token::LiteralSpell("integer".to_owned()),
                Token::Is,
                Token::Ident("int64".to_owned()),
                Token::Semicolon,
            ])
        );
    }

    #[test]
    fn tokenize_type_alias_generic_tuple_stmt() {
        let src = "type pair ('a, 'b) = 'a * 'b;";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Type,
                Token::Ident("pair".to_owned()),
                Token::OpenParen,
                Token::TypeParameter("a".to_owned()),
                Token::Comma,
                Token::TypeParameter("b".to_owned()),
                Token::CloseParen,
                Token::Is,
                Token::TypeParameter("a".to_owned()),
                Token::Op(Op::Product),
                Token::TypeParameter("b".to_owned()),
                Token::Semicolon,
            ])
        );
    }

    #[test]
    fn tokenize_type_enum_stmt() {
        let src = "type option 'a = Some of 'a | None;";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Type,
                Token::Ident("option".to_owned()),
                Token::TypeParameter("a".to_owned()),
                Token::Is,
                Token::Ident("Some".to_owned()),
                Token::Of,
                Token::TypeParameter("a".to_owned()),
                Token::Either,
                Token::Ident("None".to_owned()),
                Token::Semicolon,
            ])
        );
    }

    #[test]
    fn tokenize_type_enum_tuple_stmt() {
        let src = "type list 'a = Nil | Cons of 'a * list 'a;";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Type,
                Token::Ident("list".to_owned()),
                Token::TypeParameter("a".to_owned()),
                Token::Is,
                Token::Ident("Nil".to_owned()),
                Token::Either,
                Token::Ident("Cons".to_owned()),
                Token::Of,
                Token::TypeParameter("a".to_owned()),
                Token::Op(Op::Product),
                Token::Ident("list".to_owned()),
                Token::TypeParameter("a".to_owned()),
                Token::Semicolon,
            ])
        );
    }

    #[test]
    fn tokenize_type_record_stmt() {
        let src = "type date = {year of int, month of int, day of int}";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Type,
                Token::Ident("date".to_owned()),
                Token::Is,
                Token::OpenBrace,
                Token::Ident("year".to_owned()),
                Token::Of,
                Token::Ident("int".to_owned()),
                Token::Comma,
                Token::Ident("month".to_owned()),
                Token::Of,
                Token::Ident("int".to_owned()),
                Token::Comma,
                Token::Ident("day".to_owned()),
                Token::Of,
                Token::Ident("int".to_owned()),
                Token::CloseBrace,
            ])
        );
    }

    #[test]
    fn tokenize_module_stmt() {
        let src = r#"
module Option = struct
    type option 'a = Some of 'a | None;
    let map m f = match m
        | Some x => f x
        | None => None;
end
        "#;
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                // module declaration
                Token::Module,
                Token::Ident("Option".to_owned()),
                Token::Is,
                Token::Struct,
                // type declaration
                Token::Type,
                Token::Ident("option".to_owned()),
                Token::TypeParameter("a".to_owned()),
                Token::Is,
                Token::Ident("Some".to_owned()),
                Token::Of,
                Token::TypeParameter("a".to_owned()),
                Token::Either,
                Token::Ident("None".to_owned()),
                Token::Semicolon,
                // function declaration
                Token::Let,
                Token::Ident("map".to_owned()),
                Token::Ident("m".to_owned()),
                Token::Ident("f".to_owned()),
                Token::Is,
                // match expression
                Token::Match,
                Token::Ident("m".to_owned()),
                Token::Either,
                Token::Ident("Some".to_owned()),
                Token::Ident("x".to_owned()),
                Token::MatchArrow,
                Token::Ident("f".to_owned()),
                Token::Ident("x".to_owned()),
                Token::Either,
                Token::Ident("None".to_owned()),
                Token::MatchArrow,
                Token::Ident("None".to_owned()),
                Token::Semicolon,
                // end of module declaration
                Token::End,
            ])
        );
    }

    #[test]
    fn tokenize_commented_stmt() {
        let src = r#"
// (+) operator wrapper
let add x y = x + y;
        "#;

        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Doc("(+) operator wrapper".to_owned()),
                Token::Let,
                Token::Ident("add".to_owned()),
                Token::Ident("x".to_owned()),
                Token::Ident("y".to_owned()),
                Token::Is,
                Token::Ident("x".to_owned()),
                Token::Op(Op::Plus),
                Token::Ident("y".to_owned()),
                Token::Semicolon,
            ])
        );
    }
}
