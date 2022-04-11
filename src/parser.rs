#![allow(unused)]
use chumsky::prelude::*;

type Span = std::ops::Range<usize>;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
enum Token {
    // Words
    Int(String),
    Str(String),
    Ident(String),
    TypeParameter(String),
    // Operators
    BinOp(String),
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
    // Binders
    Of,
    Equal,
    In,
    // Finishers
    SemiColon,
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
    Product,
}

fn lexer() -> impl Parser<char, Vec<(Token, Span)>, Error = Simple<char>> {
    let int = text::int(10).map(Token::Int);
    let string = just('"')
        .ignore_then(filter(|c| *c != '"').repeated())
        .then_ignore(just('"'))
        .collect::<String>()
        .map(Token::Str);

    let atom = int.or(string);

    let alphabetic = filter(char::is_ascii_alphabetic)
        .repeated()
        .at_least(1)
        .collect::<String>();

    let word = alphabetic.map(|word| match word.as_str() {
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
    let marked = just('\'').ignore_then(alphabetic).map(Token::TypeParameter);

    let onechar_op = just('<')
        .or(just('>'))
        .map(|op| Token::BinOp(op.to_string()));
    let multichar_op = just("||").map(|op| Token::BinOp(op.to_owned()));
    let op = onechar_op.or(multichar_op);

    let ctrl = just("=>")
        .to(Token::MatchArrow)
        .or(just('=').to(Token::Equal))
        .or(just(';').to(Token::SemiColon))
        .or(just(',').to(Token::Comma))
        .or(just('|').to(Token::Either))
        .or(just('*').to(Token::Product))
        .or(just('(').to(Token::OpenParen))
        .or(just(')').to(Token::CloseParen))
        .or(just('[').to(Token::OpenBracket))
        .or(just(']').to(Token::CloseBracket))
        .or(just('{').to(Token::OpenBrace))
        .or(just('}').to(Token::CloseBrace));

    let token = atom.or(op).or(ctrl).or(marked).or(word);
    //.recover_with(skip_then_retry_until([]));

    token
        .map_with_span(|tok, span| (tok, span))
        .padded()
        .repeated()
        .then_ignore(end())
}

#[derive(Debug, PartialEq, Clone)]
enum Expr {
    // Atoms
    Int(u64),
    Ident(String),
    Str(String),
    // Collections
    List(Vec<Expr>),
    Tuple(Vec<Expr>),
    // Complex
    Call { called: Box<Expr>, args: Vec<Expr> },
}

fn expression() -> impl Parser<Token, (Expr, Span), Error = Simple<Token>> {
    recursive(|expr| {
        let atom = select! {
            Token::Int(i) => Expr::Int(i.parse().unwrap()),
            Token::Ident(i) => Expr::Ident(i),
            Token::Str(s) => Expr::Str(s),
        };

        let grouped = expr
            .clone()
            .delimited_by(just(Token::OpenParen), just(Token::CloseParen));

        let tuple = expr
            .clone()
            .separated_by(just(Token::Comma))
            .delimited_by(just(Token::OpenParen), just(Token::CloseParen))
            .collect::<Vec<_>>()
            .map(Expr::Tuple);

        let atom = atom.or(grouped).or(tuple);

        let list = recursive(|sublist| {
            let element = atom.clone().or(sublist.clone());

            element
                .clone()
                .repeated()
                .delimited_by(just(Token::OpenBracket), just(Token::CloseBracket))
                .collect::<Vec<_>>()
                .map(Expr::List)
        });

        let term = atom.or(list);

        let call = term.clone().then(term.clone().repeated()).map(|(a, args)| {
            if args.is_empty() {
                a
            } else {
                Expr::Call {
                    called: Box::new(a),
                    args,
                }
            }
        });
        let complex = call;

        complex
    })
    .map_with_span(|expr, span| (expr, span))
    .then_ignore(end())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chumsky::Stream;

    fn remove_spans<X, E>(tt: Result<Vec<(X, Span)>, E>) -> Result<Vec<X>, E> {
        tt.map(|tt| tt.into_iter().map(|(t, _s)| t).collect())
    }

    fn remove_span<X, E>(x: Result<(X, Span), E>) -> Result<X, E> {
        x.map(|i| {
            let (x, _s) = i;
            x
        })
    }

    fn parse_expr(src: &str) -> Result<Expr, Vec<Simple<Token>>> {
        let tokens = lexer().parse(src).unwrap();
        let end = src.chars().count();
        let token_stream = Stream::from_iter(end..end + 1, tokens.into_iter());

        remove_span(expression().parse(token_stream))
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
    fn parse_ident_expr() {
        let src = "zero";
        assert_eq!(parse_expr(src), Ok(Expr::Ident("zero".to_owned())))
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
    fn parse_integer_expr() {
        let src = "50";

        assert_eq!(parse_expr(src), Ok(Expr::Int(50)))
    }

    #[test]
    fn tokenize_string_expr() {
        let src = r#"
        "hello, kaffee!"
        "#;
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![Token::Str("hello, kaffee!".to_owned())])
        )
    }

    #[test]
    fn parse_string_expr() {
        let src = r#"
        "hello kaffee!"
        "#;

        assert_eq!(parse_expr(src), Ok(Expr::Str("hello kaffee!".to_owned())))
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
        )
    }

    #[test]
    fn parse_tuple_expr() {
        let src = "(zero, 5, next)";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::Tuple(vec![
                Expr::Ident("zero".to_owned()),
                Expr::Int(5),
                Expr::Ident("next".to_owned()),
            ])),
        )
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
    fn parse_list_expr() {
        let src = "[5 25]";

        assert_eq!(
            parse_expr(src),
            Ok(Expr::List(vec![Expr::Int(5), Expr::Int(25),]))
        )
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
    fn parse_grouped_integer_expr() {
        let src = "(5)";
        assert_eq!(parse_expr(src), Ok(Expr::Int(5)));
    }

    #[test]
    fn parse_call_expr() {
        let src = "next 25";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::Call {
                called: Box::new(Expr::Ident("next".to_owned())),
                args: vec![Expr::Int(25)],
            }),
        );
    }

    #[test]
    fn parse_grouped_call_expr() {
        let src = "(next 25)";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::Call {
                called: Box::new(Expr::Ident("next".to_owned())),
                args: vec![Expr::Int(25)],
            }),
        )
    }

    #[test]
    fn parse_list_with_group_expr() {
        let src = "[5 zero (next 25)]";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::List(vec![
                Expr::Int(5),
                Expr::Ident("zero".to_owned()),
                Expr::Call {
                    called: Box::new(Expr::Ident("next".to_owned())),
                    args: vec![Expr::Int(25)],
                },
            ]))
        )
    }

    #[test]
    fn parse_nested_list() {
        let src = "[[5 5] [5 5]]";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::List(vec![
                Expr::List(vec![Expr::Int(5), Expr::Int(5)]),
                Expr::List(vec![Expr::Int(5), Expr::Int(5)]),
            ]))
        )
    }

    #[test]
    fn parse_list_with_call_on_list() {
        let src = "[5 (sum [5 5])]";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::List(vec![
                Expr::Int(5),
                Expr::Call {
                    called: Box::new(Expr::Ident("sum".to_owned())),
                    args: vec![Expr::List(vec![Expr::Int(5), Expr::Int(5)])],
                },
            ]))
        )
    }

    #[test]
    fn parse_call_on_complex_list() {
        let src = "sum [5 (sum [5 5])]";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::Call {
                called: Box::new(Expr::Ident("sum".to_owned())),
                args: vec![Expr::List(vec![
                    Expr::Int(5),
                    Expr::Call {
                        called: Box::new(Expr::Ident("sum".to_owned())),
                        args: vec![Expr::List(vec![Expr::Int(5), Expr::Int(5)])],
                    }
                ])]
            })
        )
    }

    #[test]
    fn tokenize_binop_expr() {
        let src = "x < y || z > y";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Ident("x".to_owned()),
                Token::BinOp("<".to_owned()),
                Token::Ident("y".to_owned()),
                Token::BinOp("||".to_owned()),
                Token::Ident("z".to_owned()),
                Token::BinOp(">".to_owned()),
                Token::Ident("y".to_owned()),
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
                Token::BinOp("<".to_owned()),
                Token::Ident("y".to_owned()),
                Token::Then,
                Token::Int("0".to_owned()),
                Token::Else,
                Token::Int("1".to_owned()),
            ])
        );
    }

    #[test]
    fn tokenize_match_expr() {
        let src = "match m | Some x => f x | None => None";
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
                Token::Ident("None".to_owned()),
                Token::MatchArrow,
                Token::Ident("None".to_owned()),
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
                Token::Equal,
                Token::Int("5".to_owned()),
                Token::In,
                Token::Ident("Some".to_owned()),
                Token::Ident("x".to_owned()),
            ])
        );
    }

    #[test]
    fn tokenize_let_stmnt() {
        let src = "let x = 50;";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Let,
                Token::Ident("x".to_owned()),
                Token::Equal,
                Token::Int("50".to_owned()),
                Token::SemiColon,
            ])
        );
    }

    #[test]
    fn tokenize_enum_stmnt() {
        let src = "type 'a option = Some of 'a | None;";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Type,
                Token::TypeParameter("a".to_owned()),
                Token::Ident("option".to_owned()),
                Token::Equal,
                Token::Ident("Some".to_owned()),
                Token::Of,
                Token::TypeParameter("a".to_owned()),
                Token::Either,
                Token::Ident("None".to_owned()),
                Token::SemiColon,
            ])
        );
    }

    #[test]
    fn tokenize_enum_tuple_stmnt() {
        let src = "type 'a list = Nil | Cons of 'a * 'a list;";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Type,
                Token::TypeParameter("a".to_owned()),
                Token::Ident("list".to_owned()),
                Token::Equal,
                Token::Ident("Nil".to_owned()),
                Token::Either,
                Token::Ident("Cons".to_owned()),
                Token::Of,
                Token::TypeParameter("a".to_owned()),
                Token::Product,
                Token::TypeParameter("a".to_owned()),
                Token::Ident("list".to_owned()),
                Token::SemiColon,
            ])
        );
    }

    #[test]
    fn tokenize_tuple_stmnt() {
        let src = "type point = int * int;";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Type,
                Token::Ident("point".to_owned()),
                Token::Equal,
                Token::Ident("int".to_owned()),
                Token::Product,
                Token::Ident("int".to_owned()),
                Token::SemiColon,
            ])
        )
    }

    #[test]
    fn tokenize_record_stmnt() {
        let src = "type date = {year of int, month of int, day of int}";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Type,
                Token::Ident("date".to_owned()),
                Token::Equal,
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
        )
    }

    #[test]
    fn tokenize_module_stmnt() {
        let src = r#"
module Option = struct
    type 'a option = Some of 'a | None;
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
                Token::Equal,
                Token::Struct,
                // type declaration
                Token::Type,
                Token::TypeParameter("a".to_owned()),
                Token::Ident("option".to_owned()),
                Token::Equal,
                Token::Ident("Some".to_owned()),
                Token::Of,
                Token::TypeParameter("a".to_owned()),
                Token::Either,
                Token::Ident("None".to_owned()),
                Token::SemiColon,
                // function declaration
                Token::Let,
                Token::Ident("map".to_owned()),
                Token::Ident("m".to_owned()),
                Token::Ident("f".to_owned()),
                Token::Equal,
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
                Token::SemiColon,
                // end of module declaration
                Token::End,
            ])
        );
    }
}
