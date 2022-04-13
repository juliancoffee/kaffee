#![warn(clippy::pedantic)]
#![allow(unused)]
use chumsky::prelude::*;

type Span = std::ops::Range<usize>;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
enum Op {
    // +
    Plus,
    // -
    Minus,
    // *
    Product,
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
enum Token {
    // Words
    Int(String),
    Str(String),
    Ident(String),
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
    // Binders
    Of,
    Is,
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
}

impl Token {
    fn op_expect(&self) -> &Op {
        match self {
            Self::Op(op) => op,
            token => panic!("expected Token::Op(_), got {token:?}"),
        }
    }
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

    let op = just('<')
        .to(Token::Op(Op::Less))
        .or(just('>').to(Token::Op(Op::Greater)))
        .or(just("==").to(Token::Op(Op::Equal)))
        .or(just("!=").to(Token::Op(Op::NotEqual)))
        .or(just("||").to(Token::Op(Op::Or)))
        .or(just("&&").to(Token::Op(Op::And)))
        .or(just('+').to(Token::Op(Op::Plus)))
        .or(just('-').to(Token::Op(Op::Minus)))
        .or(just('*').to(Token::Op(Op::Product)));

    let ctrl = just("=>")
        .to(Token::MatchArrow)
        .or(just('=').to(Token::Is))
        .or(just(';').to(Token::SemiColon))
        .or(just(',').to(Token::Comma))
        .or(just('|').to(Token::Either))
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
enum UnOp {
    Negative,
}

#[derive(Debug, PartialEq, Clone)]
enum BinOp {
    // arithmetic
    Sum,
    Sub,
    Product,
    // logic
    Less,
    Greater,
    Equal,
    NotEqual,
    And,
    Or,
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
    Unary(UnOp, Box<Expr>),
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    App(Box<Expr>, Box<Expr>),
}

impl Expr {
    fn ident(name: &str) -> Self {
        Expr::Ident(name.to_owned())
    }

    fn app(f: Self, x: Self) -> Self {
        Self::App(Box::new(f), Box::new(x))
    }

    fn make_unary(token: Token, x: Self) -> Self {
        let op = token.op_expect();
        let unary = match op {
            Op::Minus => UnOp::Negative,
            _ => unreachable!(),
        };
        Self::unary_with(unary, x)
    }

    fn unary_with(op: UnOp, x: Self) -> Self {
        Self::Unary(op, Box::new(x))
    }

    fn make_binop(token: Token, x: Self, y: Self) -> Self {
        let op = token.op_expect();
        let binop = match op {
            Op::Minus => BinOp::Sub,
            Op::Plus => BinOp::Sum,
            Op::Product => BinOp::Product,
            Op::Less => BinOp::Less,
            Op::Greater => BinOp::Greater,
            Op::Equal => BinOp::Equal,
            Op::NotEqual => BinOp::NotEqual,
            Op::And => BinOp::And,
            Op::Or => BinOp::Or,
            _ => todo!(),
        };
        Self::binop_with(binop, x, y)
    }

    fn binop_with(op: BinOp, x: Self, y: Self) -> Self {
        Self::BinOp(op, Box::new(x), Box::new(y))
    }
}

fn expression() -> impl Parser<Token, (Expr, Span), Error = Simple<Token>> {
    recursive(|expr| {
        let literal = select! {
            Token::Int(i) => Expr::Int(i.parse().unwrap()),
            Token::Ident(i) => Expr::Ident(i),
            Token::Str(s) => Expr::Str(s),
        };

        let enclosed = literal.or(expr
            .clone()
            .delimited_by(just(Token::OpenParen), just(Token::CloseParen)));

        let tuple = expr
            .clone()
            .separated_by(just(Token::Comma))
            .delimited_by(just(Token::OpenParen), just(Token::CloseParen))
            .collect::<Vec<_>>()
            .map(Expr::Tuple);

        let atom = enclosed.or(tuple);

        #[cfg(debug_assertions)]
        let atom = atom.boxed();

        let list = recursive(|sublist| {
            let element = atom.clone().or(sublist.clone());

            element
                .clone()
                .repeated()
                .delimited_by(
                    just(Token::OpenBracket),
                    just(Token::CloseBracket),
                )
                .collect::<Vec<_>>()
                .map(Expr::List)
        });

        let term = atom.or(list);

        #[cfg(debug_assertions)]
        let term = term.boxed();
        // Parse operators
        // 0) unary
        // 1) *
        // 2) +, -
        // 3) <, >
        // 4) ==, !=
        // 5) ||, &&

        // parses to  unary(expr) | expr
        let unary = just(Token::Op(Op::Minus))
            .repeated()
            .then(term.clone())
            .foldr(Expr::make_unary);

        // parses to product(expr) | expr
        let product = unary
            .clone()
            .then(just(Token::Op(Op::Product)).then(unary.clone()).repeated())
            .foldl(|x, (op, y)| Expr::make_binop(op, x, y));

        // parses to arithmetic(expr) | expr
        let arithmetic = product
            .clone()
            .then(
                just(Token::Op(Op::Plus))
                    .or(just(Token::Op(Op::Minus)))
                    .then(product.clone())
                    .repeated(),
            )
            .foldl(|x, (op, y)| Expr::make_binop(op, x, y));
        #[cfg(debug_assertions)]
        let arithmetic = arithmetic.boxed();

        // parses to cmp_order(expr)
        let cmp_order = arithmetic
            .clone()
            .then(
                just(Token::Op(Op::Less))
                    .or(just(Token::Op(Op::Greater)))
                    .then(arithmetic.clone()),
            )
            .map(|(x, (op, y))| Expr::make_binop(op, x, y));

        // parses to cmp_order(expr) | expr
        let operand = cmp_order.or(arithmetic);

        // parses to is_equal(expr)
        let is_equal = operand
            .clone()
            .then(
                just(Token::Op(Op::Equal))
                    .or(just(Token::Op(Op::NotEqual)))
                    .then(operand.clone()),
            )
            .map(|(x, (op, y))| Expr::make_binop(op, x, y));

        // parses to is_equal(expr) | expr
        let element = is_equal.or(operand);

        // parses to and(expr) | expr
        let and = element
            .clone()
            .then(just(Token::Op(Op::And)).then(element.clone()).repeated())
            .foldl(|a, (op, b)| Expr::make_binop(op, a, b));

        // parses to or(expr) | expr
        let or = and
            .clone()
            .then(just(Token::Op(Op::Or)).then(and.clone()).repeated())
            .foldl(|a, (op, b)| Expr::make_binop(op, a, b));

        let app = or.clone().then(or.clone().repeated()).foldl(Expr::app);
        app
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
        assert_eq!(parse_expr(src), Ok(Expr::ident("zero")))
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
                Expr::ident("zero"),
                Expr::Int(5),
                Expr::ident("next"),
            ])),
        )
    }

    #[test]
    fn parse_tuple_nested_expr() {
        let src = "(zero, (1, 2), 5, next)";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::Tuple(vec![
                Expr::ident("zero"),
                Expr::Tuple(vec![Expr::Int(1), Expr::Int(2)]),
                Expr::Int(5),
                Expr::ident("next"),
            ])),
        )
    }

    #[test]
    fn parse_tuple_nested_call_expr() {
        let src = "(zero, (1, 2), (next 4), next)";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::Tuple(vec![
                Expr::ident("zero"),
                Expr::Tuple(vec![Expr::Int(1), Expr::Int(2)]),
                Expr::app(Expr::ident("next"), Expr::Int(4)),
                Expr::ident("next"),
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
            Ok(Expr::List(vec![Expr::Int(5), Expr::Int(25)]))
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
            Ok(Expr::app(Expr::ident("next"), Expr::Int(25))),
        );
    }

    #[test]
    fn parse_grouped_call_expr() {
        let src = "(next 25)";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::app(Expr::ident("next"), Expr::Int(25))),
        )
    }

    #[test]
    fn parse_list_with_group_expr() {
        let src = "[5 zero (next 25)]";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::List(vec![
                Expr::Int(5),
                Expr::ident("zero"),
                Expr::app(Expr::ident("next"), Expr::Int(25)),
            ]))
        )
    }

    #[test]
    fn parse_list_nested_expr() {
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
                Expr::app(
                    Expr::ident("sum"),
                    Expr::List(vec![Expr::Int(5), Expr::Int(5)])
                ),
            ]))
        )
    }

    #[test]
    fn parse_call_on_complex_list() {
        let src = "sum [5 (sum [5 5])]";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::app(
                Expr::ident("sum"),
                Expr::List(vec![
                    Expr::Int(5),
                    Expr::app(
                        Expr::ident("sum"),
                        Expr::List(vec![Expr::Int(5), Expr::Int(5)])
                    ),
                ])
            ))
        )
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
    fn parse_neg_expr() {
        let src = "-x";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::unary_with(UnOp::Negative, Expr::ident("x")))
        )
    }

    #[test]
    fn parse_neg_duplicated_expr() {
        let src = "- - -x";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::unary_with(
                UnOp::Negative,
                Expr::unary_with(
                    UnOp::Negative,
                    Expr::unary_with(UnOp::Negative, Expr::ident("x")),
                ),
            ))
        )
    }

    #[test]
    fn parse_product_expr() {
        let src = "x * 5";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::binop_with(
                BinOp::Product,
                Expr::ident("x"),
                Expr::Int(5),
            ))
        )
    }

    #[test]
    fn parse_sum_expr() {
        let src = "5 + x";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::binop_with(BinOp::Sum, Expr::Int(5), Expr::ident("x")))
        )
    }

    #[test]
    fn parse_sub_expr() {
        let src = "5 - x";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::binop_with(BinOp::Sub, Expr::Int(5), Expr::ident("x")))
        )
    }

    #[test]
    fn parse_arithmetic_expr() {
        let src = "-x + y * z + y";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::binop_with(
                BinOp::Sum,
                Expr::binop_with(
                    BinOp::Sum,
                    Expr::unary_with(UnOp::Negative, Expr::ident("x")),
                    Expr::binop_with(
                        BinOp::Product,
                        Expr::ident("y"),
                        Expr::ident("z"),
                    )
                ),
                Expr::ident("y"),
            ))
        )
    }

    #[test]
    fn parse_arithmetic_in_list_expr() {
        let src = "[(-x + y * z + y) (x + 5)]";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::List(vec![
                Expr::binop_with(
                    BinOp::Sum,
                    Expr::binop_with(
                        BinOp::Sum,
                        Expr::unary_with(UnOp::Negative, Expr::ident("x")),
                        Expr::binop_with(
                            BinOp::Product,
                            Expr::ident("y"),
                            Expr::ident("z"),
                        )
                    ),
                    Expr::ident("y"),
                ),
                Expr::binop_with(BinOp::Sum, Expr::ident("x"), Expr::Int(5)),
            ]))
        )
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
    fn parse_less_expr() {
        let src = "x < y";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::binop_with(
                BinOp::Less,
                Expr::ident("x"),
                Expr::ident("y")
            )),
        )
    }

    #[test]
    fn parse_greater_expr() {
        let src = "x > y";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::binop_with(
                BinOp::Greater,
                Expr::ident("x"),
                Expr::ident("y")
            )),
        )
    }

    #[test]
    fn parse_repeated_ordering_expr() {
        let src = "x < y < z";
        assert!(parse_expr(src).is_err())
    }

    #[test]
    fn parse_equal_expr() {
        let src = "x == 0";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::binop_with(
                BinOp::Equal,
                Expr::ident("x"),
                Expr::Int(0)
            ))
        )
    }

    #[test]
    fn parse_notequal_expr() {
        let src = "x != 0";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::binop_with(
                BinOp::NotEqual,
                Expr::ident("x"),
                Expr::Int(0)
            ))
        )
    }

    #[test]
    fn parse_and_expr() {
        let src = "a && b";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::binop_with(
                BinOp::And,
                Expr::ident("a"),
                Expr::ident("b")
            ))
        )
    }

    #[test]
    fn parse_or_expr() {
        let src = "a || b";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::binop_with(
                BinOp::Or,
                Expr::ident("a"),
                Expr::ident("b")
            ))
        )
    }

    #[test]
    fn parse_repeated_or_expr() {
        let src = "a || b || c";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::binop_with(
                BinOp::Or,
                Expr::binop_with(BinOp::Or, Expr::ident("a"), Expr::ident("b")),
                Expr::ident("c"),
            ))
        )
    }

    #[test]
    fn parse_logic_combination_expr() {
        let src = "a || b && c || d";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::binop_with(
                BinOp::Or,
                Expr::binop_with(
                    BinOp::Or,
                    Expr::ident("a"),
                    Expr::binop_with(
                        BinOp::And,
                        Expr::ident("b"),
                        Expr::ident("c"),
                    ),
                ),
                Expr::ident("d"),
            ))
        )
    }

    #[test]
    fn parse_logic_forced_precedence_expr() {
        let src = "a || b && (c || d)";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::binop_with(
                BinOp::Or,
                Expr::ident("a"),
                Expr::binop_with(
                    BinOp::And,
                    Expr::ident("b"),
                    Expr::binop_with(
                        BinOp::Or,
                        Expr::ident("c"),
                        Expr::ident("d"),
                    ),
                ),
            ))
        )
    }

    #[test]
    fn parse_logic_expr() {
        let src = "x < y && z > y || x == 0 && z != 100";

        assert_eq!(
            parse_expr(src),
            Ok(Expr::binop_with(
                BinOp::Or,
                Expr::binop_with(
                    BinOp::And,
                    Expr::binop_with(
                        BinOp::Less,
                        Expr::ident("x"),
                        Expr::ident("y"),
                    ),
                    Expr::binop_with(
                        BinOp::Greater,
                        Expr::ident("z"),
                        Expr::ident("y"),
                    ),
                ),
                Expr::binop_with(
                    BinOp::And,
                    Expr::binop_with(
                        BinOp::Equal,
                        Expr::ident("x"),
                        Expr::Int(0),
                    ),
                    Expr::binop_with(
                        BinOp::NotEqual,
                        Expr::ident("z"),
                        Expr::Int(100),
                    ),
                ),
            ))
        )
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
                Token::Is,
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
                Token::Is,
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
                Token::Is,
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
                Token::Is,
                Token::Ident("Nil".to_owned()),
                Token::Either,
                Token::Ident("Cons".to_owned()),
                Token::Of,
                Token::TypeParameter("a".to_owned()),
                Token::Op(Op::Product),
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
                Token::Is,
                Token::Ident("int".to_owned()),
                Token::Op(Op::Product),
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
                Token::Is,
                Token::Struct,
                // type declaration
                Token::Type,
                Token::TypeParameter("a".to_owned()),
                Token::Ident("option".to_owned()),
                Token::Is,
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
                Token::SemiColon,
                // end of module declaration
                Token::End,
            ])
        );
    }
}
