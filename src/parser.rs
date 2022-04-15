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

    let op = choice((
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
    ));

    let ctrl = choice((
        just("=>").to(Token::MatchArrow),
        just('=').to(Token::Is),
        just(';').to(Token::SemiColon),
        just(',').to(Token::Comma),
        just('|').to(Token::Either),
        just('(').to(Token::OpenParen),
        just(')').to(Token::CloseParen),
        just('[').to(Token::OpenBracket),
        just(']').to(Token::CloseBracket),
        just('{').to(Token::OpenBrace),
        just('}').to(Token::CloseBrace),
    ));

    let token = choice((atom, op, ctrl, marked, word));

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
    Divide,
    // logic
    Less,
    Greater,
    Equal,
    NotEqual,
    And,
    Or,
}

#[derive(Debug, PartialEq, Clone)]
enum Pattern {
    Ident(String),
}

impl Pattern {
    fn ident(name: &str) -> Self {
        Self::Ident(name.to_owned())
    }
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
    // Compound
    // if <cond> then <yes> else <no>
    If {
        cond: Box<Expr>,
        yes: Box<Expr>,
        // TODO: make optional?
        no: Box<Expr>,
    },
    // let <pat> = <src> in <target>
    LetIn {
        pat: Pattern,
        src: Box<Expr>,
        target: Box<Expr>,
    },
}

impl Expr {
    fn ident(name: &str) -> Self {
        Self::Ident(name.to_owned())
    }

    fn app(f: Self, x: Self) -> Self {
        Self::App(Box::new(f), Box::new(x))
    }

    #[allow(clippy::needless_pass_by_value)]
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

    #[allow(clippy::needless_pass_by_value)]
    fn make_binop(token: Token, x: Self, y: Self) -> Self {
        let op = token.op_expect();
        let binop = match op {
            Op::Minus => BinOp::Sub,
            Op::Plus => BinOp::Sum,
            Op::Product => BinOp::Product,
            Op::Divide => BinOp::Divide,
            Op::Less => BinOp::Less,
            Op::Greater => BinOp::Greater,
            Op::Equal => BinOp::Equal,
            Op::NotEqual => BinOp::NotEqual,
            Op::And => BinOp::And,
            Op::Or => BinOp::Or,
        };
        Self::binop_with(binop, x, y)
    }

    fn binop_with(op: BinOp, x: Self, y: Self) -> Self {
        Self::BinOp(op, Box::new(x), Box::new(y))
    }

    fn if_with(cond: Self, then: Self, otherwise: Self) -> Self {
        Self::If {
            cond: Box::new(cond),
            yes: Box::new(then),
            no: Box::new(otherwise),
        }
    }

    fn letin_with(pat: Pattern, src: Self, target: Self) -> Self {
        Self::LetIn {
            pat,
            src: Box::new(src),
            target: Box::new(target),
        }
    }
}

// trait aliases
trait ExprParser: Parser<Token, Expr, Error = Simple<Token>> {}
impl<T> ExprParser for T where T: Parser<Token, Expr, Error = Simple<Token>> {}

trait SubParser<O>: Parser<Token, O, Error = Simple<Token>> {}
impl<T, O> SubParser<O> for T where T: Parser<Token, O, Error = Simple<Token>> {}

// Parses pattern
fn pattern() -> impl SubParser<Pattern> + Clone {
    select! {
        Token::Ident(i) => Pattern::Ident(i)
    }
}

// Parses `let <pat> = <expr>` binding
//
// Results in `pat` and `expr`
fn binding<P: ExprParser + Clone>(
    expr: P,
) -> impl SubParser<(Pattern, Expr)> + Clone {
    just(Token::Let)
        .ignore_then(pattern())
        .then_ignore(just(Token::Is))
        .then(expr)
}

// Parses grouped expression
//
// Result in whatever Expr were inside parens
fn grouping<P: ExprParser>(expr: P) -> impl ExprParser {
    expr.delimited_by(just(Token::OpenParen), just(Token::CloseParen))
}

// Parses tuple expression
//
// TODO: remake to be used with patterns?
fn tuple<P: ExprParser>(term: P) -> impl ExprParser {
    term.separated_by(just(Token::Comma))
        .delimited_by(just(Token::OpenParen), just(Token::CloseParen))
        .collect::<Vec<_>>()
        .map(Expr::Tuple)
}

// Parses list expression
fn list<P: ExprParser + 'static>(term: P) -> impl ExprParser {
    recursive(|sublist| {
        let element = term.or(sublist.clone());

        element
            .repeated()
            .delimited_by(just(Token::OpenBracket), just(Token::CloseBracket))
            .collect::<Vec<_>>()
            .map(Expr::List)
    })
}

// Parses application `e e e`
//
// Returns expression 'as is' if none of applications were found
//
// TODO: remake to be used with patterns?
fn app<P: ExprParser + Clone>(term: P) -> impl ExprParser + Clone {
    term.clone().then(term.repeated()).foldl(Expr::app)
}

// Parses `- x` (for any amount of negations)
//
// If none negations were found just returns expression
fn unary<P: ExprParser + Clone>(term: P) -> impl ExprParser + Clone {
    just(Token::Op(Op::Minus))
        .repeated()
        .then(term)
        .foldr(Expr::make_unary)
}

// Parses product expression `x * y` | `x / y`
// (for any amount of operators)
//
// If none operators were found just returns expression
fn product<P: ExprParser + Clone>(term: P) -> impl ExprParser + Clone {
    term.clone()
        .then(
            just(Token::Op(Op::Product))
                .or(just(Token::Op(Op::Divide)))
                .then(term)
                .repeated(),
        )
        .foldl(|x, (op, y)| Expr::make_binop(op, x, y))
}

// Parses arithmetic expression `x + y` | `x - y`
// (for any amount of operators)
//
// If none operators were found just returns expression
fn arithmetic<P: ExprParser + Clone>(term: P) -> impl ExprParser + Clone {
    term.clone()
        .then(
            just(Token::Op(Op::Plus))
                .or(just(Token::Op(Op::Minus)))
                .then(term)
                .repeated(),
        )
        .foldl(|x, (op, y)| Expr::make_binop(op, x, y))
}

// Parses order comparisons `x < y` | `x > y`
fn order<P: ExprParser + Clone>(term: P) -> impl ExprParser + Clone {
    term.clone()
        .then(
            just(Token::Op(Op::Less))
                .or(just(Token::Op(Op::Greater)))
                .then(term),
        )
        .map(|(x, (op, y))| Expr::make_binop(op, x, y))
}

// Parses equality checks `a == b` | `a != b`
fn equality<P: ExprParser + Clone>(term: P) -> impl ExprParser + Clone {
    term.clone()
        .then(
            just(Token::Op(Op::Equal))
                .or(just(Token::Op(Op::NotEqual)))
                .then(term),
        )
        .map(|(x, (op, y))| Expr::make_binop(op, x, y))
}

// Parses logic and `a && b` (for any amount of &&)
//
// If none of `&&` were found just returns expression
fn and<P: ExprParser + Clone>(term: P) -> impl ExprParser + Clone {
    term.clone()
        .then(just(Token::Op(Op::And)).then(term).repeated())
        .foldl(|a, (op, b)| Expr::make_binop(op, a, b))
}

// Parses logic or `a || b` (for any amount of ||)
//
// If none of `||` were found just returns expression
fn or<P: ExprParser + Clone>(term: P) -> impl ExprParser + Clone {
    term.clone()
        .then(just(Token::Op(Op::Or)).then(term).repeated())
        .foldl(|a, (op, b)| Expr::make_binop(op, a, b))
}

// Parses if <cond> then <yes> else <no>
fn if_expr<P: ExprParser + Clone + 'static>(
    term: P,
) -> impl ExprParser + Clone {
    recursive(|subif| {
        let expr = term.clone().or(subif);

        just(Token::If)
            .ignore_then(expr.clone())
            .then_ignore(just(Token::Then))
            .then(expr.clone())
            .then_ignore(just(Token::Else))
            .then(expr)
            .map(|((cond, then), otherwise)| {
                Expr::if_with(cond, then, otherwise)
            })
    })
}

// Parses let <pat> = <src> in <tgt>
fn let_in<P: ExprParser + Clone>(term: P) -> impl ExprParser + Clone {
    binding(term.clone())
        .then_ignore(just(Token::In))
        .then(term)
        .map(|((pat, src), target)| Expr::letin_with(pat, src, target))
}

fn expression() -> impl ExprParser + Clone {
    recursive(|expr| {
        let literal = select! {
            Token::Int(i) => Expr::Int(i.parse().unwrap()),
            Token::Ident(i) => Expr::Ident(i),
            Token::Str(s) => Expr::Str(s),
        };

        // literal | grouping | tuple
        let atom =
            choice((literal, grouping(expr.clone()), tuple(expr.clone())));
        #[cfg(debug_assertions)]
        let atom = atom.boxed();
        // list
        let list = list(atom.clone());
        let term = atom.or(list);
        #[cfg(debug_assertions)]
        let term = term.boxed();
        // application (e e e)
        let app = app(term.clone());
        // unary negation
        let unary = unary(app.clone());
        // product (*, /)
        let product = product(unary.clone());
        // arithmetic (+, -)
        let arithmetic = arithmetic(product.clone());
        #[cfg(debug_assertions)]
        let arithmetic = arithmetic.boxed();
        // order (<, >)
        let order = order(arithmetic.clone());
        let operand = order.or(arithmetic);
        // equality (==, !=)
        let equality = equality(operand.clone());
        let element = equality.or(operand);
        #[cfg(debug_assertions)]
        let element = element.boxed();
        // and (&&)
        let and = and(element.clone());
        // or
        let or = or(and.clone());
        // if <cond> then <yes> else <no>
        let if_expr = if_expr(or.clone());
        // let <pat> = <src> in <target>
        let let_in = let_in(if_expr.clone().or(or.clone()));

        choice((if_expr, or, let_in))
    })
}

#[derive(Debug, PartialEq, Clone)]
enum Statement {
    Let { pat: Pattern, expr: Expr },
}

fn statement() -> impl SubParser<Statement> {
    let stmt = binding(expression())
        .then_ignore(just(Token::SemiColon))
        .map(|(pat, expr)| Statement::Let { pat, expr });

    stmt.then_ignore(end())
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

        #[allow(clippy::range_plus_one)]
        let token_stream = Stream::from_iter(end..end + 1, tokens.into_iter());

        expression().parse(token_stream)
    }

    fn parse_stmt(src: &str) -> Result<Statement, Vec<Simple<Token>>> {
        let tokens = lexer().parse(src).unwrap();
        let end = src.chars().count();

        #[allow(clippy::range_plus_one)]
        let token_stream = Stream::from_iter(end..end + 1, tokens.into_iter());

        statement().parse(token_stream)
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
        assert_eq!(parse_expr(src), Ok(Expr::ident("zero")));
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

        assert_eq!(parse_expr(src), Ok(Expr::Int(50)));
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
    fn parse_string_expr() {
        let src = r#"
        "hello kaffee!"
        "#;

        assert_eq!(parse_expr(src), Ok(Expr::Str("hello kaffee!".to_owned())));
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
    fn parse_tuple_expr() {
        let src = "(zero, 5, next)";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::Tuple(vec![
                Expr::ident("zero"),
                Expr::Int(5),
                Expr::ident("next"),
            ])),
        );
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
        );
    }

    #[test]
    fn parse_unit_expr() {
        let src = "()";
        assert_eq!(parse_expr(src), Ok(Expr::Tuple(vec![])));
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
    fn parse_list_expr() {
        let src = "[5 25]";

        assert_eq!(
            parse_expr(src),
            Ok(Expr::List(vec![Expr::Int(5), Expr::Int(25)]))
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
        );
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
        );
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
        );
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
        );
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
    fn parse_neg_expr() {
        let src = "-x";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::unary_with(UnOp::Negative, Expr::ident("x")))
        );
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
        );
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
        );
    }

    #[test]
    fn parse_division_expr() {
        let src = "x / 5";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::binop_with(
                BinOp::Divide,
                Expr::ident("x"),
                Expr::Int(5),
            ))
        );
    }

    #[test]
    fn parse_sum_expr() {
        let src = "5 + x";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::binop_with(BinOp::Sum, Expr::Int(5), Expr::ident("x")))
        );
    }

    #[test]
    fn parse_sub_expr() {
        let src = "5 - x";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::binop_with(BinOp::Sub, Expr::Int(5), Expr::ident("x")))
        );
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
        );
    }

    #[test]
    fn parse_arithmetic_in_tuple_expr() {
        let src = "(0, 0 + 1, 1 + 2)";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::Tuple(vec![
                Expr::Int(0),
                Expr::binop_with(BinOp::Sum, Expr::Int(0), Expr::Int(1)),
                Expr::binop_with(BinOp::Sum, Expr::Int(1), Expr::Int(2)),
            ]))
        );
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
    fn parse_less_expr() {
        let src = "x < y";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::binop_with(
                BinOp::Less,
                Expr::ident("x"),
                Expr::ident("y")
            )),
        );
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
        );
    }

    #[test]
    fn parse_repeated_ordering_expr() {
        let src = "(x < y < z)";
        assert!(parse_expr(src).is_err());
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
        );
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
        );
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
        );
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
        );
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
        );
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
        );
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
        );
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
    fn parse_if_expr() {
        let src = "if x < y then 0 else 1";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::if_with(
                Expr::binop_with(
                    BinOp::Less,
                    Expr::ident("x"),
                    Expr::ident("y")
                ),
                Expr::Int(0),
                Expr::Int(1),
            ))
        );
    }

    #[test]
    fn parse_if_deep_expr() {
        let src = r#"
if x < y then
    if x < 0 then
        -1
    else 0
else 1
        "#;

        assert_eq!(
            parse_expr(src),
            Ok(Expr::if_with(
                Expr::binop_with(
                    BinOp::Less,
                    Expr::ident("x"),
                    Expr::ident("y")
                ),
                Expr::if_with(
                    Expr::binop_with(
                        BinOp::Less,
                        Expr::ident("x"),
                        Expr::Int(0)
                    ),
                    Expr::unary_with(UnOp::Negative, Expr::Int(1)),
                    Expr::Int(0),
                ),
                Expr::Int(1),
            ))
        );
    }

    #[test]
    fn parse_if_with_call_cond() {
        let src = "if next x > 10 then 0 else 1";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::if_with(
                Expr::binop_with(
                    BinOp::Greater,
                    Expr::app(Expr::ident("next"), Expr::ident("x")),
                    Expr::Int(10),
                ),
                Expr::Int(0),
                Expr::Int(1),
            ))
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
    fn parse_let_in_expr() {
        let src = "let x = 5 in Some x";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::letin_with(
                Pattern::ident("x"),
                Expr::Int(5),
                Expr::app(Expr::ident("Some"), Expr::ident("x"))
            ))
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
    fn parse_let_stmt() {
        let src = "let x = 50;";
        assert_eq!(
            parse_stmt(src),
            Ok(Statement::Let {
                pat: Pattern::ident("x"),
                expr: Expr::Int(50),
            })
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
        );
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
        );
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
