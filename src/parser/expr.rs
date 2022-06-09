use chumsky::prelude::*;

#[cfg(test)]
use super::Name;

use super::{
    bind::{bind, Bind},
    common::{foldl_binops, from_module, grouping, tuple_like, RecordEntry},
    lex::{Op, Token},
    lookup,
    pattern::{pattern, Pattern},
    spanned, transfer_span, unspan, Spanned,
};

pub(super) trait ExprParser:
    Parser<Token, Spanned<Expr>, Error = Simple<Token>>
{
}
impl<T> ExprParser for T where
    T: Parser<Token, Spanned<Expr>, Error = Simple<Token>>
{
}

#[derive(Debug, PartialEq, Clone)]
pub enum UnOp {
    Negative,
}

#[derive(Debug, PartialEq, Clone)]
pub enum BinOp {
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
pub enum Expr {
    // Atoms
    Int(u64),
    Ident(String),
    Str(String),
    // Collections
    List(Vec<Spanned<Self>>),
    Tuple(Vec<Spanned<Self>>),
    Record(Vec<RecordEntry<Self>>),
    // Complex
    Unary(UnOp, Box<Spanned<Self>>),
    BinOp(BinOp, Box<Spanned<Self>>, Box<Spanned<Self>>),
    App(Box<Spanned<Self>>, Box<Spanned<Self>>),
    FromModule(Spanned<String>, Box<Spanned<Self>>),
    // Compound
    // if <cond> then <yes> else <no>
    If {
        cond: Box<Spanned<Self>>,
        yes: Box<Spanned<Self>>,
        // TODO: make optional?
        no: Box<Spanned<Self>>,
    },
    // let <bind> = <src> in <target>
    LetIn {
        bind: Spanned<Bind>,
        src: Box<Spanned<Self>>,
        target: Box<Spanned<Self>>,
    },
    // match <matched>
    //   | <pat> => <target>
    //   ...
    Match {
        matched: Box<Spanned<Self>>,
        branches: Vec<(Spanned<Pattern>, Spanned<Self>)>,
    },
}

impl Expr {
    #[cfg(test)]
    pub(super) fn ident(name: &str) -> Self {
        Self::Ident(name.to_owned())
    }

    #[cfg(test)]
    pub(super) fn string(s: &str) -> Self {
        Self::Str(s.to_owned())
    }

    pub(super) fn from_module(
        mod_name: Spanned<String>,
        x: Spanned<Self>,
    ) -> Self {
        Self::FromModule(mod_name, Box::new(x))
    }

    pub(super) fn app(f: Spanned<Self>, x: Spanned<Self>) -> Spanned<Self> {
        #[cfg(not(test))]
        let span = f.reach(&x);
        #[cfg(test)]
        let span = 0..0;
        spanned(Self::App(Box::new(f), Box::new(x)), span)
    }

    fn make_unary(token: Spanned<Token>, x: Spanned<Self>) -> Spanned<Self> {
        let op = lookup(&token).op_expect();
        let unary = match op {
            Op::Minus => UnOp::Negative,
            _ => unreachable!(),
        };

        Self::unary_with(transfer_span(token, unary), x)
    }

    pub(super) fn unary_with(
        op: Spanned<UnOp>,
        x: Spanned<Self>,
    ) -> Spanned<Self> {
        #[cfg(not(test))]
        let span = op.reach(&x);
        #[cfg(test)]
        let span = 0..0;

        spanned(Self::Unary(unspan(op), Box::new(x)), span)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn make_binop(
        token: Token,
        x: Spanned<Self>,
        y: Spanned<Self>,
    ) -> Spanned<Self> {
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

    pub(super) fn binop_with(
        op: BinOp,
        x: Spanned<Self>,
        y: Spanned<Self>,
    ) -> Spanned<Self> {
        #[cfg(not(test))]
        let span = x.reach(&y);
        #[cfg(test)]
        let span = 0..0;
        spanned(Self::BinOp(op, Box::new(x), Box::new(y)), span)
    }

    pub(super) fn if_with(
        cond: Spanned<Self>,
        then: Spanned<Self>,
        otherwise: Spanned<Self>,
    ) -> Self {
        Self::If {
            cond: Box::new(cond),
            yes: Box::new(then),
            no: Box::new(otherwise),
        }
    }

    pub(super) fn letin_with(
        bind: Spanned<Bind>,
        src: Spanned<Self>,
        target: Spanned<Self>,
    ) -> Self {
        Self::LetIn {
            bind,
            src: Box::new(src),
            target: Box::new(target),
        }
    }

    #[cfg(test)]
    pub(super) fn let_bound_in(pat: Pattern, src: Self, target: Self) -> Self {
        Self::letin_with(Bind::Bound(pat), src, target)
    }

    #[cfg(test)]
    pub(super) fn let_abstraction_in(
        name: Name,
        args: Vec<Pattern>,
        src: Self,
        target: Self,
    ) -> Self {
        Self::letin_with(Bind::Abstraction(name, args), src, target)
    }

    pub(super) fn match_with(
        matched: Spanned<Self>,
        branches: Vec<(Spanned<Pattern>, Spanned<Self>)>,
    ) -> Self {
        Self::Match {
            matched: Box::new(matched),
            branches,
        }
    }
}

// Parses tuple expression
fn tuple<P: ExprParser + Clone>(term: P) -> impl ExprParser + Clone {
    tuple_like(term, Expr::Tuple).map_with_span(spanned)
}

// Parses module subscription
fn imported<P: ExprParser + Clone + 'static>(
    term: P,
) -> impl ExprParser + Clone {
    from_module(term, Expr::from_module)
}

// Parses list expression
fn list<P: ExprParser + 'static>(term: P) -> impl ExprParser {
    recursive(|sublist| {
        let element = term.or(sublist);

        element
            .repeated()
            .delimited_by(just(Token::OpenBracket), just(Token::CloseBracket))
            .collect::<Vec<_>>()
            .map(Expr::List)
            .map_with_span(spanned)
    })
}

// Parses record expression
fn record<P: ExprParser>(term: P) -> impl ExprParser {
    let entry = select! {Token::Ident(field) => field}
        .map_with_span(spanned)
        .then_ignore(just(Token::Colon))
        .then(term);

    entry
        .separated_by(just(Token::Comma))
        .delimited_by(just(Token::OpenBrace), just(Token::CloseBrace))
        .map(Expr::Record)
        .map_with_span(spanned)
}

// Parses application `e e e`
//
// Returns expression 'as is' if none of applications were found
fn app<P: ExprParser + Clone>(term: P) -> impl ExprParser + Clone {
    term.clone().then(term.repeated()).foldl(Expr::app)
}

// Parses `- x` (for any amount of negations)
//
// If none negations were found just returns expression
fn unary<P: ExprParser + Clone>(term: P) -> impl ExprParser + Clone {
    just(Token::Op(Op::Minus))
        .map_with_span(spanned)
        .repeated()
        .then(term)
        .foldr(Expr::make_unary)
}

// Parses product expression `x * y` | `x / y`
// (for any amount of operators)
//
// If none operators were found just returns expression
fn product<P: ExprParser + Clone>(term: P) -> impl ExprParser + Clone {
    foldl_binops(
        term,
        just(Token::Op(Op::Product)).or(just(Token::Op(Op::Divide))),
        |x, (op, y)| Expr::make_binop(op, x, y),
    )
}

// Parses arithmetic expression `x + y` | `x - y`
// (for any amount of operators)
//
// If none operators were found just returns expression
fn arithmetic<P: ExprParser + Clone>(term: P) -> impl ExprParser + Clone {
    foldl_binops(
        term,
        just(Token::Op(Op::Plus)).or(just(Token::Op(Op::Minus))),
        |x, (op, y)| Expr::make_binop(op, x, y),
    )
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
    foldl_binops(term, just(Token::Op(Op::And)), |a, (op, b)| {
        Expr::make_binop(op, a, b)
    })
}

// Parses logic or `a || b` (for any amount of ||)
//
// If none of `||` were found just returns expression
fn or<P: ExprParser + Clone>(term: P) -> impl ExprParser + Clone {
    foldl_binops(term, just(Token::Op(Op::Or)), |a, (op, b)| {
        Expr::make_binop(op, a, b)
    })
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
            .map_with_span(spanned)
    })
}

// Parses let <pat> = <src> in <tgt>
fn let_in<P: ExprParser + Clone>(term: P) -> impl ExprParser + Clone {
    just(Token::Let)
        .ignore_then(bind())
        .then_ignore(just(Token::Is))
        .then(term.clone())
        .then_ignore(just(Token::In))
        .then(term)
        .map(|((bind, src), target)| Expr::letin_with(bind, src, target))
        .map_with_span(spanned)
}

fn match_expr<P: ExprParser + Clone>(term: P) -> impl ExprParser + Clone {
    just(Token::Match)
        .ignore_then(term.clone())
        .then(
            just(Token::Either)
                .ignore_then(pattern())
                .then_ignore(just(Token::MatchArrow))
                .then(term)
                .repeated()
                .at_least(1)
                .collect::<Vec<_>>(),
        )
        .map(|(matched, branches)| Expr::match_with(matched, branches))
        .map_with_span(spanned)
}

#[allow(clippy::let_and_return)]
pub(super) fn expression() -> impl ExprParser + Clone {
    recursive(|expr| {
        let literal = select! {
            // FIXME: this unwrap may panic (u64 is "finite")
            Token::Int(i) => Expr::Int(i.parse().unwrap()),
            Token::Ident(i) => Expr::Ident(i),
            Token::Str(s) => Expr::Str(s),
        };
        let literal = literal.map_with_span(spanned);

        // literal | grouping | tuple
        let atom = choice((
            literal,
            grouping(expr.clone()),
            tuple(expr.clone()),
            record(expr.clone()),
        ));
        #[cfg(debug_assertions)]
        let atom = atom.boxed();
        // list
        let list = list(atom.clone());
        let term = atom.or(list);
        #[cfg(debug_assertions)]
        let term = term.boxed();
        // module subscription (Mod.e)
        let from_module = imported(term.clone());
        // application (e e e)
        let app = app(from_module.or(term));
        // unary negation
        let unary = unary(app);
        // product (*, /)
        let product = product(unary);
        // arithmetic (+, -)
        let arithmetic = arithmetic(product);
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
        let and = and(element);
        // or
        let or = or(and);
        let complex = or;
        #[cfg(debug_assertions)]
        let complex = complex.boxed();
        // if <cond> then <yes> else <no>
        let if_expr = if_expr(complex.clone());
        let complex = complex.or(if_expr);
        #[cfg(debug_assertions)]
        let complex = complex.boxed();
        // let <pat> = <src> in <target>
        let let_in = let_in(complex.clone());
        let complex = complex.or(let_in);
        #[cfg(debug_assertions)]
        let complex = complex.boxed();
        // match
        let match_expr = match_expr(complex.clone());
        let complex = complex.or(match_expr);
        #[cfg(debug_assertions)]
        let complex = complex.boxed();

        complex
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::tests::tokens;

    fn parse_expr(src: &str) -> Result<Expr, Vec<Simple<Token>>> {
        let token_stream = tokens(src);

        expression().parse(token_stream)
    }

    #[test]
    fn parse_ident_expr() {
        let src = "zero";
        assert_eq!(parse_expr(src), Ok(Expr::ident("zero")));
    }

    #[test]
    fn parse_integer_expr() {
        let src = "50";

        assert_eq!(parse_expr(src), Ok(Expr::Int(50)));
    }

    #[test]
    fn parse_string_expr() {
        let src = r#"
        "hello kaffee!"
        "#;

        assert_eq!(parse_expr(src), Ok(Expr::Str("hello kaffee!".to_owned())));
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
    fn parse_list_expr() {
        let src = "[5 25]";

        assert_eq!(
            parse_expr(src),
            Ok(Expr::List(vec![Expr::Int(5), Expr::Int(25)]))
        );
    }

    #[test]
    fn parse_record_expr() {
        let src = "{x: 5, y: 8}";

        assert_eq!(
            parse_expr(src),
            Ok(Expr::Record(vec![
                ("x".to_owned(), Expr::Int(5)),
                ("y".to_owned(), Expr::Int(8)),
            ]))
        );
    }

    #[test]
    fn parse_nested_record_expr() {
        let src = "{yay: {x: 5, y: 5}, yoy: {x: 7, y: 0}}";

        assert_eq!(
            parse_expr(src),
            Ok(Expr::Record(vec![
                (
                    "yay".to_owned(),
                    Expr::Record(vec![
                        ("x".to_owned(), Expr::Int(5)),
                        ("y".to_owned(), Expr::Int(5)),
                    ])
                ),
                (
                    "yoy".to_owned(),
                    Expr::Record(vec![
                        ("x".to_owned(), Expr::Int(7)),
                        ("y".to_owned(), Expr::Int(0)),
                    ])
                ),
            ]))
        );
    }

    #[test]
    fn parse_record_from_module_expr() {
        let src = "Date.{day: 32, month: 44}";

        assert_eq!(
            parse_expr(src),
            Ok(Expr::from_module(
                "Date".to_owned(),
                Expr::Record(vec![
                    ("day".to_owned(), Expr::Int(32)),
                    ("month".to_owned(), Expr::Int(44)),
                ]),
            ))
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
    fn parse_call_deep_expr() {
        let src = "chain next next 5";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::app(
                Expr::app(
                    Expr::app(Expr::ident("chain"), Expr::ident("next")),
                    Expr::ident("next"),
                ),
                Expr::Int(5),
            ))
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
    fn parse_from_module_ident_expr() {
        let src = "Nat.succ 5";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::app(
                Expr::from_module("Nat".to_owned(), Expr::ident("succ")),
                Expr::Int(5)
            ))
        );
    }

    #[test]
    fn parse_from_module_complex_expr() {
        let src = "List.(map double xs)";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::from_module(
                "List".to_owned(),
                Expr::app(
                    Expr::app(Expr::ident("map"), Expr::ident("double")),
                    Expr::ident("xs"),
                )
            ))
        );
    }

    #[test]
    fn parse_from_module_complex_nested_expr() {
        let src = "List.(map Int.double xs)";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::from_module(
                "List".to_owned(),
                Expr::app(
                    Expr::app(
                        Expr::ident("map"),
                        Expr::from_module(
                            "Int".to_owned(),
                            Expr::ident("double")
                        )
                    ),
                    Expr::ident("xs"),
                )
            ))
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
    fn parse_let_in_expr() {
        let src = "let x = 5 in Some x";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::let_bound_in(
                Pattern::ident("x"),
                Expr::Int(5),
                Expr::app(Expr::ident("Some"), Expr::ident("x"))
            ))
        );
    }

    #[test]
    fn parse_let_in_tuple_expr() {
        let src = "let (x, y) = (5, 10) in x + y";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::let_bound_in(
                Pattern::Tuple(vec![Pattern::ident("x"), Pattern::ident("y")]),
                Expr::Tuple(vec![Expr::Int(5), Expr::Int(10)]),
                Expr::binop_with(
                    BinOp::Sum,
                    Expr::ident("x"),
                    Expr::ident("y"),
                )
            )),
        );
    }

    #[test]
    fn parse_let_in_tuple_deep_expr() {
        let src = "let (x, (y, z)) = pack in x + y + z";
        assert_eq!(
            parse_expr(src),
            Ok(Expr::let_bound_in(
                Pattern::Tuple(vec![
                    Pattern::ident("x"),
                    Pattern::Tuple(vec![
                        Pattern::ident("y"),
                        Pattern::ident("z"),
                    ]),
                ]),
                Expr::ident("pack"),
                Expr::binop_with(
                    BinOp::Sum,
                    Expr::binop_with(
                        BinOp::Sum,
                        Expr::ident("x"),
                        Expr::ident("y"),
                    ),
                    Expr::ident("z"),
                )
            )),
        );
    }

    #[test]
    fn parse_let_in_abstraction_expr() {
        let src = r#"
let chain f g x = g (f x)
    in
chain next next 5
        "#;

        assert_eq!(
            parse_expr(src),
            Ok(Expr::let_abstraction_in(
                Name::plain("chain"),
                vec![
                    Pattern::ident("f"),
                    Pattern::ident("g"),
                    Pattern::ident("x"),
                ],
                Expr::app(
                    Expr::ident("g"),
                    Expr::app(Expr::ident("f"), Expr::ident("x"))
                ),
                Expr::app(
                    Expr::app(
                        Expr::app(Expr::ident("chain"), Expr::ident("next")),
                        Expr::ident("next"),
                    ),
                    Expr::Int(5),
                ),
            ))
        );
    }

    #[test]
    fn parse_let_in_abstraction_with_pattern_expr() {
        let src = r#"
let add (x, y) = x + y
    in
add (5, 10)
        "#;
        assert_eq!(
            parse_expr(src),
            Ok(Expr::let_abstraction_in(
                Name::plain("add"),
                vec![Pattern::Tuple(vec![
                    Pattern::ident("x"),
                    Pattern::ident("y"),
                ]),],
                Expr::binop_with(
                    BinOp::Sum,
                    Expr::ident("x"),
                    Expr::ident("y"),
                ),
                Expr::app(
                    Expr::ident("add"),
                    Expr::Tuple(vec![Expr::Int(5), Expr::Int(10)]),
                ),
            ))
        );
    }

    #[test]
    fn parse_let_in_abstraction_with_complex_pattern_expr() {
        // Tbh, it's unclear whether that code actually correct.
        //
        // In following code, we can see that match is practically valid.
        // Though if we de-sugar this to match (and we probably should),
        // we can see that `Some y` is not exhaustive match, considering
        // that we probably match `option`, nor is `10` pattern.
        //
        // But from parser pespective code like `let 5 = 5` is valid.
        // And considering the fact that let-binding for abstraction does
        // use pattern() in slightly different way from `match`
        // we want to be sure that even this code can be *correctly* parsed.
        let src = r#"
let f x (Some y) (One z | Two z) (a, Some b) 10 _ = x + y + z + a + b
    in
f 5 (Some 6) (One 8) (5, 10) 10 "whatever"
        "#;

        let args = vec![
            Pattern::ident("x"),
            Pattern::variant_with("Some", Pattern::ident("y")),
            Pattern::choice(
                Pattern::variant_with("One", Pattern::ident("z")),
                Pattern::variant_with("Two", Pattern::ident("z")),
            ),
            Pattern::Tuple(vec![
                Pattern::ident("a"),
                Pattern::variant_with("Some", Pattern::ident("b")),
            ]),
            Pattern::Int(10),
            Pattern::Wildcard,
        ];

        let expr = Expr::binop_with(
            BinOp::Sum,
            Expr::binop_with(
                BinOp::Sum,
                Expr::binop_with(
                    BinOp::Sum,
                    Expr::binop_with(
                        BinOp::Sum,
                        Expr::ident("x"),
                        Expr::ident("y"),
                    ),
                    Expr::ident("z"),
                ),
                Expr::ident("a"),
            ),
            Expr::ident("b"),
        );

        let target = Expr::app(
            Expr::app(
                Expr::app(
                    Expr::app(
                        Expr::app(
                            Expr::app(Expr::ident("f"), Expr::Int(5)),
                            Expr::app(Expr::ident("Some"), Expr::Int(6)),
                        ),
                        Expr::app(Expr::ident("One"), Expr::Int(8)),
                    ),
                    Expr::Tuple(vec![Expr::Int(5), Expr::Int(10)]),
                ),
                Expr::Int(10),
            ),
            Expr::string("whatever"),
        );

        assert_eq!(
            parse_expr(src),
            Ok(Expr::let_abstraction_in(
                Name::plain("f"),
                args,
                expr,
                target
            ))
        );
    }

    #[test]
    fn parse_let_in_abstraction_with_spell_expr() {
        let src = r#"
let @plus x y = Int.plus
in x + y
        "#;

        assert_eq!(
            parse_expr(src),
            Ok(Expr::let_abstraction_in(
                Name::spell("plus"),
                vec![Pattern::ident("x"), Pattern::ident("y"),],
                Expr::from_module("Int".to_owned(), Expr::ident("plus")),
                Expr::binop_with(
                    BinOp::Sum,
                    Expr::ident("x"),
                    Expr::ident("y"),
                )
            ))
        );
    }

    #[test]
    fn parse_match_variant_expr() {
        let src = r#"
match opt
| Some x => f x
| None => None
        "#;

        assert_eq!(
            parse_expr(src),
            Ok(Expr::match_with(
                Expr::ident("opt"),
                vec![
                    (
                        Pattern::variant_with("Some", Pattern::ident("x")),
                        Expr::app(Expr::ident("f"), Expr::ident("x")),
                    ),
                    (Pattern::ident("None"), Expr::ident("None"),),
                ],
            ))
        );
    }

    #[test]
    fn parse_match_tuple_expr() {
        let src = r#"
match t
| (x, 0) => -x
| (0, y) => -y
| (x, y) => x + y

        "#;

        assert_eq!(
            parse_expr(src),
            Ok(Expr::match_with(
                Expr::ident("t"),
                vec![
                    (
                        Pattern::Tuple(vec![
                            Pattern::ident("x"),
                            Pattern::Int(0)
                        ]),
                        Expr::unary_with(UnOp::Negative, Expr::ident("x")),
                    ),
                    (
                        Pattern::Tuple(vec![
                            Pattern::Int(0),
                            Pattern::ident("y")
                        ]),
                        Expr::unary_with(UnOp::Negative, Expr::ident("y")),
                    ),
                    (
                        Pattern::Tuple(vec![
                            Pattern::ident("x"),
                            Pattern::ident("y"),
                        ]),
                        Expr::binop_with(
                            BinOp::Sum,
                            Expr::ident("x"),
                            Expr::ident("y"),
                        )
                    ),
                ],
            ))
        );
    }

    #[test]
    fn parse_match_nested_expr() {
        let src = r#"
match
    (match t | (x, y) => x + y)
    | 0 => None
    | any => Some any
        "#;

        assert_eq!(
            parse_expr(src),
            Ok(Expr::match_with(
                Expr::match_with(
                    Expr::ident("t"),
                    vec![(
                        Pattern::Tuple(vec![
                            Pattern::ident("x"),
                            Pattern::ident("y"),
                        ]),
                        Expr::binop_with(
                            BinOp::Sum,
                            Expr::ident("x"),
                            Expr::ident("y"),
                        )
                    )],
                ),
                vec![
                    (Pattern::Int(0), Expr::ident("None")),
                    (
                        Pattern::ident("any"),
                        Expr::app(Expr::ident("Some"), Expr::ident("any"))
                    ),
                ]
            ))
        );
    }

    #[test]
    fn parse_match_wildcard_expr() {
        let src = r#"
match e
| NotFound => "not found"
| PermissionDenied => "permission error"
| _ => "unexpected filesystem error"
        "#;

        assert_eq!(
            parse_expr(src),
            Ok(Expr::match_with(
                Expr::ident("e"),
                vec![
                    (Pattern::ident("NotFound"), Expr::string("not found")),
                    (
                        Pattern::ident("PermissionDenied"),
                        Expr::string("permission error")
                    ),
                    (
                        Pattern::Wildcard,
                        Expr::string("unexpected filesystem error")
                    ),
                ],
            ))
        );
    }

    #[test]
    fn parse_match_choice_expr() {
        let src = r#"
match element
| Str _ | Bytes _ => "string-like"
| _ => "atom-like" 
        "#;

        assert_eq!(
            parse_expr(src),
            Ok(Expr::match_with(
                Expr::ident("element"),
                vec![
                    (
                        Pattern::choice(
                            Pattern::variant_with("Str", Pattern::Wildcard),
                            Pattern::variant_with("Bytes", Pattern::Wildcard),
                        ),
                        Expr::string("string-like"),
                    ),
                    (Pattern::Wildcard, Expr::string("atom-like"))
                ]
            ))
        );
    }
}
