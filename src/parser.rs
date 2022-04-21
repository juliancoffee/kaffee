#![warn(clippy::pedantic)]
use chumsky::{prelude::*, Stream};

// aliases
trait ExprParser: Parser<Token, Spanned<Expr>, Error = Simple<Token>> {}
impl<T> ExprParser for T where
    T: Parser<Token, Spanned<Expr>, Error = Simple<Token>>
{
}

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
    item: T,
    span: Span,
}

#[cfg(not(test))]
impl<T> Spanned<T> {
    fn new(item: T, span: Span) -> Self {
        Self { item, span }
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
    // Misc
    Doc(String),
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
    Wildcard,
    // Binders
    Of,
    Is,
    In,
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
    fn op_expect(&self) -> &Op {
        match self {
            Self::Op(op) => op,
            token => panic!("expected Token::Op(_), got {token:?}"),
        }
    }
}

fn lexer() -> impl Parser<char, Vec<(Token, Span)>, Error = Simple<char>> {
    let row_comment = just("//")
        .ignore_then(take_until(text::newline()))
        .map(|(line, ())| line.into_iter().skip_while(|c| c.is_whitespace()))
        .collect::<String>();

    let doc = row_comment
        .repeated()
        .at_least(1)
        .collect::<String>()
        .map(Token::Doc);

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
        just('_').to(Token::Wildcard),
        just('=').to(Token::Is),
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

    let token = choice((doc, atom, op, ctrl, marked, word));

    token
        .map_with_span(|tok, span| (tok, span))
        .padded()
        .repeated()
        .then_ignore(end())
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
    fn ident(name: &str) -> Self {
        Self::Ident(name.to_owned())
    }

    fn variant(name: String, inner: Spanned<Self>) -> Self {
        Self::Variant(name, Box::new(inner))
    }

    #[cfg(test)]
    fn variant_with(name: &str, inner: Spanned<Self>) -> Self {
        Self::Variant(name.to_owned(), Box::new(inner))
    }

    fn choice(a: Spanned<Self>, b: Spanned<Self>) -> Spanned<Self> {
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
fn atomic_pattern<P: SubParser<Spanned<Pattern>> + Clone>(
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
fn complex_pattern<P: SubParser<Spanned<Pattern>> + Clone + 'static>(
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
fn pattern() -> impl SubParser<Spanned<Pattern>> + Clone {
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

#[derive(Debug, PartialEq, Clone)]
pub enum Bind {
    Abstraction(String, Vec<Spanned<Pattern>>),
    Bound(Spanned<Pattern>),
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
    // Complex
    Unary(UnOp, Box<Spanned<Self>>),
    BinOp(BinOp, Box<Spanned<Self>>, Box<Spanned<Self>>),
    App(Box<Spanned<Self>>, Box<Spanned<Self>>),
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
    fn ident(name: &str) -> Self {
        Self::Ident(name.to_owned())
    }

    #[cfg(test)]
    fn string(s: &str) -> Self {
        Self::Str(s.to_owned())
    }

    fn app(f: Spanned<Self>, x: Spanned<Self>) -> Spanned<Self> {
        #[cfg(not(test))]
        let span = f.reach(&x);
        #[cfg(test)]
        let span = 0..0;
        spanned(Self::App(Box::new(f), Box::new(x)), span)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn make_unary(token: Spanned<Token>, x: Spanned<Self>) -> Spanned<Self> {
        let op = lookup(&token).op_expect();
        let unary = match op {
            Op::Minus => UnOp::Negative,
            _ => unreachable!(),
        };

        Self::unary_with(transfer_span(token, unary), x)
    }

    fn unary_with(op: Spanned<UnOp>, x: Spanned<Self>) -> Spanned<Self> {
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

    fn binop_with(
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

    fn if_with(
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

    fn letin_with(
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
    fn let_bound_in(pat: Pattern, src: Self, target: Self) -> Self {
        Self::letin_with(Bind::Bound(pat), src, target)
    }

    #[cfg(test)]
    fn let_abstraction_in(
        name: &str,
        args: Vec<Pattern>,
        src: Self,
        target: Self,
    ) -> Self {
        Self::letin_with(Bind::Abstraction(name.to_owned(), args), src, target)
    }

    fn match_with(
        matched: Spanned<Self>,
        branches: Vec<(Spanned<Pattern>, Spanned<Self>)>,
    ) -> Self {
        Self::Match {
            matched: Box::new(matched),
            branches,
        }
    }
}

// Parses `let <bind> = <expr>` binding using parser for bind
fn let_binding_maker<B, E>(
    bind: B,
    expr: E,
) -> impl SubParser<(Spanned<Bind>, Spanned<Expr>)> + Clone
where
    B: SubParser<Spanned<Bind>> + Clone,
    E: ExprParser + Clone,
{
    just(Token::Let)
        .ignore_then(bind)
        .then_ignore(just(Token::Is))
        .then(expr)
}

// Parses `let <declaration> = <src>` binding
//
// Results in `pat` and `expr`
fn let_binding<P: ExprParser + Clone>(
    expr: P,
) -> impl SubParser<(Spanned<Bind>, Spanned<Expr>)> + Clone {
    // used for things like
    // `let x = 5`
    // `let (x, y) = (5, 10);
    let bound = pattern().map(Bind::Bound).map_with_span(spanned);

    // used for "function" declaration
    let abstraction = select! { Token::Ident(i) => i}
        .then(
            atomic_pattern(pattern())
                .or(complex_pattern(pattern()).delimited_by(
                    just(Token::OpenParen),
                    just(Token::CloseParen),
                ))
                .repeated()
                .at_least(1)
                .collect::<Vec<_>>(),
        )
        .map(|(name, args)| Bind::Abstraction(name, args))
        .map_with_span(spanned);

    #[cfg(debug_assertions)]
    let abstraction = abstraction.boxed();

    let_binding_maker(abstraction.or(bound), expr)
}

// Parses tuple expression
fn tuple<P: ExprParser + Clone>(term: P) -> impl ExprParser + Clone {
    tuple_like(term, Expr::Tuple).map_with_span(spanned)
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
            .map_with_span(spanned)
    })
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
    let_binding(term.clone())
        .then_ignore(just(Token::In))
        .then(term)
        .map(|((pat, src), target)| Expr::letin_with(pat, src, target))
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
fn expression() -> impl ExprParser + Clone {
    recursive(|expr| {
        let literal = select! {
            // FIXME: this unwrap may panic (u64 is "finite")
            Token::Int(i) => Expr::Int(i.parse().unwrap()),
            Token::Ident(i) => Expr::Ident(i),
            Token::Str(s) => Expr::Str(s),
        };
        let literal = literal.map_with_span(spanned);

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
        let complex = or.clone();
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

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Ident(String),
    Var(String),
    Product(Vec<Spanned<Self>>),
}

impl Type {
    #[cfg(test)]
    fn ident(name: &str) -> Self {
        Self::Ident(name.to_owned())
    }

    #[cfg(test)]
    fn var(name: &str) -> Self {
        Self::Var(name.to_owned())
    }
}

fn product_type<P: SubParser<Spanned<Type>>>(
    item: P,
) -> impl SubParser<Spanned<Type>> {
    item.separated_by(just(Token::Op(Op::Product)))
        .at_least(2)
        .collect::<Vec<_>>()
        .map(Type::Product)
        .map_with_span(spanned)
}

fn ty() -> impl SubParser<Spanned<Type>> {
    let type_var = select! {
        Token::TypeParameter(p) => Type::Var(p),
    };
    let type_var = type_var.map_with_span(spanned);

    let ident = select! {
        Token::Ident(i) => Type::Ident(i),
    };
    let ident = ident.map_with_span(spanned);

    let product = product_type(ident.or(type_var));

    product.or(ident)
}

#[derive(Debug, PartialEq, Clone)]
pub enum TypeBind {
    Bound(Spanned<String>),
    Abstraction(Spanned<String>, Vec<Spanned<String>>),
}

impl TypeBind {
    #[cfg(test)]
    fn bound(name: &str) -> Self {
        Self::Bound(name.to_owned())
    }

    #[cfg(test)]
    fn abstraction(name: &str, vars: Vec<&str>) -> Self {
        Self::Abstraction(
            name.to_owned(),
            vars.into_iter().map(str::to_owned).collect(),
        )
    }
}

fn type_bind() -> impl SubParser<TypeBind> {
    let bound = select! {Token::Ident(i) => i}
        .map_with_span(spanned)
        .map(TypeBind::Bound);

    let abstraction = tuple_like(
        select! {Token::TypeParameter(p) => p}.map_with_span(spanned),
        std::convert::identity,
    )
    .then(select! {Token::Ident(name) => name}.map_with_span(spanned))
    .map(|(vars, name)| TypeBind::Abstraction(name, vars));

    bound.or(abstraction)
}

#[derive(Debug, PartialEq, Clone)]
pub enum TypeDef {
    Alias(TypeBind, Spanned<Type>),
}

#[allow(clippy::let_and_return)]
fn typedef() -> impl SubParser<TypeDef> {
    let alias = just(Token::Type)
        .ignore_then(type_bind())
        .then_ignore(just(Token::Is))
        .then(ty())
        .then_ignore(just(Token::Semicolon))
        .map(|(bind, ty)| TypeDef::Alias(bind, ty));

    alias
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    Let {
        bind: Spanned<Bind>,
        expr: Spanned<Expr>,
    },
    TypeDef(TypeDef),
    ModuleStruct(String, Vec<Spanned<Self>>),
    Commented(Spanned<String>, Box<Spanned<Self>>),
}

impl Statement {
    #[cfg(test)]
    fn let_bound(pat: Pattern, expr: Expr) -> Self {
        Self::Let {
            bind: Bind::Bound(pat),
            expr,
        }
    }

    #[cfg(test)]
    fn let_abstraction(name: &str, args: Vec<Pattern>, expr: Expr) -> Self {
        Self::Let {
            bind: Bind::Abstraction(name.to_owned(), args),
            expr,
        }
    }

    #[cfg(test)]
    fn module_struct(name: &str, defs: Vec<Self>) -> Self {
        Self::ModuleStruct(name.to_owned(), defs)
    }

    fn with_doc(docstring: Spanned<String>, stmt: Spanned<Self>) -> Self {
        Self::Commented(docstring, Box::new(stmt))
    }
}

fn let_stmt() -> impl SubParser<Spanned<Statement>> {
    let_binding(expression())
        .then_ignore(just(Token::Semicolon))
        .map(|(bind, expr)| Statement::Let { bind, expr })
        .map_with_span(spanned)
}

fn module<S: SubParser<Spanned<Statement>>>(
    stmt: S,
) -> impl SubParser<Spanned<Statement>> {
    just(Token::Module)
        .ignore_then(select! {Token::Ident(i) => i})
        .then_ignore(just(Token::Is))
        .then_ignore(just(Token::Struct))
        .then(stmt.repeated())
        .then_ignore(just(Token::End))
        .map(|(name, defs)| Statement::ModuleStruct(name, defs))
        .map_with_span(spanned)
}

fn commented<S: SubParser<Spanned<Statement>>>(
    stmt: S,
) -> impl SubParser<Spanned<Statement>> {
    select! {Token::Doc(doc) => doc}
        .map_with_span(spanned)
        .then(stmt)
        .map(|(doc, stmt)| Statement::with_doc(doc, stmt))
        .map_with_span(spanned)
}

fn statement() -> impl SubParser<Spanned<Statement>> {
    recursive(|stmt| {
        let let_stmt = let_stmt();
        let module = module(stmt.clone());
        let commented = commented(stmt);
        let typedef = typedef().map(Statement::TypeDef).map_with_span(spanned);

        choice((commented, let_stmt, module, typedef))
    })
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn remove_spans<X, E>(tt: Result<Vec<(X, Span)>, E>) -> Result<Vec<X>, E> {
        tt.map(|tt| tt.into_iter().map(|(t, _s)| t).collect())
    }

    // This could be a function, but I don't want to mess with types
    macro_rules! tokenize {
        ($src:expr) => {{
            let tokens = lexer().parse($src).unwrap();
            let end = $src.chars().count();

            #[allow(clippy::range_plus_one)]
            let token_stream =
                Stream::from_iter(end..end + 1, tokens.into_iter());

            token_stream
        }};
    }

    fn parse_pattern(src: &str) -> Result<Pattern, Vec<Simple<Token>>> {
        let token_stream = tokenize!(src);

        pattern().parse(token_stream)
    }

    fn parse_expr(src: &str) -> Result<Expr, Vec<Simple<Token>>> {
        let token_stream = tokenize!(src);

        expression().parse(token_stream)
    }

    fn parse_stmt(src: &str) -> Result<Statement, Vec<Simple<Token>>> {
        let token_stream = tokenize!(src);

        statement().parse(token_stream)
    }

    /* * * * * * * *
     * Expressions *
     * * * * * * * */

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
                "chain",
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
                "add",
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
            Ok(Expr::let_abstraction_in("f", args, expr, target))
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

    /* * * * * * * *
     * Patterns    *
     * * * * * * * */
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

    /* * * * * * * *
     * Statements  *
     * * * * * * * */
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
    fn parse_let_stmt() {
        let src = "let x = 50;";
        assert_eq!(
            parse_stmt(src),
            Ok(Statement::let_bound(Pattern::ident("x"), Expr::Int(50)))
        );
    }

    #[test]
    fn parse_let_unit_stmt() {
        let src = r#"
let () = print "wow";
        "#;

        assert_eq!(
            parse_stmt(src),
            Ok(Statement::let_bound(
                Pattern::Tuple(vec![]),
                Expr::app(Expr::ident("print"), Expr::string("wow")),
            ))
        );
    }

    #[test]
    fn parse_let_discard_stmt() {
        let src = r#"
let _ = rm "tmp.kf";
        "#;

        assert_eq!(
            parse_stmt(src),
            Ok(Statement::let_bound(
                Pattern::Wildcard,
                Expr::app(Expr::ident("rm"), Expr::string("tmp.kf")),
            ))
        );
    }

    #[test]
    fn parse_let_abstraction_stmt() {
        let src = "let neg x = - x;";

        assert_eq!(
            parse_stmt(src),
            Ok(Statement::let_abstraction(
                "neg",
                vec![Pattern::ident("x")],
                Expr::unary_with(UnOp::Negative, Expr::ident("x"))
            ))
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
    fn parse_type_alias_stmt() {
        let src = "type idx = nat;";

        assert_eq!(
            parse_stmt(src),
            Ok(Statement::TypeDef(TypeDef::Alias(
                TypeBind::bound("idx"),
                Type::ident("nat"),
            )))
        );
    }

    #[test]
    fn parse_type_alias_tuple_stmt() {
        let src = "type point = int * int;";

        assert_eq!(
            parse_stmt(src),
            Ok(Statement::TypeDef(TypeDef::Alias(
                TypeBind::bound("point"),
                Type::Product(vec![Type::ident("int"), Type::ident("int")]),
            )))
        );
    }

    #[test]
    fn tokenize_type_alias_generic_tuple_stmt() {
        let src = "type ('a, 'b) pair = 'a * 'b;";
        assert_eq!(
            remove_spans(lexer().parse(src)),
            Ok(vec![
                Token::Type,
                Token::OpenParen,
                Token::TypeParameter("a".to_owned()),
                Token::Comma,
                Token::TypeParameter("b".to_owned()),
                Token::CloseParen,
                Token::Ident("pair".to_owned()),
                Token::Is,
                Token::TypeParameter("a".to_owned()),
                Token::Op(Op::Product),
                Token::TypeParameter("b".to_owned()),
                Token::Semicolon,
            ])
        );
    }

    #[test]
    fn parse_type_alias_generic_tuple_stmt() {
        let src = "type ('a, 'b) pair = 'a * 'b;";
        assert_eq!(
            parse_stmt(src),
            Ok(Statement::TypeDef(TypeDef::Alias(
                TypeBind::abstraction("pair", vec!["a", "b"]),
                Type::Product(vec![Type::var("a"), Type::var("b")]),
            )))
        );
    }

    #[test]
    fn tokenize_type_enum_stmt() {
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
                Token::Semicolon,
            ])
        );
    }

    #[test]
    fn tokenize_type_enum_tuple_stmt() {
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
    fn parse_module_stmt() {
        let src = r#"
module Ops = struct
    let add x y = x + y;
    let sub x y = x - y;
end
        "#;

        assert_eq!(
            parse_stmt(src),
            Ok(Statement::module_struct(
                "Ops",
                vec![
                    Statement::let_abstraction(
                        "add",
                        vec![Pattern::ident("x"), Pattern::ident("y")],
                        Expr::binop_with(
                            BinOp::Sum,
                            Expr::ident("x"),
                            Expr::ident("y"),
                        )
                    ),
                    Statement::let_abstraction(
                        "sub",
                        vec![Pattern::ident("x"), Pattern::ident("y")],
                        Expr::binop_with(
                            BinOp::Sub,
                            Expr::ident("x"),
                            Expr::ident("y"),
                        )
                    ),
                ]
            ))
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

    #[test]
    fn parse_commented_stmt() {
        let src = r#"
// (+) operator wrapper
let add x y = x + y;
        "#;

        assert_eq!(
            parse_stmt(src),
            Ok(Statement::with_doc(
                "(+) operator wrapper".to_owned(),
                Statement::let_abstraction(
                    "add",
                    vec![Pattern::ident("x"), Pattern::ident("y")],
                    Expr::binop_with(
                        BinOp::Sum,
                        Expr::ident("x"),
                        Expr::ident("y"),
                    )
                ),
            ))
        );
    }
}
